"""CPU boundary/failure tests: importing these tests must not import torch."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from kvolt_runtime.backend.cache import pack_cache, pack_initial, partition_lengths, validate_packed_cache
from kvolt_runtime.backend.decode import decode_one
from kvolt_runtime.contracts import CacheFormat, CacheHandle, CacheMetadata
from kvolt_runtime.errors import BackendError, CacheStateError


class Device:
    type = 'cuda'
    def __str__(self): return 'cuda:0'


class Tensor:
    device = Device()
    def __init__(self, shape, dtype='torch.float16', storage=None):
        self.shape, self.dtype, self.storage = shape, dtype, storage or object()
    def __getitem__(self, indices):
        shape = list(self.shape)
        for axis, index in enumerate(indices):
            if isinstance(index, slice):
                shape[axis] = len(range(*index.indices(shape[axis])))
        return Tensor(tuple(shape), self.dtype, self.storage)
    def transpose(self, a, b):
        shape = list(self.shape); shape[a], shape[b] = shape[b], shape[a]
        return Tensor(tuple(shape), self.dtype, self.storage)
    def contiguous(self): return self
    def clone(self): return Tensor(self.shape, self.dtype)
    def all(self): return self
    def item(self): return 17
    def argmax(self, **kwargs): return self


def raw(length): return tuple((Tensor((1, 8, length, 128)), Tensor((1, 8, length, 128))) for _ in range(32))


def quantizer(data, group, bits):
    assert bits == 2
    a, b, c, d = data.shape
    return Tensor((a,b,c,d//16), 'torch.int32'), Tensor((a,b,c,d//group)), Tensor((a,b,c,d//group))


@pytest.mark.parametrize('length', [127,128,129,255,256,257,1024,1025,8192])
@pytest.mark.parametrize('group', [8,16,32,64,128])
def test_all_native_shapes_and_independent_tail_storage(length, group):
    source = raw(length)
    packed = pack_cache(source, length, group, quantizer=quantizer)
    assert validate_packed_cache(packed, length, group) == 'cuda:0'
    kp, kr, vp, vr = partition_lengths(length)
    assert kp + kr == vp + vr == length
    assert kp % 128 == 0 and kr < 128 and vr <= 128
    assert packed[0][0] is None if kp == 0 else packed[0][0].shape == (1,8,128,kp//16)
    for before, row in zip(source, packed):
        assert row[8] == length
        if kr: assert row[1].storage is not before[0].storage
        else: assert row[1] is None
        assert row[5].storage is not before[1].storage


def test_every_raw_layer_checked_before_first_kernel():
    source = list(raw(129)); source[-1][1].dtype = 'torch.float32'
    calls = []
    with pytest.raises(BackendError):
        pack_cache(source, 129, 8, quantizer=lambda *args: calls.append(args))
    assert calls == []


def test_partial_quant_failure_does_not_publish_or_change_settings():
    source = raw(129)
    cache = CacheHandle(CacheMetadata('one',129,'cuda:0',CacheFormat.RAW),source)
    attention = SimpleNamespace(group_size=None, config=SimpleNamespace(group_size=None))
    model = SimpleNamespace(config=SimpleNamespace(group_size=None), model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]*32))
    backend = SimpleNamespace(model=model, _require_active=lambda handle, fmt: handle.require(fmt), torch=SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=lambda:None)))
    count = []
    def fail(*args):
        count.append(1)
        if len(count) == 3: raise RuntimeError('injected kernel failure')
        return quantizer(*args)
    with pytest.raises(BackendError): pack_initial(backend,cache,8,quantizer=fail)
    assert cache.format is CacheFormat.RAW and cache.require(CacheFormat.RAW) is source
    assert attention.group_size is None and model.config.group_size is None


def test_gpu_completion_failure_keeps_raw_handle():
    source = raw(129)
    cache = CacheHandle(CacheMetadata('one',129,'cuda:0',CacheFormat.RAW),source)
    def fail(): raise RuntimeError('GPU error')
    backend = SimpleNamespace(_require_active=lambda h,f:h.require(f), torch=SimpleNamespace(inference_mode=nullcontext,cuda=SimpleNamespace(synchronize=fail)))
    with pytest.raises(BackendError): pack_initial(backend,cache,32,quantizer=quantizer)
    assert cache.require(CacheFormat.RAW) is source


def test_reject_invalid_group_before_kernel():
    for group in [True, 4, 256, None]:
        with pytest.raises(BackendError): pack_cache(raw(128),128,group,quantizer=quantizer)


def test_packed_validation_rejects_last_layer_wrong_length_or_metadata():
    packed = list(pack_cache(raw(129),129,8,quantizer=quantizer))
    row = list(packed[-1]); row[8] = 128; packed[-1] = tuple(row)
    with pytest.raises(BackendError): validate_packed_cache(packed,129,8)
    packed = list(pack_cache(raw(129),129,8,quantizer=quantizer))
    packed[-1][2].shape = (1,8,128,8)
    with pytest.raises(BackendError): validate_packed_cache(packed,129,8)


def test_decode_position_length_completion_and_no_partial_commit():
    cache = CacheHandle(CacheMetadata('one',129,'cuda:0',CacheFormat.RAW),raw(129))
    cache.commit_packed(CacheMetadata('one',129,'cuda:0',CacheFormat.PACKED,8),pack_cache(raw(129),129,8,quantizer=quantizer))
    events=[]
    class Torch:
        long='torch.int64'
        inference_mode=staticmethod(nullcontext)
        def tensor(self, ids, **kwargs): return ids
        def isfinite(self, value): return SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:True))
        cuda=SimpleNamespace(synchronize=lambda:events.append('sync'))
    class Model:
        def __call__(self, **kwargs):
            assert kwargs['input_ids'] == [[10]] and kwargs['position_ids'] == [[129]]
            assert kwargs['past_key_values'] is cache.require(CacheFormat.PACKED)
            return SimpleNamespace(logits=Tensor((1,1,128256)),past_key_values=pack_cache(raw(130),130,8,quantizer=quantizer))
    backend=SimpleNamespace(_require_active=lambda h,f:h.require(f),torch=Torch(),model=Model(),_clock=lambda:events.append('clock') or 123)
    token=decode_one(backend,cache,10)
    assert token.token_id==17 and token.ready_ns==123 and token.cache is cache
    assert cache.metadata.length==130 and events==['sync','clock']
    with pytest.raises(BackendError): decode_one(backend,cache,True)


@pytest.mark.parametrize('clock,token', [(-1,17),(123,128256)])
def test_invalid_decode_result_does_not_commit(clock,token):
    source=pack_cache(raw(129),129,8,quantizer=quantizer)
    cache=CacheHandle(CacheMetadata('one',129,'cuda:0',CacheFormat.RAW),raw(129))
    cache.commit_packed(CacheMetadata('one',129,'cuda:0',CacheFormat.PACKED,8),source)
    class BadLogits(Tensor):
        def __getitem__(self,index): return self
        def item(self): return token
    torch=SimpleNamespace(long='long',inference_mode=nullcontext,tensor=lambda x,**k:x,
        isfinite=lambda x:SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:True)),
        cuda=SimpleNamespace(synchronize=lambda:None))
    backend=SimpleNamespace(_require_active=lambda h,f:h.require(f),torch=torch,_clock=lambda:clock,
        model=lambda **k:SimpleNamespace(logits=BadLogits((1,1,128256)),past_key_values=pack_cache(raw(130),130,8,quantizer=quantizer)))
    with pytest.raises(BackendError): decode_one(backend,cache,10)
    assert cache.metadata.length==129 and cache.require(CacheFormat.PACKED) is source
