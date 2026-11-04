"""B1 CPU contracts; no torch/transformers import or model load."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvolt_runtime.backend.model import InferenceBackend, LoadedModel, load_model, model_identity
from kvolt_runtime.backend.prefill import install_raw_attention, raw_attention_class, validate_raw_cache
from kvolt_runtime.config import RuntimeConfig
from kvolt_runtime.contracts import CacheFormat, RequestSpec
from kvolt_runtime.errors import BackendError, CacheStateError, QualificationError


class Device:
    type = 'cuda'
    def __str__(self): return 'cuda:0'
    def __eq__(self, other): return str(self) == str(other)


class Tensor:
    dtype = 'torch.float16'
    device = Device()
    def __init__(self, shape=(1, 8, 3, 128), value=7, trace=None, dtype='torch.float16'):
        self.shape, self.value, self.trace = shape, value, trace
        self.dtype = dtype
    def size(self): return self.shape
    def view(self, *shape):
        return Tensor(shape, self.value, self.trace, self.dtype)
    def transpose(self, a, b):
        shape = list(self.shape); shape[a], shape[b] = shape[b], shape[a]
        return Tensor(tuple(shape), self.value, self.trace, self.dtype)
    def to(self, dtype): return Tensor(self.shape, self.value, self.trace, dtype)
    def reshape(self, *shape): return self.view(*shape)
    def __getitem__(self, key): return self
    def all(self): return self
    def item(self): return self.value
    def argmax(self, **kwargs): return Tensor((1, 1), self.value)


class Torch:
    float16 = 'torch.float16'
    float32 = 'torch.float32'
    long = 'torch.int64'
    def __init__(self): self.syncs = 0
    def inference_mode(self): return nullcontext()
    def tensor(self, ids, **kwargs): return Tensor((1, len(ids[0])))
    def isfinite(self, logits): return Tensor(value=True)
    @property
    def cuda(self): return self
    def synchronize(self): self.syncs += 1


class Model:
    def __init__(self):
        self.parameter = object(); self.calls = []
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        length = kwargs['input_ids'].shape[1]
        return SimpleNamespace(logits=Tensor((1, length, 128256), value=getattr(self, 'next_token', 7)),
                               past_key_values=tuple((Tensor((1, 8, length, 128)), Tensor((1, 8, length, 128))) for _ in range(32)))


def request(identifier='one', nmax=128):
    return RequestSpec(identifier, (1, 2, 3), 'a' * 64, nmax)


def backend():
    model, torch, calls = Model(), Torch(), []
    def load(path, config):
        calls.append((path, config))
        return LoadedModel(model, torch, object(), {'test': True})
    result = InferenceBackend(loader=load)
    result.load_session(Path('snapshot'), RuntimeConfig())
    return result, model, torch, calls


def test_import_does_not_import_gpu_dependencies():
    import os, subprocess, sys
    code = "import sys; import kvolt_runtime.backend.model; assert not ({'torch','transformers','triton','flash_attn'} & set(sys.modules))"
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    assert subprocess.run([sys.executable, '-c', code], env=environment, check=False).returncode == 0


def test_one_model_two_requests_no_reload_or_past_leak():
    b, model, torch, loads = backend()
    parameter = model.parameter
    for identifier in ('one', 'two'):
        token = b.prefill_raw(request(identifier))
        assert token.token_id == 7 and token.ready_ns > 0
        assert token.cache.metadata.length == 3
        assert token.cache.metadata.format is CacheFormat.RAW
        assert token.cache.metadata.group_size is None
        assert len(token.cache.require(CacheFormat.RAW)) == 32
        assert model.calls[-1]['past_key_values'] is None
        assert model.calls[-1]['use_cache'] is True
        b.release_request(token.cache)
        b.release_request(token.cache)
        assert token.cache.format is CacheFormat.RELEASED
        assert b.model is model and model.parameter is parameter
    assert len(loads) == 1 and torch.syncs >= 2
    with pytest.raises(BackendError): b.load_session(Path('snapshot'), RuntimeConfig())


def test_busy_session_and_foreign_release_rejected():
    b, _, _, _ = backend(); first = b.prefill_raw(request())
    with pytest.raises(CacheStateError): b.prefill_raw(request('two'))
    other, _, _, _ = backend(); foreign = other.prefill_raw(request('foreign'))
    with pytest.raises(CacheStateError): b.release_request(foreign.cache)
    assert first.cache.format is CacheFormat.RAW
    b.release_request(first.cache); other.release_request(foreign.cache)


def test_invalid_raw_cache_never_published_and_next_request_is_clean():
    b, model, _, _ = backend()
    original = model.__class__.__call__
    class Broken(Model):
        def __call__(self, **kwargs):
            output = original(self, **kwargs)
            output.past_key_values = output.past_key_values[:-1]
            return output
    model.__class__ = Broken
    with pytest.raises(BackendError): b.prefill_raw(request())
    model.__class__ = Model
    result = b.prefill_raw(request('fresh')); b.release_request(result.cache)


@pytest.mark.parametrize('nmax', [1, 2, 128, 512])
def test_prefill_ignores_output_cap_and_never_packs(nmax):
    b, _, _, _ = backend()
    b.pack_initial = lambda *a: pytest.fail('Prefill attempted packing')
    result = b.prefill_raw(request(nmax=nmax))
    assert result.cache.metadata.length == request().length
    b.release_request(result.cache)


def test_raw_cache_checks_every_layer_and_device():
    rows = [(Tensor(), Tensor()) for _ in range(32)]
    assert validate_raw_cache(rows, 3) == 'cuda:0'
    rows[-1][1].dtype = 'torch.float32'
    with pytest.raises(BackendError): validate_raw_cache(rows, 3)


@pytest.mark.parametrize('failure', ['synchronize', 'clock', 'nonfinite', 'token'])
def test_failed_prefill_does_not_publish_cache_and_allows_fresh_request(failure):
    b, model, torch, _ = backend()
    with pytest.MonkeyPatch.context() as patch:
        if failure == 'synchronize':
            def fail(): raise RuntimeError('injected CUDA failure')
            patch.setattr(torch, 'synchronize', fail)
        elif failure == 'clock':
            patch.setattr(b, '_clock', lambda: -1)
        elif failure == 'nonfinite':
            patch.setattr(torch, 'isfinite', lambda logits: Tensor(value=False))
        else:
            patch.setattr(model, 'next_token', 128256, raising=False)
        with pytest.raises(BackendError): b.prefill_raw(request())
        assert b._active_cache is None
    token = b.prefill_raw(request('fresh'))
    b.release_request(token.cache)


def test_load_model_one_instance_preserves_parameters_buffers_and_override(monkeypatch):
    import sys
    from kvolt_runtime.backend import provenance
    class Storage(Tensor):
        def data_ptr(self): return id(self)
    class Base: pass
    class Resident:
        calls = []
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            return cls()
        def __init__(self):
            self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=Base()) for _ in range(32)])
            self.parameter, self.buffer = Storage(), Storage()
            self.buffer.dtype = 'torch.float32'
            self.generation_config = SimpleNamespace(eos_token_id=[128001, 128008, 128009], do_sample=True, num_beams=4)
        def named_parameters(self): return [('weight', self.parameter)]
        def named_buffers(self): return [('rope.inv_freq', self.buffer)]
        def eval(self): return self
    settings = SimpleNamespace(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        num_hidden_layers=32, vocab_size=128256, pretraining_tp=1, rope_theta=500000.0,
        rope_scaling={'rope_type': 'llama3'}, max_position_embeddings=131072)
    source = SimpleNamespace(LlamaForCausalLM_KIVI=Resident, LlamaFlashAttention_KIVI=Base,
                             apply_rotary_pos_emb=lambda *args: args)
    monkeypatch.setattr(provenance, 'validate_sources', lambda *args: {})
    monkeypatch.setattr(provenance, 'import_pinned_source', lambda *args: source)
    monkeypatch.setitem(sys.modules, 'torch', Torch())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        LlamaConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: settings),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(eos_token_id=128009))))
    result = load_model(Path(RuntimeConfig().snapshot), RuntimeConfig())
    assert len(Resident.calls) == 1
    assert Resident.calls[0][1]['torch_dtype'] == 'torch.float16'
    assert Resident.calls[0][1]['local_files_only'] is True
    assert result.provenance['identity_at_load'] == model_identity(result.model)
    assert result.model.buffer.dtype == 'torch.float32'
    assert result.model.generation_config.do_sample is False
    assert settings.group_size is None
    assert all(type(layer.self_attn) is not Base for layer in result.model.model.layers)
    reference = load_model(Path(RuntimeConfig().snapshot), RuntimeConfig(), raw_prefill=False)
    assert all(type(layer.self_attn) is Base for layer in reference.model.model.layers)


def test_source_gate_rejects_changed_file_before_gpu_import(tmp_path):
    from kvolt_runtime.backend.provenance import checked_file
    path = tmp_path / 'changed.py'; path.write_text('changed')
    with pytest.raises(QualificationError): checked_file(path, '0' * 64)


def test_attention_install_changes_only_instance_dispatch():
    class Base: pass
    class Storage(Tensor):
        def data_ptr(self): return id(self)
    class Resident:
        def __init__(self):
            self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=Base()) for _ in range(32)])
            self.parameter, self.buffer = Storage(), Storage()
        def named_parameters(self): return [('weight', self.parameter)]
        def named_buffers(self): return [('rope', self.buffer)]
    model = Resident(); before = model_identity(model); baseline_class = dict(Base.__dict__)
    source = SimpleNamespace(LlamaFlashAttention_KIVI=Base, apply_rotary_pos_emb=lambda *args: args)
    install_raw_attention(model, Torch(), source)
    assert model_identity(model) == before and dict(Base.__dict__) == baseline_class


@pytest.mark.parametrize('projection_dtype', ['torch.float16', 'torch.float32'])
def test_raw_attention_never_calls_native_quant_branch_and_decode_delegates(projection_dtype):
    trace = []
    class Base:
        def forward(self, *args, **kwargs):
            trace.append('native'); return 'native-output'
        def _flash_attention_forward(self, query, key, value, mask, length, **kwargs):
            trace.append('flash')
            assert query.shape == (1, 3, 32, 128)
            assert key.shape == value.shape == (1, 3, 8, 128)
            assert query.dtype == key.dtype == value.dtype == 'torch.float16'
            return Tensor((1, 3, 32, 128))
    def rotary(query, key, cos, sin, positions):
        trace.append('rope'); return query, key
    cls = raw_attention_class(Base, Torch(), rotary)
    attention = cls()
    attention.config = SimpleNamespace(pretraining_tp=1)
    attention.num_heads, attention.num_key_value_heads = 32, 8
    attention.head_dim, attention.hidden_size = 128, 4096
    for name, size in [('q_proj', 4096), ('k_proj', 1024), ('v_proj', 1024)]:
        def project(hidden, label=name, width=size):
            trace.append(label); return Tensor((1, 3, width), dtype=projection_dtype)
        project.weight = Tensor()
        setattr(attention, name, project)
    attention.rotary_emb = lambda value, positions: (None, None)
    attention.o_proj = lambda hidden: trace.append('o_proj') or hidden
    output, weights, raw = attention.forward(Tensor((1, 3, 4096)), use_cache=True)
    assert trace == ['q_proj', 'k_proj', 'v_proj', 'rope', 'flash', 'o_proj']
    assert weights is None and raw[0].shape == raw[1].shape == (1, 8, 3, 128)
    assert attention.forward(Tensor(), past_key_value=('packed',), use_cache=True) == 'native-output'
    assert trace[-1] == 'native'
