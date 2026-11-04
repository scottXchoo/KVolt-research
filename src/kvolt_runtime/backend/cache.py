"""One-shot post-RoPE FP16 → native K2V2 cache transition.

The partition and quantizer calls follow the existing project helper
scripts/k2_decode_lut/backend.py:38–85 (baseline ae3c07e6); the unchanged
KIVI quantizer owns arithmetic and packing. KIVI's MIT license is retained in
backend/vendor/LICENSE.KIVI. No model or GPU package is imported at collection.
"""
from dataclasses import replace

from ..config import GROUPS
from ..contracts import CacheFormat
from ..errors import BackendError
from .prefill import validate_raw_cache

FIELDS = ('key_code', 'key_tail', 'key_scale', 'key_min', 'value_code',
          'value_tail', 'value_scale', 'value_min', 'length')


def _group(group):
    if type(group) is not int or group not in GROUPS:
        raise BackendError('K2V2 requires G8/16/32/64/128')
    return group


def partition_lengths(length):
    if type(length) is not int or length < 1:
        raise BackendError('native cache length must be a positive integer')
    return length // 128 * 128, length % 128, max(0, length - 128), min(length, 128)


def validate_packed_cache(cache, length, group, *, device=None):
    """Validate every native field before publishing any replacement handle."""
    _group(group)
    kp, kr, vp, vr = partition_lengths(length)
    if not isinstance(cache, (tuple, list)) or len(cache) != 32:
        raise BackendError('packed cache requires all 32 layers')
    specifications = (
        ((1,8,128,kp//16), 'torch.int32', bool(kp)),
        ((1,8,kr,128), 'torch.float16', bool(kr)),
        ((1,8,128,kp//group), 'torch.float16', bool(kp)),
        ((1,8,128,kp//group), 'torch.float16', bool(kp)),
        ((1,8,vp,8), 'torch.int32', bool(vp)),
        ((1,8,vr,128), 'torch.float16', True),
        ((1,8,vp,128//group), 'torch.float16', bool(vp)),
        ((1,8,vp,128//group), 'torch.float16', bool(vp)),
    )
    for layer, row in enumerate(cache):
        if not isinstance(row, (tuple, list)) or len(row) != 9 or type(row[8]) is not int or row[8] != length:
            raise BackendError(f'layer {layer}: native nine-field length mismatch')
        for field, (shape, dtype, present) in enumerate(specifications):
            tensor = row[field]
            if not present:
                if tensor is not None:
                    raise BackendError(f'layer {layer}/{FIELDS[field]}: empty field must be None')
                continue
            if tensor is None or tuple(tensor.shape) != shape or str(tensor.dtype) != dtype:
                raise BackendError(f'layer {layer}/{FIELDS[field]}: shape/dtype mismatch')
            if tensor.device.type != 'cuda' or (device is not None and str(tensor.device) != device):
                raise BackendError(f'layer {layer}/{FIELDS[field]}: CUDA device mismatch')
            device = str(tensor.device)
    return device


def pack_cache(raw, length, group, *, quantizer=None):
    """Pure payload construction; caller synchronizes and commits atomically.

    Both full raw tensors are inspected in all 32 layers before the first
    quantizer call. Tails own new storage even for T<128 and contiguous slices.
    """
    _group(group)
    device = validate_raw_cache(raw, length)
    kp, kr, vp, _ = partition_lengths(length)
    if quantizer is None:
        from quant.new_pack import triton_quantize_and_pack_along_last_dim
        quantizer = triton_quantize_and_pack_along_last_dim
    result = []
    for key, value in raw:
        if kp:
            prefix = key if not kr else key[:, :, :kp, :].contiguous()
            kcode, kscale, kmin = quantizer(prefix.transpose(2,3).contiguous(), group, 2)
        else:
            kcode = kscale = kmin = None
        ktail = key[:, :, kp:, :].clone().contiguous() if kr else None
        if vp:
            vcode, vscale, vmin = quantizer(value[:, :, :vp, :].contiguous(), group, 2)
        else:
            vcode = vscale = vmin = None
        vtail = value[:, :, vp:, :].clone().contiguous()
        result.append((kcode,ktail,kscale,kmin,vcode,vtail,vscale,vmin,length))
    packed = tuple(result)
    validate_packed_cache(packed,length,group,device=device)
    return packed


def set_group(model, group):
    """Fix all 32 attention/config settings together; no arithmetic changes."""
    _group(group)
    layers = model.model.layers
    if len(layers) != 32:
        raise BackendError('model requires 32 attention layers')
    targets = [model.config]
    for layer in layers:
        attention = layer.self_attn
        if any(getattr(attention, name, None) != fixed for name, fixed in
               [('k_bits',2),('v_bits',2),('residual_length',128)]):
            raise BackendError('attention must use K2V2/R128')
        targets.extend((attention,attention.config))
    for target in targets:
        target.group_size = group


def pack_initial(backend, cache, group, *, quantizer=None):
    """Commit RAW→PACKED only after validation and successful GPU completion."""
    raw = backend._require_active(cache, CacheFormat.RAW)
    _group(group)
    metadata = cache.metadata
    if validate_raw_cache(raw, metadata.length) != metadata.device:
        raise BackendError('raw handle device does not match payload')
    try:
        with backend.torch.inference_mode():
            packed = pack_cache(raw, metadata.length, group, quantizer=quantizer)
            backend.torch.cuda.synchronize()
            # Settings change only after every kernel has completed. The
            # handle still owns raw KV if preparation or validation fails.
            set_group(backend.model, group)
            cache.commit_packed(replace(metadata, format=CacheFormat.PACKED, group_size=group),packed)
    except BackendError:
        raise
    except Exception as error:
        raise BackendError('initial K2V2 packing failed; raw cache retained') from error
