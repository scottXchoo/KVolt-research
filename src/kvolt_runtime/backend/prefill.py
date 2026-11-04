"""Request-local raw Prefill, retaining the patched KIVI arithmetic order.

No GPU packages are imported here. Only the no-past branch changes: quantization
is deferred and the exact post-RoPE/cast FlashAttention K/V inputs are returned.
Packed-cache Decode delegates to the byte-identical vendored implementation.
"""
from ..errors import BackendError


def validate_raw_cache(cache, length: int) -> str:
    if not isinstance(cache, (tuple, list)) or len(cache) != 32:
        raise BackendError('raw Prefill must return all 32 layers')
    device = None
    for index, row in enumerate(cache):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise BackendError(f'layer {index}: expected raw K/V pair')
        for tensor in row:
            if tuple(tensor.shape) != (1, 8, length, 128) or str(tensor.dtype) != 'torch.float16':
                raise BackendError(f'layer {index}: raw shape/dtype mismatch')
            if tensor.device.type != 'cuda' or (device is not None and str(tensor.device) != device):
                raise BackendError(f'layer {index}: raw KV must share one CUDA device')
            device = str(tensor.device)
    return device


def raw_attention_class(base, torch, apply_rotary_pos_emb):
    """Create a local subclass; no global registry or baseline class mutation.

    Source correspondence: vendor/llama_kivi.py:299-311,404-424,457-467.
    M1's frozen snapshot has pretraining_tp=1 and requires batch-one/no padding.
    RoPE, its buffers, conditional cast, flash call and output projection keep
    their original order. The original FP32 softmax path remains in Decode.
    """
    class RawPrefillAttention(base):
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
            if past_key_value is not None:
                return super().forward(hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=output_attentions, use_cache=use_cache, **kwargs)
            if self.config.pretraining_tp != 1 or not use_cache:
                raise BackendError('raw Prefill requires cached pretraining_tp=1')
            if attention_mask is not None or 'padding_mask' in kwargs:
                raise BackendError('raw Prefill requires prepared inputs without padding masks')
            bsz, q_len, _ = hidden_states.size()
            if bsz != 1:
                raise BackendError('raw Prefill requires batch one')
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)
            if query_states.dtype == torch.float32:
                target_dtype = getattr(self.config, '_pre_quantization_dtype', self.q_proj.weight.dtype)
                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)
            if key_states.dtype != torch.float16 or value_states.dtype != torch.float16:
                raise BackendError('raw Prefill requires original FP16 KV')
            attn_output = self._flash_attention_forward(
                query_states.transpose(1, 2), key_states.transpose(1, 2),
                value_states.transpose(1, 2), None, q_len, dropout=0.0)
            # Only upstream lines 425-453 (initial quantization) are deferred.
            cache = (key_states, value_states)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, cache
    return RawPrefillAttention


def install_raw_attention(model, torch, source):
    layers = model.model.layers
    if len(layers) != 32 or any(type(layer.self_attn) is not source.LlamaFlashAttention_KIVI for layer in layers):
        raise BackendError('expected 32 original patched KIVI attention instances')
    cls = raw_attention_class(source.LlamaFlashAttention_KIVI, torch, source.apply_rotary_pos_emb)
    # __class__ changes method dispatch only; every module/parameter/buffer stays.
    for layer in layers:
        layer.self_attn.__class__ = cls
    return cls
