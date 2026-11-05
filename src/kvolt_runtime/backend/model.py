"""Single-instance FP16 KIVI session. Importing this module is CPU safe."""
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from ..config import RuntimeConfig
from ..contracts import CacheFormat, CacheHandle, CacheMetadata, RequestSpec, TokenResult
from ..errors import BackendError, CacheStateError, QualificationError
from .prefill import install_raw_attention, validate_raw_cache

DEFAULT_KIVI_ROOT = Path('/home/agx3/kvolt/third_party/KIVI')


@dataclass
class LoadedModel:
    model: Any
    torch: Any
    tokenizer: Any
    provenance: dict


def model_identity(model):
    """In-process identity diagnostics; preserve exact parameter and buffer objects."""
    return {'root': id(model),
            'parameters': tuple((name, id(t), t.data_ptr(), str(t.device), str(t.dtype))
                                for name, t in model.named_parameters()),
            'buffers': tuple((name, id(t), t.data_ptr(), str(t.device), str(t.dtype))
                             for name, t in model.named_buffers())}


def load_model(snapshot_path, config, *, kivi_root=DEFAULT_KIVI_ROOT, raw_prefill=True,
               native_profile='agx1'):
    """The reference passes raw_prefill=False and gets the unchanged source class.

    Both modes construct exactly one LlamaForCausalLM_KIVI. No HF model, meta
    skeleton, weight copy or global model registration is used.
    """
    from .provenance import digest, import_pinned_source, validate_sources
    snapshot = Path(snapshot_path).resolve()
    if snapshot.name != config.snapshot:
        raise QualificationError('model directory must name the fixed snapshot revision')
    provenance = validate_sources(kivi_root, snapshot, native_profile=native_profile)
    # All hash/version validation above is stdlib-only and precedes GPU imports.
    import torch
    from transformers import AutoTokenizer, LlamaConfig
    source = import_pinned_source(kivi_root, provenance)
    settings = LlamaConfig.from_pretrained(str(snapshot), local_files_only=True)
    architecture = (settings.hidden_size, settings.num_attention_heads, settings.num_key_value_heads,
                    settings.num_hidden_layers, settings.vocab_size, settings.pretraining_tp)
    if architecture != (4096, 32, 8, 32, 128256, 1):
        raise QualificationError('unexpected fixed model architecture')
    if settings.rope_theta != 500000.0 or settings.rope_scaling.get('rope_type') != 'llama3' or settings.max_position_embeddings < 8704:
        raise QualificationError('unexpected Llama-3.1 RoPE/capacity')
    settings.use_cache = settings.use_flash = settings._flash_attn_2_enabled = True
    settings._attn_implementation = 'flash_attention_2'
    settings.k_bits, settings.v_bits = config.key_bits, config.value_bits
    settings.residual_length, settings.group_size = config.residual_length, None
    model = source.LlamaForCausalLM_KIVI.from_pretrained(
        str(snapshot), config=settings, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={'': 'cuda:0'}, local_files_only=True,
        attn_implementation='flash_attention_2').eval()
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float16 or parameter.device.type != 'cuda':
            raise QualificationError(f'nonresident/non-FP16 model parameter: {name}')
    for name, buffer in model.named_buffers():
        if buffer.device.type != 'cuda':
            raise QualificationError(f'nonresident model buffer: {name}')
    identity = model_identity(model)
    if raw_prefill:
        install_raw_attention(model, torch, source)
    if model_identity(model) != identity:
        raise QualificationError('installing raw adapter changed model parameter/buffer identity')
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if tuple(model.generation_config.eos_token_id) != config.eos_token_ids or tokenizer.eos_token_id not in config.eos_token_ids:
        raise QualificationError('unexpected model/tokenizer EOS IDs')
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    provenance.update({'execution_class': type(model).__name__, 'raw_prefill_adapter': raw_prefill,
        'model_instances': 1, 'effective_overrides': {'torch_dtype': 'float16', 'do_sample': False,
        'num_beams': 1, 'eos_token_ids': list(config.eos_token_ids), 'use_cache': True},
        'identity_at_load': identity,
        'adapter_source_sha256': {name: digest(Path(__file__).with_name(name))
                                  for name in ['model.py', 'prefill.py', 'provenance.py']}})
    torch.cuda.synchronize()
    return LoadedModel(model, torch, tokenizer, provenance)


class InferenceBackend:
    def __init__(self, *, loader=None, kivi_root=DEFAULT_KIVI_ROOT, clock=time.monotonic_ns,
                 native_profile='agx1'):
        from .provenance import native_build_hashes
        native_build_hashes(native_profile)
        self._loader = loader or (lambda path, config: load_model(
            path, config, kivi_root=kivi_root, native_profile=native_profile))
        self._clock, self._loaded, self._active_cache = clock, None, None

    @property
    def model(self):
        if self._loaded is None:
            raise BackendError('session has not been loaded')
        return self._loaded.model

    @property
    def torch(self):
        self.model
        return self._loaded.torch

    @property
    def tokenizer(self):
        self.model
        return self._loaded.tokenizer

    @property
    def provenance(self):
        self.model
        return self._loaded.provenance

    def load_session(self, snapshot_path: Path, config: RuntimeConfig) -> None:
        if self._loaded is not None:
            raise BackendError('one session may load its model only once')
        if not isinstance(config, RuntimeConfig):
            raise BackendError('load_session requires RuntimeConfig')
        try:
            loaded = self._loader(Path(snapshot_path), config)
            if not isinstance(loaded, LoadedModel):
                raise BackendError('loader must return LoadedModel')
            self._loaded = loaded
        except BackendError:
            raise
        except Exception as error:
            raise BackendError('model session load failed') from error

    def prefill_raw(self, request: RequestSpec) -> TokenResult:
        self.model
        if self._active_cache is not None:
            raise CacheStateError('release the active request before starting another')
        if not isinstance(request, RequestSpec) or request.length > 8192:
            raise BackendError('Prefill requires RequestSpec with at most 8192 input tokens')
        output = raw = token = None
        try:
            with self.torch.inference_mode():
                input_ids = self.torch.tensor([request.token_ids], dtype=self.torch.long, device='cuda:0')
                output = self.model(input_ids=input_ids, past_key_values=None,
                                    use_cache=True, return_dict=True)
                raw = output.past_key_values
                device = validate_raw_cache(raw, request.length)
                logits = output.logits[:, -1, :]
                if not bool(self.torch.isfinite(logits).all().item()):
                    raise BackendError('nonfinite Prefill logits')
                token = logits.argmax(dim=-1, keepdim=True)
                token_id = int(token.item())
                # Token selection alone need not wait unrelated streams. The API
                # promises all returned KV and output work is complete here.
                self.torch.cuda.synchronize()
                ready_ns = self._clock()
                metadata = CacheMetadata(request.request_id, request.length, device, CacheFormat.RAW)
                cache = CacheHandle(metadata, raw)
                result = TokenResult(token_id, ready_ns, cache)
                self._active_cache = cache
                return result
        except BackendError:
            raise
        except Exception as error:
            raise BackendError('raw Prefill failed') from error
        finally:
            # Never retain CausalLMOutput/logits or raw sidecars on the model.
            output = raw = token = None

    def _require_active(self, cache, format=None):
        if cache is not self._active_cache:
            raise CacheStateError('cache does not belong to this active request')
        if format is not None:
            return cache.require(format)
        return cache

    def pack_initial(self, cache: CacheHandle, group_size: int) -> None:
        from .cache import pack_initial
        pack_initial(self, cache, group_size)

    def decode_one(self, cache: CacheHandle, previous_token_id: int) -> TokenResult:
        from .decode import decode_one
        return decode_one(self, cache, previous_token_id)

    def release_request(self, cache: CacheHandle) -> None:
        if cache is not self._active_cache:
            if isinstance(cache, CacheHandle) and cache.format is CacheFormat.RELEASED:
                return
            raise CacheStateError('cache does not belong to this active request')
        try:
            self.torch.cuda.synchronize()
        finally:
            cache.release()
            self._active_cache = None
