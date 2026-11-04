"""Independent unchanged native-model wrapper and candidate orchestration.

The native wrapper calls the pinned model directly. It never uses raw Prefill,
pack_cache, the deferred K2 backend, or the candidate Decode wrapper.
"""
from dataclasses import dataclass

from ..config import EOS_TOKEN_IDS, GROUPS
from ..contracts import generation_stop
from ..errors import QualificationError


@dataclass(frozen=True)
class GenerationRecord:
    token_ids: tuple[int, ...]
    stop_reason: str
    decode_forwards: int
    final_cache_length: int


def generate_candidate(backend, request, group, *, initial_cache=None, first_token=None):
    if type(group) is not int or group not in GROUPS:
        raise QualificationError('invalid candidate G')
    first = backend.prefill_raw(request)
    cache = first.cache
    tokens = (first.token_id,)
    forwards = 0
    try:
        if first_token is not None:
            first_token(first.token_id)
        stop = generation_stop(tokens, request.nmax)
        if stop is None:
            backend.pack_initial(cache, group)
            if initial_cache is not None:
                from ..contracts import CacheFormat
                initial_cache(cache.require(CacheFormat.PACKED))
            while stop is None:
                result = backend.decode_one(cache, tokens[-1])
                tokens += (result.token_id,)
                forwards += 1
                stop = generation_stop(tokens, request.nmax)
        return GenerationRecord(tokens, stop.value, forwards, cache.metadata.length)
    finally:
        backend.release_request(cache)


def generate_reference(loaded, request, group, *, initial_cache=None):
    """Only fixed G, direct model forwards, greedy EOS/cap, and recording."""
    from ..backend.cache import set_group
    if loaded.provenance.get('raw_prefill_adapter') is not False:
        raise QualificationError('native reference must use unchanged attention')
    set_group(loaded.model, group)  # loader intentionally starts with G=None
    torch, model = loaded.torch, loaded.model
    tokens, cache, output = (), None, None
    forwards = 0
    try:
        with torch.inference_mode():
            ids = torch.tensor([request.token_ids], dtype=torch.long, device='cuda:0')
            while True:
                options = dict(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
                if cache is not None:
                    options['position_ids'] = torch.tensor([[request.length + len(tokens) - 1]], dtype=torch.long, device='cuda:0')
                output = model(**options)
                cache = output.past_key_values
                logits = output.logits[:, -1, :]
                if not bool(torch.isfinite(logits).all().item()):
                    raise QualificationError('nonfinite unchanged-native logits')
                token = int(logits.argmax(dim=-1).item())
                tokens += (token,)
                expected_length = request.length + len(tokens) - 1
                if len(cache) != 32 or any(row[-1] != expected_length for row in cache):
                    raise QualificationError('unchanged-native cache length mismatch')
                torch.cuda.synchronize()
                if len(tokens) == 1 and initial_cache is not None and token not in EOS_TOKEN_IDS and request.nmax > 1:
                    initial_cache(cache)
                output = logits = None
                # This wrapper independently expresses EOS/cap semantics.
                if token in EOS_TOKEN_IDS:
                    reason = 'eos'
                    break
                if len(tokens) == request.nmax:
                    reason = 'length'
                    break
                ids = torch.tensor([[token]], dtype=torch.long, device='cuda:0')
                forwards += 1
            return GenerationRecord(tokens, reason, forwards, expected_length)
    finally:
        output = cache = None
        torch.cuda.synchronize()
