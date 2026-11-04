"""One original patched-KIVI forward on the same resident model."""
from dataclasses import replace

from ..contracts import CacheFormat, TokenResult
from ..errors import BackendError
from .cache import validate_packed_cache


def decode_one(backend, cache, previous_token_id):
    packed = backend._require_active(cache, CacheFormat.PACKED)
    metadata = cache.metadata
    if type(previous_token_id) is not int or not 0 <= previous_token_id < 128256:
        raise BackendError('Decode requires one integer token ID')
    validate_packed_cache(packed,metadata.length,metadata.group_size,device=metadata.device)
    output = replacement = token = None
    try:
        with backend.torch.inference_mode():
            ids = backend.torch.tensor([[previous_token_id]],dtype=backend.torch.long,device=metadata.device)
            positions = backend.torch.tensor([[metadata.length]],dtype=backend.torch.long,device=metadata.device)
            output = backend.model(input_ids=ids,position_ids=positions,past_key_values=packed,
                                   use_cache=True,return_dict=True)
            replacement = output.past_key_values
            validate_packed_cache(replacement,metadata.length+1,metadata.group_size,device=metadata.device)
            logits = output.logits[:, -1, :]
            if not bool(backend.torch.isfinite(logits).all().item()):
                raise BackendError('nonfinite Decode logits')
            token = logits.argmax(dim=-1,keepdim=True)
            token_id = int(token.item())
            backend.torch.cuda.synchronize()
            ready_ns = backend._clock()
            result = TokenResult(token_id,ready_ns,cache)
            cache.commit_decode(replace(metadata,length=metadata.length+1),replacement)
            return result
    except BackendError:
        raise
    except Exception as error:
        raise BackendError('K2V2 Decode failed; cache was not committed') from error
    finally:
        output = replacement = token = None
