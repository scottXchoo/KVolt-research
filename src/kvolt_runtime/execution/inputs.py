"""CPU-only, immutable input preparation before any GPU import or model load."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from ..config import MISSING, require_int, resolve_nmax
from ..contracts import PromptInput, RequestSpec
from ..errors import ContractError
from .contracts import PreparedRequest

MANIFEST_SCHEMA = 'kvolt-request-manifest-v1'


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ContractError('input must contain finite JSON') from error


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ContractError('nonfinite JSON number')
    return number


def _reject_constant(value):
    raise ContractError(f'nonfinite JSON number: {value}')


def _read(path, kind):
    try:
        raw = path.read_bytes()
        text = raw.decode('utf-8', errors='strict')
    except (OSError, UnicodeError, ValueError) as error:
        raise ContractError(f'{kind} must be readable strict UTF-8: {path}') from error
    return raw, text


@dataclass(frozen=True)
class PreparedInputs:
    requests: tuple[PreparedRequest, ...]
    manifest_path: Path | None
    manifest_raw: bytes | None
    manifest_sha256: str | None
    resolved_manifest_json: str

    @property
    def resolved_manifest(self):
        """A caller may annotate its copy without changing prepared evidence."""
        return json.loads(self.resolved_manifest_json)


def _manifest_requests(text, cli_nmax):
    try:
        data = json.loads(text, object_pairs_hook=_unique_object,
                          parse_float=_finite_float, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        raise ContractError('manifest must be valid JSON') from error
    if type(data) is not dict or set(data) != {'schema_version', 'requests'}:
        raise ContractError('manifest requires only schema_version and requests')
    if data['schema_version'] != MANIFEST_SCHEMA:
        raise ContractError(f'manifest schema_version must be {MANIFEST_SCHEMA}')
    requests = data['requests']
    if type(requests) is not list or not requests:
        raise ContractError('manifest requests must be a nonempty array')
    seen = set()
    for item in requests:
        if type(item) is not dict or not {'request_id', 'prompt_file'} <= item.keys():
            raise ContractError('request requires request_id and prompt_file')
        if item.keys() - {'request_id', 'prompt_file', 'nmax', 'metadata'}:
            raise ContractError('unknown request field')
        request_id, prompt_file = item['request_id'], item['prompt_file']
        if type(request_id) is not str or not request_id.strip():
            raise ContractError('request_id must be a nonblank string')
        if request_id in seen:
            raise ContractError(f'duplicate request_id: {request_id}')
        seen.add(request_id)
        if type(prompt_file) is not str or not prompt_file.strip() or '\x00' in prompt_file:
            raise ContractError('prompt_file must be a nonblank path string')
        if type(item.get('metadata', {})) is not dict:
            raise ContractError('metadata must be an object')
        # Validate the whole manifest, including overridden values, before I/O.
        resolve_nmax(item=item.get('nmax', MISSING), cli=cli_nmax)
    return requests


def preflight_inputs(*, prompt_file: Path | None, manifest: Path | None,
                     tokenizer, repo, cli_nmax=MISSING) -> PreparedInputs:
    """Freeze original bytes, token IDs and all LUT support before GPU work.

    Repository preflight validates support without online prediction or Knee
    detection. The caller supplies its already pinned CPU tokenizer/repository.
    """
    if (prompt_file is None) == (manifest is None):
        raise ContractError('exactly one of prompt_file and manifest is required')
    resolve_nmax(cli=cli_nmax)
    manifest_path = manifest_raw = manifest_sha256 = None
    if manifest is not None:
        manifest_path = Path(manifest).resolve()
        manifest_raw, text = _read(manifest_path, 'manifest')
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        items = _manifest_requests(text, cli_nmax)
        base = manifest_path.parent
    else:
        items = [{'request_id': 'request-000000', 'prompt_file': str(Path(prompt_file).resolve())}]
        base = Path.cwd()

    prepared, resolved = [], []
    for item in items:
        path = Path(item['prompt_file'])
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        raw, text = _read(path, 'prompt')
        prompt = PromptInput(path, text, hashlib.sha256(raw).hexdigest())
        nmax = resolve_nmax(item=item.get('nmax', MISSING), cli=cli_nmax)
        request = RequestSpec.from_prompt(item['request_id'], prompt, tokenizer, nmax=nmax)
        require_int(request.length, 'actual prompt token length', 1024, 8192)
        repo.preflight(request.length)
        metadata_json = _canonical(item.get('metadata', {}))
        prepared.append(PreparedRequest(request, prompt, metadata_json))
        resolved.append({'request_id': request.request_id, 'prompt_file': str(path),
                         'nmax': request.nmax, 'metadata': json.loads(metadata_json)})
    evidence = _canonical({'schema_version': MANIFEST_SCHEMA, 'requests': resolved})
    # Reject unpaired JSON surrogates before later UTF-8 serialization can fail.
    try:
        evidence.encode('utf-8', errors='strict')
    except UnicodeError as error:
        raise ContractError('manifest strings must be valid UTF-8') from error
    return PreparedInputs(tuple(prepared), manifest_path, manifest_raw,
                          manifest_sha256, evidence)
