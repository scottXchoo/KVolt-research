"""CPU-only contracts shared by the runner and GPU backend.

Cache payloads are opaque here. Only the backend may validate GPU tensors and
publish a replacement after *all* layers and GPU work have completed.
"""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Protocol

from .config import EOS_TOKEN_IDS, GROUPS, require_int, resolve_nmax
from .errors import CacheStateError, ContractError


class Clock(Protocol):
    """One shared monotonic host clock, returning integer nanoseconds.

    Backend token-ready timestamps and future runner event timestamps must use
    this same clock. A production implementation may use time.perf_counter_ns.
    """

    def __call__(self) -> int:
        ...


def _tokens(ids: tuple[int, ...]) -> None:
    if type(ids) is not tuple or not ids:
        raise ContractError('token IDs must be a nonempty tuple')
    for token in ids:
        require_int(token, 'token ID', 0, 128255)


def token_ids_sha256(ids: tuple[int, ...]) -> str:
    _tokens(ids)
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class PromptInput:
    path: Path
    text: str
    sha256: str

    @classmethod
    def read(cls, path: Path) -> 'PromptInput':
        raw = Path(path).read_bytes()  # no universal-newline conversion
        try:
            text = raw.decode('utf-8', errors='strict')
        except UnicodeDecodeError as error:
            raise ContractError('prompt must be UTF-8') from error
        return cls(Path(path).resolve(), text, hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    token_ids: tuple[int, ...]
    prompt_sha256: str
    nmax: int = 512

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id.strip():
            raise ContractError('request_id must be a nonempty string')
        _tokens(self.token_ids)
        if type(self.prompt_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', self.prompt_sha256):
            raise ContractError('prompt_sha256 must be a lowercase SHA-256 hex digest')
        resolve_nmax(item=self.nmax)

    @classmethod
    def from_prompt(cls, request_id: str, prompt: PromptInput, tokenizer: Callable[..., Any], *, nmax: int = 512) -> 'RequestSpec':
        encoded = tokenizer(prompt.text, add_special_tokens=False)
        return cls(request_id, tuple(encoded['input_ids']), prompt.sha256, nmax)

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def token_sha256(self) -> str:
        return token_ids_sha256(self.token_ids)


class CacheFormat(str, Enum):
    RAW = 'raw_fp16'
    PACKED = 'packed_k2v2'
    RELEASED = 'released'


@dataclass(frozen=True)
class CacheMetadata:
    request_id: str
    length: int
    device: str
    format: CacheFormat
    group_size: int | None = None
    num_layers: int = 32
    batch_size: int = 1
    num_kv_heads: int = 8
    head_dim: int = 128

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id.strip():
            raise ContractError('cache request_id must be nonempty')
        require_int(self.length, 'cache length', 1)
        if type(self.device) is not str or not self.device:
            raise ContractError('cache device must be nonempty')
        if type(self.format) is not CacheFormat or self.format is CacheFormat.RELEASED:
            raise ContractError('cache metadata requires a live format')
        if self.format is CacheFormat.PACKED:
            if type(self.group_size) is not int or self.group_size not in GROUPS:
                raise ContractError('packed cache requires a valid G')
        elif self.group_size is not None:
            raise ContractError('raw cache cannot have a group size')
        for key, value in [('num_layers', 32), ('batch_size', 1), ('num_kv_heads', 8), ('head_dim', 128)]:
            if type(getattr(self, key)) is not int or getattr(self, key) != value:
                raise ContractError(f'M1 cache fixes {key}={value}')


class CacheHandle:
    """One active cache, request-scoped; not thread-safe.

    A successful commit drops this handle's old full-cache reference. The caller
    must not retain aliases. Metadata and payload are replaced together; failed
    validation leaves the old cache live. release() is idempotent for finally.
    """

    def __init__(self, metadata: CacheMetadata, payload: Any):
        if not isinstance(metadata, CacheMetadata) or metadata.format is not CacheFormat.RAW or payload is None:
            raise ContractError('a new cache handle requires raw metadata and payload')
        self._active = (metadata, payload)

    @property
    def format(self) -> CacheFormat:
        return CacheFormat.RELEASED if self._active is None else self._active[0].format

    @property
    def metadata(self) -> CacheMetadata:
        if self._active is None:
            raise CacheStateError('cache has been released')
        return self._active[0]

    def require(self, format: CacheFormat) -> Any:
        if self.format is not format or format is CacheFormat.RELEASED:
            raise CacheStateError(f'expected {format.value}, got {self.format.value}')
        return self._active[1]

    def _validate_replacement(self, metadata: CacheMetadata, payload: Any, *, decode: bool):
        old = self.metadata
        expected = CacheFormat.PACKED if decode else CacheFormat.RAW
        if old.format is not expected:
            raise CacheStateError(f'cache transition requires {expected.value}')
        if not isinstance(metadata, CacheMetadata) or metadata.format is not CacheFormat.PACKED or payload is None:
            raise CacheStateError('replacement must be complete packed cache')
        if metadata.request_id != old.request_id or metadata.device != old.device:
            raise CacheStateError('cache identity/device cannot change')
        if metadata.length != old.length + int(decode):
            raise CacheStateError('packing preserves T; one decode increments T by one')
        if decode and metadata.group_size != old.group_size:
            raise CacheStateError('G cannot change during a request')

    def commit_packed(self, metadata: CacheMetadata, payload: Any) -> None:
        self._validate_replacement(metadata, payload, decode=False)
        self._active = (metadata, payload)

    def commit_decode(self, metadata: CacheMetadata, payload: Any) -> None:
        self._validate_replacement(metadata, payload, decode=True)
        self._active = (metadata, payload)

    def release(self) -> None:
        self._active = None


class StopReason(str, Enum):
    EOS = 'eos'
    LENGTH = 'length'
    INTERRUPTED = 'interrupted'


def generation_stop(token_ids: tuple[int, ...], nmax: int) -> StopReason | None:
    _tokens(token_ids)
    resolve_nmax(item=nmax)
    if len(token_ids) > nmax:
        raise ContractError('generated tokens exceed Nmax')
    if any(token in EOS_TOKEN_IDS for token in token_ids[:-1]):
        raise ContractError('generation continued after EOS')
    if token_ids[-1] in EOS_TOKEN_IDS:
        return StopReason.EOS
    return StopReason.LENGTH if len(token_ids) == nmax else None


def ns_to_seconds(elapsed_ns: int) -> float:
    return require_int(elapsed_ns, 'elapsed_ns', 0) / 1_000_000_000


def mhz_to_hz(frequency_mhz: int) -> int:
    return require_int(frequency_mhz, 'frequency_mhz', 1) * 1_000_000


@dataclass(frozen=True)
class TokenResult:
    token_id: int
    ready_ns: int
    cache: CacheHandle

    def __post_init__(self):
        require_int(self.token_id, 'token ID', 0, 128255)
        require_int(self.ready_ns, 'ready_ns', 0)
        if not isinstance(self.cache, CacheHandle) or self.cache.format is CacheFormat.RELEASED:
            raise ContractError('token result requires a live CacheHandle')
