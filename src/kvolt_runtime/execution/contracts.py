"""CPU-only contracts for M3 resource ownership and recorded attempts.

Controllers own decisions, Backend owns tensors, Device owns GPU settings,
Collector owns power samples, and SessionRunner owns their invocation order.
Results use JSON-compatible dictionaries so cleanup and generation can retain
independent statuses without overwriting an earlier error.
"""
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Protocol

from ..config import require_int
from ..contracts import PromptInput, RequestSpec
from ..errors import ContractError, KVoltError

EXIT_OK = 0
EXIT_TECHNICAL = 1
EXIT_INPUT = 2
EXIT_PREPARATION = 3
EXIT_INTERRUPT = 130


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate metadata key')
        result[key] = value
    return result


class ExecutionError(KVoltError, RuntimeError):
    """A stage-specific technical failure; preserve cause and optional receipt."""

    def __init__(self, code, stage, message, *, receipt=None):
        super().__init__(message)
        self.code, self.stage, self.receipt = code, stage, receipt


class PreparationError(ExecutionError):
    """Busy hardware or unprepared device; no measured request has started."""


@dataclass(frozen=True)
class PreparedRequest:
    spec: RequestSpec
    prompt: PromptInput
    metadata_json: str = '{}'

    def __post_init__(self):
        if not isinstance(self.spec, RequestSpec) or not isinstance(self.prompt, PromptInput):
            raise ContractError('prepared request requires validated spec and prompt')
        if self.spec.prompt_sha256 != self.prompt.sha256:
            raise ContractError('prepared prompt hash must match the request')
        try:
            metadata = json.loads(self.metadata_json, object_pairs_hook=_unique_object,
                                  parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            canonical = json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False)
            canonical.encode('utf-8', errors='strict')
        except (TypeError, ValueError, RecursionError) as error:
            raise ContractError('request metadata must be finite JSON') from error
        if type(metadata) is not dict:
            raise ContractError('request metadata must be an object')
        object.__setattr__(self, 'metadata_json', canonical)

    @property
    def metadata(self):
        """Return a fresh object, preventing changes to preflighted metadata."""
        return json.loads(self.metadata_json)


@dataclass(frozen=True)
class SessionOptions:
    model_dir: Path
    output_dir: Path
    mode: str = 'development'
    collect_power: bool = False
    native_profile: str = 'agx1'

    def __post_init__(self):
        if not isinstance(self.model_dir, Path) or not isinstance(self.output_dir, Path):
            raise ContractError('session paths must be Path objects')
        if self.mode not in ('development', 'performance'):
            raise ContractError('mode must be development or performance')
        if type(self.collect_power) is not bool:
            raise ContractError('collect_power must be boolean')
        if self.native_profile not in ('agx1', 'agx2'):
            raise ContractError('unknown native build profile')
        if self.mode == 'performance' and not self.collect_power:
            raise ContractError('performance mode requires power collection')


class StopController:
    """Signal handler state. Handlers never call CUDA, storage or restoration.

    Runner observes this flag after the current forward returns. Repeated
    signals keep the first reason; SIGKILL and power loss cannot be recovered.
    """

    def __init__(self):
        self.signal_number = None

    def request(self, signal_number=2):
        require_int(signal_number, 'signal number', 1, 64)
        if self.signal_number is None:
            self.signal_number = signal_number

    @property
    def requested(self):
        return self.signal_number is not None

    @property
    def exit_code(self):
        return 128 + self.signal_number if self.requested else None


class Device(Protocol):
    """All methods require the caller to hold the common hardware lock."""

    def capture(self) -> dict: ...
    def verify_prepared(self) -> dict: ...
    def prepare_request(self) -> dict:
        """Set/verify common GPU DVFS start, outside t0; GPU changes only."""
        ...
    def apply(self, frequency_mhz: int) -> dict:
        """Fresh min/max and cur_freq confirmation, or raise with receipt."""
        ...
    def restore(self, snapshot: dict) -> dict:
        """Restore original governor/min/max, not a transient DVFS cur_freq."""
        ...


class PowerCollector(Protocol):
    """Host monotonic timestamps; collector lifetime is outside token intervals."""

    def start(self) -> dict:
        """Wait for a valid pre-t0 sample under a finite readiness deadline."""
        ...
    def finish(self, end_ns: int) -> dict:
        """Acquire right bracket, stop/join and return trace/metadata/errors."""
        ...
    def stop(self) -> dict:
        """Idempotent cleanup, including attempts without an output token."""
        ...


def error_record(error, *, stage=None, code=None):
    """Keep primary and cleanup errors as separate entries in caller-owned lists."""
    result = {'code': code or getattr(error, 'code', 'TECHNICAL_ERROR'),
              'stage': stage or getattr(error, 'stage', 'unknown'),
              'type': type(error).__name__, 'message': str(error)}
    receipt = getattr(error, 'receipt', None)
    if receipt is not None:
        result['receipt'] = receipt
    cause = error.__cause__
    if cause is not None:
        result['cause'] = {'type': type(cause).__name__, 'message': str(cause)}
    return result
