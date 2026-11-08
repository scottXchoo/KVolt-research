"""Frozen binary64 predictions and controller decisions shared by M2 modules."""
from dataclasses import dataclass
import math
from typing import Protocol

from ..config import GROUPS, require_int
from ..errors import ContractError, KVoltError

FREQUENCIES_MHZ = (408, 510, 612, 714, 816, 918, 1020, 1122)
PERFORMANCE_LENGTHS = tuple(range(1024, 8193, 1024))
QUALITY_LENGTHS = tuple(range(1024, 8193, 512))


class LUTError(ContractError):
    """Missing, unsupported, corrupt, or incompatible LUT data; reject early."""


class PolicyDependencyError(KVoltError, RuntimeError):
    """Pinned policy dependencies are unavailable; not a detector recovery."""


def length(value):
    return require_int(value, 'L', 1024, 8192)


def frequency(value):
    if type(value) is not int or value not in FREQUENCIES_MHZ:
        raise ContractError('frequency must be one of the eight integer MHz candidates')
    return value


def group(value):
    if type(value) is not int or value not in GROUPS:
        raise ContractError('G must be one of 8/16/32/64/128')
    return value


def finite(value, name, *, positive=False, unit_interval=False):
    if type(value) not in (int, float):
        raise ContractError(f'{name} must be a finite binary64 number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ContractError(f'{name} must be finite') from error
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ContractError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    if unit_interval and result > 1:
        raise ContractError(f'{name} must be a raw score in [0,1]')
    return result


def elapsed_since(start_ns, now_ns):
    require_int(start_ns, 'phase start ns', 0)
    require_int(now_ns, 'phase current ns', 0)
    if now_ns < start_ns:
        raise ContractError('phase clock moved backwards')
    return now_ns - start_ns


@dataclass(frozen=True)
class Bracket:
    lower: int
    upper: int
    weight: float

    def __post_init__(self):
        length(self.lower)
        length(self.upper)
        if self.lower > self.upper:
            raise ContractError('interpolation bracket is reversed')
        object.__setattr__(self, 'weight', finite(self.weight, 'interpolation weight', unit_interval=True))
        if self.lower == self.upper and self.weight != 0:
            raise ContractError('an exact knot requires weight zero')


@dataclass(frozen=True)
class PrefillLatencyPrediction:
    """Time-only screening evidence; an unqueried energy remains null."""

    length: int
    frequency_mhz: int
    ttft_s: float
    bracket: Bracket
    energy_j: None = None

    def __post_init__(self):
        length(self.length)
        frequency(self.frequency_mhz)
        object.__setattr__(self, 'ttft_s', finite(self.ttft_s, 'TTFT seconds', positive=True))
        _bracket(self.bracket, self.length)
        if self.energy_j is not None:
            raise ContractError('latency-only Prefill prediction cannot contain energy')


@dataclass(frozen=True)
class PrefillPrediction:
    length: int
    frequency_mhz: int
    ttft_s: float
    energy_j: float
    bracket: Bracket

    def __post_init__(self):
        length(self.length)
        frequency(self.frequency_mhz)
        object.__setattr__(self, 'ttft_s', finite(self.ttft_s, 'TTFT seconds', positive=True))
        object.__setattr__(self, 'energy_j', finite(self.energy_j, 'Prefill joules'))
        _bracket(self.bracket, self.length)


@dataclass(frozen=True)
class DecodePrediction:
    length: int
    group: int
    frequency_mhz: int
    tpot_s: float
    energy_j_per_token: float
    bracket: Bracket

    def __post_init__(self):
        length(self.length)
        group(self.group)
        frequency(self.frequency_mhz)
        object.__setattr__(self, 'tpot_s', finite(self.tpot_s, 'actual TPOT seconds/token', positive=True))
        object.__setattr__(self, 'energy_j_per_token', finite(self.energy_j_per_token, 'actual joules/Decode token'))
        _bracket(self.bracket, self.length)


@dataclass(frozen=True)
class QualityPrediction:
    length: int
    group: int
    score: float
    bracket: Bracket

    def __post_init__(self):
        length(self.length)
        group(self.group)
        object.__setattr__(self, 'score', finite(self.score, 'RULER mean_score', unit_interval=True))
        _bracket(self.bracket, self.length)


def _bracket(value, requested):
    if not isinstance(value, Bracket) or not value.lower <= requested <= value.upper:
        raise ContractError('prediction requires a bracket containing L')
    expected = 0.0 if value.lower == value.upper else (requested - value.lower) / (value.upper - value.lower)
    if value.weight != expected:
        raise ContractError('bracket weight must match L exactly')


@dataclass(frozen=True)
class PrefillLookup:
    """Prefill predictions and the FP16 goal at one shared length bracket."""

    length: int
    predictions: tuple[PrefillPrediction, ...] | tuple[PrefillLatencyPrediction, ...]
    target_s: float
    bracket: Bracket

    def __post_init__(self):
        length(self.length)
        object.__setattr__(self, 'target_s', finite(self.target_s, 'Prefill target seconds', positive=True))
        _bracket(self.bracket, self.length)
        if type(self.predictions) is not tuple:
            raise ContractError('Prefill lookup predictions must be an immutable tuple')


@dataclass(frozen=True)
class DecodeLookup:
    """Decode predictions and the FP16 goal at one shared length bracket."""

    length: int
    predictions: tuple[DecodePrediction, ...]
    target_s: float
    bracket: Bracket

    def __post_init__(self):
        length(self.length)
        object.__setattr__(self, 'target_s', finite(self.target_s, 'Decode target seconds/token', positive=True))
        _bracket(self.bracket, self.length)
        if type(self.predictions) is not tuple:
            raise ContractError('Decode lookup predictions must be an immutable tuple')


@dataclass(frozen=True)
class LUTProvenance:
    kind: str
    source_device: str
    source_commit: str | None = None
    manifest_sha256: str | None = None
    files_sha256: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    metadata_json: str = '{}'


class LUTRepository(Protocol):
    provenance: LUTProvenance

    def preflight(self, length: int) -> None:
        """Check full support without invoking online interpolation/detection."""
        ...

    def prefill(self, length: int) -> tuple[PrefillPrediction, ...]: ...
    def prefill_lookup(self, length: int, *, latency_only: bool = False) -> PrefillLookup: ...
    def prefill_within_target(self, length: int, target_s: float) -> PrefillLookup: ...
    def prefill_latencies(self, length: int) -> tuple[PrefillLatencyPrediction, ...]: ...
    def prefill_candidates(self, length: int, frequencies: tuple[int, ...]) -> tuple[PrefillPrediction, ...]: ...
    def quality(self, length: int) -> tuple[QualityPrediction, ...]: ...
    def decode(self, length: int) -> tuple[DecodePrediction, ...]: ...
    def decode_lookup(self, length: int) -> DecodeLookup: ...
    def prefill_target(self, length: int) -> float: ...
    def decode_target(self, length: int) -> float: ...


@dataclass(frozen=True)
class KneeResult:
    status: str
    groups: tuple[int, ...] = ()
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self):
        if self.status not in ('found', 'not_found', 'error_recovered'):
            raise ContractError('unknown Knee status')
        if type(self.groups) is not tuple:
            raise ContractError('Knee groups must be an immutable tuple')
        for item in self.groups:
            group(item)
        if self.groups != tuple(sorted(set(self.groups))):
            raise ContractError('Knee groups must be sorted and unique')
        if (self.status == 'found') != bool(self.groups):
            raise ContractError('only a found Knee has nonempty groups')


@dataclass(frozen=True)
class PrefillCandidate:
    prediction: PrefillPrediction | PrefillLatencyPrediction
    meets_budget: bool
    selected: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecodeCandidate:
    prediction: DecodePrediction
    quality: float
    active_knee: bool
    meets_budget: bool
    eligible: bool
    selected: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrefillDecision:
    length: int
    selected: PrefillPrediction
    target_s: float
    budget_s: float
    elapsed_ns: int
    selection_path: str
    predicted_feasible: bool
    candidates: tuple[PrefillCandidate, ...]
    t0_ns: int
    tp_ns: int
    prefilter_ns: int
    selection_rule: str = 'target_ttft'


@dataclass(frozen=True)
class DecodeDecision:
    length: int
    nmax: int
    first_token_id: int
    status: str
    selection_path: str
    selected: DecodePrediction | None = None
    selected_quality: float | None = None
    target_s: float | None = None
    budget_s: float | None = None
    elapsed_ns: int | None = None
    predicted_feasible: bool | None = None
    quality: tuple[QualityPrediction, ...] = ()
    knee: KneeResult | None = None
    candidates: tuple[DecodeCandidate, ...] = ()
    t1_ns: int | None = None
    td_ns: int | None = None
    skip_reason: str | None = None
