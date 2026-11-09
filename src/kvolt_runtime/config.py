"""Fixed M1 model settings and global policy. No GPU imports or installation."""

from dataclasses import dataclass, fields
import math

from .errors import ContractError

SNAPSHOT = '0e9e39f249a16976918f6564b8830bc894c89659'
GROUPS = (8, 16, 32, 64, 128)
EOS_TOKEN_IDS = (128001, 128008, 128009)
MISSING = object()
# Relative TPOT allowance over the immutable FP16 Reference. 0.20 means +20%.
# Change this one value before starting a new runtime/diagnostic process.
TPOT_BUFFER_RATIO = 0.20


def resolve_tpot_buffer_ratio(value: object = MISSING) -> float:
    """Resolve and validate a ratio before freezing it for a controller/session."""
    value = TPOT_BUFFER_RATIO if value is MISSING else value
    if type(value) not in (int, float):
        raise ContractError('TPOT buffer ratio must be a finite nonnegative number')
    try:
        ratio = float(value)
    except (OverflowError, ValueError) as error:
        raise ContractError('TPOT buffer ratio must be a finite nonnegative number') from error
    if not math.isfinite(ratio) or ratio < 0:
        raise ContractError('TPOT buffer ratio must be a finite nonnegative number')
    return ratio


def buffered_tpot_target(reference_s: object, buffer_ratio: object) -> float:
    """Apply the allowance once, after raw Reference length interpolation."""
    if type(reference_s) not in (int, float):
        raise ContractError('TPOT Reference must be finite positive seconds/token')
    try:
        reference = float(reference_s)
    except (OverflowError, ValueError) as error:
        raise ContractError('TPOT Reference must be finite positive seconds/token') from error
    if not math.isfinite(reference) or reference <= 0:
        raise ContractError('TPOT Reference must be finite positive seconds/token')
    target = reference * (1.0 + resolve_tpot_buffer_ratio(buffer_ratio))
    if not math.isfinite(target) or target <= 0:
        raise ContractError('buffered TPOT target must be finite positive seconds/token')
    return target


def require_int(value: object, name: str, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def resolve_nmax(*, item: object = MISSING, cli: object = MISSING) -> int:
    """Missing differs from explicit null. Validate every supplied value."""
    for name, value in [('CLI Nmax', cli), ('item Nmax', item)]:
        if value is not MISSING:
            require_int(value, name, 1, 512)
    return 512 if item is MISSING and cli is MISSING else (cli if item is MISSING else item)


@dataclass(frozen=True)
class RuntimeConfig:
    snapshot: str = SNAPSHOT
    dtype: str = 'float16'
    batch_size: int = 1
    key_bits: int = 2
    value_bits: int = 2
    residual_length: int = 128
    groups: tuple[int, ...] = GROUPS
    eos_token_ids: tuple[int, ...] = EOS_TOKEN_IDS
    do_sample: bool = False
    num_beams: int = 1

    def __post_init__(self):
        for field in fields(self):
            actual, expected = getattr(self, field.name), field.default
            same_type = type(actual) is type(expected)
            if same_type and isinstance(expected, tuple):
                same_type = len(actual) == len(expected) and all(
                    type(value) is type(fixed) for value, fixed in zip(actual, expected)
                )
            if not same_type or actual != expected:
                raise ContractError(f'M1 fixes {field.name}={expected!r}; got {actual!r}')
