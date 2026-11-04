"""Fixed M1 model settings. No dependency installation or GPU imports."""

from dataclasses import dataclass, fields

from .errors import ContractError

SNAPSHOT = '0e9e39f249a16976918f6564b8830bc894c89659'
GROUPS = (8, 16, 32, 64, 128)
EOS_TOKEN_IDS = (128001, 128008, 128009)
MISSING = object()


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
