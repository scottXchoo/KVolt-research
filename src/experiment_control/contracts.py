"""Dependency-free contracts for detachable experiment environment controls."""
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol


class EnvironmentError(RuntimeError):
    """Environment preparation, observation, or cleanup failed."""


class ReadinessError(EnvironmentError):
    pass


class ViolationError(EnvironmentError):
    pass


class RestorationError(EnvironmentError):
    pass


def finite_number(value, name, *, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


@dataclass(frozen=True)
class Reading:
    """Missing values stay None, with a reason; never substitute zero/False."""
    values: Mapping[str, Any] = field(default_factory=dict)
    errors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.values, Mapping) or not isinstance(self.errors, Mapping):
            raise ValueError('reading values/errors must be mappings')
        for key, value in self.values.items():
            if not isinstance(key, str) or not key:
                raise ValueError('metric names must be nonempty strings')
            if value is not None and type(value) not in (str, bool, int, float):
                raise ValueError(f'metric {key} must be a JSON scalar or None')
            if type(value) in (int, float):
                finite_number(value, key)
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.errors.items()):
            raise ValueError('reading errors must map strings to strings')


class Probe(Protocol):
    name: str

    def sample(self) -> Reading: ...


class Control(Protocol):
    """capture all controls before applying any; restore even partial apply."""
    name: str

    def capture(self) -> Any: ...
    def apply(self, snapshot: Any) -> None: ...
    def restore(self, snapshot: Any) -> None: ...


@dataclass(frozen=True)
class MetricRule:
    probe: str
    metric: str
    minimum: float | None = None
    maximum: float | None = None
    expected: Any = None
    max_increase: float | None = None
    max_span: float | None = None
    required: bool = True
    phases: tuple[str, ...] = ('ready', 'run')

    def __post_init__(self):
        if any(type(v) is not str or not v for v in (self.probe, self.metric)):
            raise ValueError('rule probe/metric must be nonempty strings')
        for key in ('minimum', 'maximum', 'max_increase', 'max_span'):
            value = getattr(self, key)
            if value is not None:
                finite_number(value, key, minimum=0 if key in ('max_increase', 'max_span') else None)
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError('rule minimum exceeds maximum')
        if type(self.required) is not bool:
            raise ValueError('rule required must be bool')
        if (not isinstance(self.phases, tuple) or not self.phases or
                len(set(self.phases)) != len(self.phases) or any(p not in ('ready', 'run') for p in self.phases)):
            raise ValueError('rule phases must contain ready and/or run')
        if self.expected is not None:
            Reading({'expected': self.expected})
