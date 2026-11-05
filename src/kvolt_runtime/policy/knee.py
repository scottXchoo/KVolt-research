"""Pinned Kneedle adapter with a narrow, same-request detector recovery boundary."""
import importlib
import math
from numbers import Real
from typing import Callable

from ..config import GROUPS
from ..errors import ContractError
from .types import Bracket, KneeResult, PolicyDependencyError, QualityPrediction


def validate_quality(quality, requested_length=None):
    """Require every original G and valid raw scores before detector execution."""
    if type(quality) is not tuple or len(quality) != len(GROUPS):
        raise ContractError('Quality LUT must return an immutable tuple of all five G')
    normalized = []
    for row in quality:
        if not isinstance(row, QualityPrediction):
            raise ContractError('Quality LUT must return QualityPrediction rows')
        if not isinstance(row.bracket, Bracket):
            raise ContractError('Quality prediction requires an interpolation bracket')
        # Retain canonical binary64 copies, including nested bracket metadata.
        # A foreign frozen dataclass may have been altered through object.__setattr__.
        bracket = Bracket(row.bracket.lower, row.bracket.upper, row.bracket.weight)
        normalized.append(QualityPrediction(row.length, row.group, row.score, bracket))
    if {row.group for row in normalized} != set(GROUPS):
        raise ContractError('Quality LUT must contain each original G exactly once')
    if len({row.length for row in normalized}) != 1:
        raise ContractError('Quality LUT rows must have the same L')
    if requested_length is not None and normalized[0].length != requested_length:
        raise ContractError('Quality LUT rows must match requested L')
    return tuple(sorted(normalized, key=lambda row: row.group))


class KneeAdapter:
    """Use the fixed package pins; factory injection is for isolated unit tests.

    Package loading/version validation happens at construction, outside the
    recoverable detector call. Invalid source Quality data is also fatal.
    """

    def __init__(self, locator_factory: Callable | None = None):
        if locator_factory is not None:
            if not callable(locator_factory):
                raise ContractError('locator_factory must be callable')
            self._locator_factory = locator_factory
            return
        try:
            packages = {}
            for name, expected in (('kneed', '0.9.0'), ('scipy', '1.15.3')):
                package = importlib.import_module(name)
                observed = getattr(package, '__version__', None)
                if observed != expected:
                    raise PolicyDependencyError(
                        f'{name}=={expected} required; found {observed!r}')
                packages[name] = package
            self._locator_factory = packages['kneed'].KneeLocator
            if not callable(self._locator_factory):
                raise PolicyDependencyError('kneed.KneeLocator must be callable')
        except PolicyDependencyError:
            raise
        except Exception as error:
            raise PolicyDependencyError(
                f'Cannot load pinned kneed==0.9.0/SciPy==1.15.3: {error}') from error

    def detect(self, quality: tuple[QualityPrediction, ...]) -> KneeResult:
        rows = validate_quality(quality)
        values = [row.score for row in rows]
        if all(value == values[0] for value in values):
            return KneeResult('not_found', reason='flat_quality')

        try:
            locator = self._locator_factory(
                x=list(GROUPS), y=values, S=1.0, curve='concave',
                direction='decreasing', interp_method='interp1d', online=True)
            result = locator.all_knees
            # kneed 0.9.0 exposes a list, not the set used by older releases.
            # Method applies SET(result) after validating every returned item.
            if not isinstance(result, (list, tuple, set, frozenset)):
                raise ValueError('all_knees must be a list, tuple, set or frozenset of original G')
            for item in result:
                if (isinstance(item, bool) or not isinstance(item, Real)
                        or not math.isfinite(item) or item not in GROUPS):
                    raise ValueError('all_knees contains a nonfinite or noncandidate G')
            groups = tuple(sorted(set(int(item) for item in result)))
            if not groups:
                return KneeResult('not_found', reason='no_knee')
            return KneeResult('found', groups)
        except Exception as error:
            return KneeResult('error_recovered', reason='detector_error',
                              error_type=type(error).__name__, error_message=str(error))
