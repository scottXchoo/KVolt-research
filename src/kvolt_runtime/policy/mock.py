"""Explicit, deterministic synthetic fixtures with the same full support grids."""
import json

from ..config import GROUPS
from .lut import _TabulatedLUT
from .types import FREQUENCIES_MHZ, LUTError, LUTProvenance, PERFORMANCE_LENGTHS, QUALITY_LENGTHS


class MockLUTRepository(_TabulatedLUT):
    """Synthetic scenarios, optionally replacing complete tables with fixtures.

    Keys: Prefill (L,f), Decode (L,G,f), Quality (L,G), targets L.
    Performance values are (seconds,joules); Quality/targets are scalars.
    Provided tables must contain exactly the full canonical grid.
    """

    def __init__(self, scenario='baseline', *, prefill=None, decode=None, quality=None,
                 prefill_target=None, decode_target=None):
        if scenario not in ('baseline', 'flat_quality', 'nonmonotonic', 'exact_ties'):
            raise LUTError(f'unknown synthetic LUT scenario: {scenario!r}')
        ties = scenario == 'exact_ties'
        prefill_default = {
            (n, f): (n / 2048 if ties else n / 1024 * 408 / f,
                     n / 1024 * (2 if ties else 1 + f / 1122))
            for n in PERFORMANCE_LENGTHS for f in FREQUENCIES_MHZ
        }
        decode_default = {
            (n, g, f): (0.05 + n / 819200 if ties else (0.04 + n / 819200 + 1 / g) * 408 / f,
                        0.1 if ties else (0.02 + 1 / g) * (1 + f / 1122))
            for n in PERFORMANCE_LENGTHS for g in GROUPS for f in FREQUENCIES_MHZ
        }
        base = (0.99, 0.98, 0.95, 0.85, 0.65)
        if scenario in ('flat_quality', 'exact_ties'):
            base = (0.9,) * 5
        elif scenario == 'nonmonotonic':
            base = (0.99, 0.80, 0.94, 0.70, 0.76)
        quality_default = {(n, g): base[index] for n in QUALITY_LENGTHS for index, g in enumerate(GROUPS)}
        provenance = LUTProvenance(
            kind='synthetic', source_device='synthetic',
            notes=('Synthetic policy fixture; no measured runtime qualification.',),
            metadata_json=json.dumps({'synthetic': True, 'scenario': scenario,
                                      'custom_tables': [name for name, table in [('prefill', prefill), ('decode', decode), ('quality', quality), ('prefill_target', prefill_target), ('decode_target', decode_target)] if table is not None]}),
        )
        super().__init__(
            prefill=prefill_default if prefill is None else prefill,
            decode=decode_default if decode is None else decode,
            quality=quality_default if quality is None else quality,
            prefill_target={n: n / 1024 for n in PERFORMANCE_LENGTHS} if prefill_target is None else prefill_target,
            decode_target={n: 0.08 + n / 819200 for n in PERFORMANCE_LENGTHS} if decode_target is None else decode_target,
            provenance=provenance,
        )
