"""Sealed CPU-only measured LUT adapter; interpolate only the requested length."""
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from ..config import GROUPS
from ..errors import ContractError
from .types import (
    Bracket, DecodePrediction, FREQUENCIES_MHZ, LUTError, LUTProvenance,
    PERFORMANCE_LENGTHS, PrefillPrediction, QUALITY_LENGTHS, QualityPrediction,
    finite, group as validate_group, length as validate_length,
)

SOURCE_COMMIT = 'd6cf87a60984aa167c5b5d72de354fd3d587c2d5'
MANIFEST_SHA256 = '3b9755054b485ce81350bfbd4a675c4deff871c92784e5a20370211cac4029d9'
_FILE_NAMES = frozenset((
    'decode-lut.csv', 'decode-lut.json', 'decode-pwl.json',
    'fp16-reference-curves.json', 'prefill-lut.csv', 'prefill-lut.json',
    'prefill-pwl.json', 'quality-lut.json', 'source-export-manifest.json',
    'source-validation.json',
))


def _number(value, name, *, positive=False, quality=False):
    try:
        return finite(value, name, positive=positive, unit_interval=quality)
    except ContractError as error:
        raise LUTError(str(error)) from error


def _grid_keys(table, expected, name):
    # Check identifier types before equality (bool and float compare equal to int).
    if not hasattr(table, 'keys'):
        raise LUTError(f'{name} must be a complete mapping')
    for key in table:
        parts = key if type(key) is tuple else (key,)
        if any(type(part) is not int for part in parts):
            raise LUTError(f'{name} identifiers must be integers')
    if set(table) != expected:
        raise LUTError(f'{name} has missing, duplicate, or unsupported grid cells')


class _TabulatedLUT:
    """Private validated tables shared by measured and explicit synthetic sources."""

    def __init__(self, *, prefill, decode, quality, prefill_target, decode_target, provenance):
        _grid_keys(prefill, {(n, f) for n in PERFORMANCE_LENGTHS for f in FREQUENCIES_MHZ}, 'Prefill')
        _grid_keys(decode, {(n, g, f) for n in PERFORMANCE_LENGTHS for g in GROUPS for f in FREQUENCIES_MHZ}, 'Decode')
        _grid_keys(quality, {(n, g) for n in QUALITY_LENGTHS for g in GROUPS}, 'Quality')
        for name, values in [('Prefill target', prefill_target), ('Decode target', decode_target)]:
            _grid_keys(values, set(PERFORMANCE_LENGTHS), name)
        self._prefill = self._pairs(prefill, 'Prefill')
        self._decode = self._pairs(decode, 'Decode')
        self._quality = MappingProxyType({k: _number(v, 'mean_score', quality=True) for k, v in quality.items()})
        self._prefill_target = MappingProxyType({k: _number(v, 'Prefill target', positive=True) for k, v in prefill_target.items()})
        self._decode_target = MappingProxyType({k: _number(v, 'Decode target', positive=True) for k, v in decode_target.items()})
        self.provenance = provenance

    @staticmethod
    def _pairs(table, name):
        result = {}
        for key, values in table.items():
            if type(values) not in (tuple, list) or len(values) != 2:
                raise LUTError(f'{name} cell requires time and energy')
            result[key] = (_number(values[0], f'{name} time', positive=True), _number(values[1], f'{name} energy'))
        return MappingProxyType(result)

    def preflight(self, length: int) -> None:
        # Constructor has checked the full Cartesian product, both targets and
        # every required value. No online lookup or interpolation runs here.
        validate_length(length)

    def prefill(self, length: int) -> tuple[PrefillPrediction, ...]:
        bracket = _bracket(length, PERFORMANCE_LENGTHS)
        return tuple(PrefillPrediction(length, f, *self._pair_at(self._prefill, (f,), bracket), bracket) for f in FREQUENCIES_MHZ)

    def decode_within_budget(self, length: int, groups: tuple[int, ...],
                             budget_s: float) -> tuple[DecodePrediction, ...]:
        """Return active budget-eligible rows, or every global fastest TPOT tie.

        Interpolate TPOT first and energy only for returned rows. A negative
        planning budget is valid and takes the global latency fallback.
        """
        if type(groups) is not tuple or not groups:
            raise ContractError('Decode groups must be a nonempty immutable tuple')
        for value in groups:
            validate_group(value)
        if len(set(groups)) != len(groups):
            raise ContractError('Decode groups must be unique')
        if type(budget_s) not in (int, float):
            raise ContractError('Decode budget must be finite seconds/token')
        try:
            budget = float(budget_s)
        except (OverflowError, ValueError) as error:
            raise ContractError('Decode budget must be finite seconds/token') from error
        if not math.isfinite(budget):
            raise ContractError('Decode budget must be finite seconds/token')
        bracket = _bracket(length, PERFORMANCE_LENGTHS)
        times = {}

        def tpot(group, frequency_mhz):
            key = (group, frequency_mhz)
            if key not in times:
                times[key] = _at(self._decode[(bracket.lower, *key)][0],
                                 self._decode[(bracket.upper, *key)][0], bracket)
            return times[key]

        pairs = tuple((group, f) for group in sorted(groups) for f in FREQUENCIES_MHZ
                      if tpot(group, f) <= budget)
        if not pairs:
            all_pairs = tuple((group, f) for group in GROUPS for f in FREQUENCIES_MHZ)
            fastest = min(tpot(*pair) for pair in all_pairs)
            pairs = tuple(pair for pair in all_pairs if times[pair] == fastest)
        return tuple(DecodePrediction(
            length, group, f, times[(group, f)],
            _at(self._decode[(bracket.lower, group, f)][1],
                self._decode[(bracket.upper, group, f)][1], bracket), bracket)
            for group, f in pairs)

    def decode(self, length: int) -> tuple[DecodePrediction, ...]:
        bracket = _bracket(length, PERFORMANCE_LENGTHS)
        return tuple(DecodePrediction(length, g, f, *self._pair_at(self._decode, (g, f), bracket), bracket) for g in GROUPS for f in FREQUENCIES_MHZ)

    def quality(self, length: int) -> tuple[QualityPrediction, ...]:
        bracket = _bracket(length, QUALITY_LENGTHS)
        return tuple(QualityPrediction(length, g, _at(self._quality[(bracket.lower, g)], self._quality[(bracket.upper, g)], bracket), bracket) for g in GROUPS)

    def prefill_target(self, length: int) -> float:
        bracket = _bracket(length, PERFORMANCE_LENGTHS)
        return _at(self._prefill_target[bracket.lower], self._prefill_target[bracket.upper], bracket)

    def decode_target(self, length: int) -> float:
        bracket = _bracket(length, PERFORMANCE_LENGTHS)
        return _at(self._decode_target[bracket.lower], self._decode_target[bracket.upper], bracket)

    @staticmethod
    def _pair_at(table, candidate, bracket):
        lo, hi = table[(bracket.lower, *candidate)], table[(bracket.upper, *candidate)]
        return tuple(_at(a, b, bracket) for a, b in zip(lo, hi))


def _bracket(requested, grid):
    validate_length(requested)
    index = bisect_left(grid, requested)
    if index < len(grid) and grid[index] == requested:
        return Bracket(requested, requested, 0.0)
    if index == 0 or index == len(grid):
        raise LUTError('length extrapolation is unsupported')
    lo, hi = grid[index - 1], grid[index]
    return Bracket(lo, hi, (requested - lo) / (hi - lo))


def _at(lo, hi, bracket):
    # Returning an exact knot directly preserves its original binary64 value.
    if bracket.lower == bracket.upper:
        return lo
    return (1 - bracket.weight) * lo + bracket.weight * hi


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LUTError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise LUTError('nonfinite JSON number')
    return result


def _json(data, name):
    try:
        return json.loads(data, object_pairs_hook=_unique_object, parse_float=_json_float,
                          parse_constant=lambda value: (_ for _ in ()).throw(LUTError(f'nonfinite JSON number in {name}')))
    except (ValueError, UnicodeError) as error:
        raise LUTError(f'invalid JSON in {name}') from error


def _read(path):
    try:
        return path.read_bytes()
    except OSError as error:
        raise LUTError(f'missing or unreadable LUT file: {path.name}') from error


def _insert(table, key, value, name):
    if key in table:
        raise LUTError(f'duplicate {name} cell: {key}')
    table[key] = value


def _pwl(records, metric, expected_group, expected_frequency):
    if records.get('supported') is not True or records.get('interpolation') != 'piecewise_linear' or records.get('extrapolation') != 'unsupported':
        raise LUTError(f'unsupported PWL curve: {metric}')
    if type(records.get('group')) is not int or records['group'] != expected_group or type(records.get('gpu_mhz')) is not int or records['gpu_mhz'] != expected_frequency:
        raise LUTError('PWL candidate identifiers mismatch')
    knots = records['knots']
    if [k['length'] for k in knots] != list(PERFORMANCE_LENGTHS) or any(type(k['length']) is not int for k in knots):
        raise LUTError('PWL length grid is invalid')
    return {k['length']: _number(k['value'], metric, positive=metric in ('ttft_s', 'tpot_actual_s')) for k in knots}


class MeasuredLUTRepository(_TabulatedLUT):
    """Load only the pinned AGX2 repeat5 distribution, with no Mock fallback."""

    def __init__(self, bundle_path):
        root = Path(bundle_path).resolve()
        manifest_data = _read(root / 'bundle-manifest.json')
        if hashlib.sha256(manifest_data).hexdigest() != MANIFEST_SHA256:
            raise LUTError('unknown or modified bundle manifest SHA-256')
        manifest = _json(manifest_data, 'bundle-manifest.json')
        hashes = manifest['files_sha256']
        if set(hashes) != _FILE_NAMES:
            raise LUTError('manifest must seal exactly the ten published files')
        files = {}
        for name, expected in hashes.items():
            target = root / name
            if target.resolve().parent != root:
                raise LUTError('manifest file escapes the bundle')
            data = _read(target)
            if hashlib.sha256(data).hexdigest() != expected:
                raise LUTError(f'LUT SHA-256 mismatch: {name}')
            if name.endswith('.json'):
                files[name] = _json(data, name)
        try:
            self._load(files, manifest)
        except (KeyError, TypeError, IndexError, ContractError) as error:
            if isinstance(error, LUTError):
                raise
            raise LUTError(f'invalid sealed LUT structure: {error}') from error

    def _load(self, files, manifest):
        p, d = files['prefill-pwl.json'], files['decode-pwl.json']
        if set(p) != {str(f) for f in FREQUENCIES_MHZ} or set(d) != {f'{g}:{f}' for g in GROUPS for f in FREQUENCIES_MHZ}:
            raise LUTError('PWL candidate grid is incomplete or unsupported')
        prefill, decode, quality = {}, {}, {}
        for f in FREQUENCIES_MHZ:
            t = _pwl(p[str(f)]['ttft_s'], 'ttft_s', 8, f)
            e = _pwl(p[str(f)]['prefill_energy_j'], 'prefill_energy_j', 8, f)
            prefill.update({(n, f): (t[n], e[n]) for n in PERFORMANCE_LENGTHS})
        for g in GROUPS:
            for f in FREQUENCIES_MHZ:
                t = _pwl(d[f'{g}:{f}']['tpot_actual_s'], 'tpot_actual_s', g, f)
                e = _pwl(d[f'{g}:{f}']['epsilon_actual_j_per_token'], 'epsilon_actual_j_per_token', g, f)
                decode.update({(n, g, f): (t[n], e[n]) for n in PERFORMANCE_LENGTHS})
        q = files['quality-lut.json']
        for row in q['entries']:
            if row.get('supported') is not True or type(row['input_length_target']) is not int or type(row['group']) is not int:
                raise LUTError('invalid Quality cell support or identifiers')
            _insert(quality, (row['input_length_target'], row['group']), row['mean_score'], 'Quality')
        ref = files['fp16-reference-curves.json']
        rp, rd = {}, {}
        for row in ref['knots']:
            n = row['length']
            if type(n) is not int:
                raise LUTError('reference lengths must be integers')
            _insert(rp, n, row['R_p_ttft_s'], 'Prefill reference')
            _insert(rd, n, row['R_d_tpot_s'], 'Decode reference')
        metadata = {
            'bundle': manifest, 'quality': q,
            'source_export': files['source-export-manifest.json'],
            'source_validation': files['source-validation.json'],
            'reference': {k: v for k, v in ref.items() if k not in ('knots', 'requests')},
        }
        provenance = LUTProvenance(
            kind='measured', source_device=manifest['source_device'], source_commit=SOURCE_COMMIT,
            manifest_sha256=MANIFEST_SHA256, files_sha256=tuple(sorted(manifest['files_sha256'].items())),
            notes=tuple(manifest['limitations']), metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
        )
        super().__init__(prefill=prefill, decode=decode, quality=quality, prefill_target=rp, decode_target=rd, provenance=provenance)
