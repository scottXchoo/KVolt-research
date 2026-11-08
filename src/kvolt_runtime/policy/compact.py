"""Pinned compact predictions with explicitly supplied pinned AGX2 FP16 targets."""
import hashlib
import json
from pathlib import Path

from ..config import GROUPS
from .lut import (
    MANIFEST_SHA256 as REFERENCE_MANIFEST_SHA256, MeasuredLUTRepository,
    _TabulatedLUT, _insert, _json, _read,
)
from .reference import (
    COMPACT_REFERENCE_MANIFEST_SHA256S, CompactReferenceRepository, load_reference_repository,
)
from .types import FREQUENCIES_MHZ, LUTError, LUTProvenance, PERFORMANCE_LENGTHS, QUALITY_LENGTHS

COMPACT_MANIFEST_SHA256 = '86b2343eed65432d1ee1f071fa8188a33789bdcf0309f325e689ec0bae37591a'
AGX2_COMPACT_MANIFEST_SHA256 = 'b421fcd10ecb57df7d6962aa9a5f48e264e90defc0c76a3544218d31685f783b'
_COMMON_CONTEXT = {
    'model_id': 'meta-llama/Llama-3.1-8B-Instruct',
    'model_revision': '0e9e39f249a16976918f6564b8830bc894c89659',
    'weight_dtype': 'float16', 'kv_precision': 'K2V2', 'residual_length': 128,
    'cpu_khz': 2201600, 'emc_hz': 3199000000, 'prefill_cache': 'original_fp16',
    'performance_generation': {'force_length': False, 'max_new_tokens': 512},
    'quality_generation': {'do_sample': False, 'force_length': False,
                           'max_new_tokens': 128, 'num_beams': 1},
    'quality_dataset': 'RULER', 'quality_metric': 'RULER_string_match_all',
    'quality_items_per_length': 100,
}
# Only these exact original manifests are trusted. Context contracts also guard
# semantic assumptions independently of the byte/hash boundary.
_TRUSTED_COMPACT_MANIFESTS = {
    COMPACT_MANIFEST_SHA256: {
        **_COMMON_CONTEXT, 'performance_device_id': 'A1',
        'decode_measurement_boundary': 't1..tN includes actual callback and initial K2V2 packing once',
    },
    AGX2_COMPACT_MANIFEST_SHA256: {
        **_COMMON_CONTEXT, 'performance_device_id': 'A2',
        'performance_repeats_per_group': 5,
        'prefill_aggregation': {
            'method': 'arithmetic_mean_over_all_group_request_observations',
            'groups': list(GROUPS), 'repeats_per_group': 5, 'samples_per_cell': 25,
            'total_requests': 1600, 'documents_per_length': 1,
        },
        'decode_aggregation': {'method': 'source_cell_means_preserved', 'repeats_per_cell': 5},
        'decode_measurement_boundary': 't1..tN, includes remaining controller work and initial K2V2 packing once',
        'quality_source_device_id': 'A1',
    },
}
_TABLES = {
    'prefill': (('length', 'gpu_mhz', 'ttft_s', 'prefill_energy_j'),
                {'length': PERFORMANCE_LENGTHS, 'gpu_mhz': FREQUENCIES_MHZ}),
    'decode': (('length', 'group', 'gpu_mhz', 'tpot_actual_s', 'epsilon_actual_j_per_token'),
               {'length': PERFORMANCE_LENGTHS, 'group': GROUPS, 'gpu_mhz': FREQUENCIES_MHZ}),
    'quality': (('length', 'group', 'mean_score'),
                {'length': QUALITY_LENGTHS, 'group': GROUPS}),
}
_UNITS = {
    'length': 'tokens', 'gpu_mhz': 'MHz', 'group': 'elements per quantization group',
    'ttft_s': 's', 'prefill_energy_j': 'J', 'tpot_actual_s': 's/token',
    'epsilon_actual_j_per_token': 'J/token', 'mean_score': 'raw score, 0..1',
}


def _manifest_contract(manifest, expected_context):
    if (type(manifest) is not dict or manifest.get('schema') != 'kvolt.runtime-lut.v1'
            or manifest.get('units') != _UNITS
            or manifest.get('interpolation') != {'axis': 'length', 'method': 'piecewise_linear', 'extrapolation': False}):
        raise LUTError('unsupported compact LUT schema, units or interpolation')
    # JSON comparison distinguishes bools from integer measurement counts.
    if json.dumps(manifest['context'], sort_keys=True, allow_nan=False) != json.dumps(expected_context, sort_keys=True):
        raise LUTError('compact LUT source context does not match its trusted manifest contract')
    validation = manifest['validation']
    if (validation['performance_runtime_accuracy_or_SLO_validated'] is not False
            or validation['quality_runtime_eligible'] is not False):
        raise LUTError('compact LUT runtime accuracy/SLO and quality remain unvalidated')
    if set(manifest['tables']) != set(_TABLES):
        raise LUTError('compact manifest must describe all three tables')
    for name, (columns, axes) in _TABLES.items():
        spec = manifest['tables'][name]
        if spec['file'] != name + '.json' or spec['columns'] != list(columns) or set(spec['axes']) != set(axes):
            raise LUTError(f'invalid compact table contract: {name}')
        expected_rows = 1
        for axis, expected in axes.items():
            values = spec['axes'][axis]
            if type(values) is not list or any(type(v) is not int for v in values) or values != list(expected):
                raise LUTError(f'invalid compact axis: {name}.{axis}')
            expected_rows *= len(expected)
        if type(spec['rows']) is not int or spec['rows'] != expected_rows:
            raise LUTError(f'invalid compact row count: {name}')


def _table(data, name, spec):
    rows = _json(data, spec['file'])
    columns, axes = _TABLES[name]
    width = len(axes)
    if type(rows) is not list or len(rows) != spec['rows']:
        raise LUTError(f'{name} must contain the complete compact row grid')
    table = {}
    for row in rows:
        if type(row) is not list or len(row) != len(columns):
            raise LUTError(f'invalid compact {name} row width or type')
        key = tuple(row[:width])
        if any(type(part) is not int for part in key):
            raise LUTError(f'{name} identifiers must be integers')
        if any(part not in axis for part, axis in zip(key, axes.values())):
            raise LUTError(f'unsupported compact {name} grid cell: {key}')
        _insert(table, key, row[width] if name == 'quality' else row[width:], name)
    # _TabulatedLUT validates the full Cartesian product and every numeric value.
    return table


def _target_metadata(repository):
    provenance = repository.provenance
    raw = json.loads(provenance.metadata_json)
    result = {
        'kind': provenance.kind, 'source_device': provenance.source_device,
        'source_commit': provenance.source_commit, 'manifest_sha256': provenance.manifest_sha256,
        'files_sha256': dict(provenance.files_sha256),
    }
    if type(repository) is CompactReferenceRepository:
        manifest = raw['reference_manifest']
        result.update(source_conditions=manifest['context'], reference=manifest,
                      compact_reference_manifest=manifest)
    else:
        result.update(source_conditions={k: v for k, v in raw['bundle'].items() if k != 'files_sha256'},
                      reference=raw['reference'])
    return result


class CompactLUTRepository(_TabulatedLUT):
    """Load 459 pinned A1/A2 cells; keep separately supplied AGX2 FP16 targets unchanged.

    Development predictions establish no accuracy, SLO or formal
    qualification. The source's false runtime eligibility flags remain intact.
    """

    def __init__(self, compact_path, *, reference_repository):
        reference_pins = {
            MeasuredLUTRepository: {REFERENCE_MANIFEST_SHA256},
            CompactReferenceRepository: COMPACT_REFERENCE_MANIFEST_SHA256S,
        }
        expected_reference_pin = reference_pins.get(type(reference_repository))
        if (expected_reference_pin is None
                or reference_repository.provenance.kind != 'measured'
                or reference_repository.provenance.manifest_sha256 not in expected_reference_pin
                or reference_repository.provenance.source_device.casefold() != 'agx2'):
            raise LUTError('compact LUT requires a sealed AGX2 MeasuredLUTRepository or pinned '
                           'CompactReferenceRepository as explicit reference_repository')
        root = Path(compact_path).resolve()
        data = _read(root / 'manifest.json')
        manifest_sha256 = hashlib.sha256(data).hexdigest()
        expected_context = _TRUSTED_COMPACT_MANIFESTS.get(manifest_sha256)
        if expected_context is None:
            raise LUTError('unknown or modified compact manifest SHA-256')
        try:
            manifest = _json(data, 'manifest.json')
            _manifest_contract(manifest, expected_context)
            tables, hashes = {}, {}
            for name in _TABLES:
                spec = manifest['tables'][name]
                target = root / spec['file']
                if target.resolve().parent != root:
                    raise LUTError('compact LUT file escapes the bundle')
                raw = _read(target)
                if hashlib.sha256(raw).hexdigest() != spec['sha256']:
                    raise LUTError(f'LUT SHA-256 mismatch: {spec["file"]}')
                if type(spec['bytes']) is not int or len(raw) != spec['bytes']:
                    raise LUTError(f'compact LUT byte count mismatch: {spec["file"]}')
                hashes[spec['file']] = spec['sha256']
                tables[name] = _table(raw, name, spec)
            source_device = {'A1': 'agx1', 'A2': 'agx2'}[manifest['context']['performance_device_id']]
            target_device = reference_repository.provenance.source_device
            cross_device = source_device.casefold() != target_device.casefold()
            metadata = {
                'compact_manifest': manifest,
                'target_repository': _target_metadata(reference_repository),
                'target_source_device': reference_repository.provenance.source_device,
                'cross_device_reference': cross_device,
            }
            provenance = LUTProvenance(
                kind='measured', source_device=source_device, source_commit=None,
                manifest_sha256=manifest_sha256, files_sha256=tuple(sorted(hashes.items())),
                notes=(
                    f'{source_device.upper()} compact measured predictions; source runtime accuracy/SLO and quality eligibility remain unvalidated.',
                    'Targets are explicitly supplied pinned AGX2 FP16 references, separate from compact predictions.',
                    'Development use, including matching devices, does not establish runtime accuracy, SLO or formal qualification.',
                ),
                metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
            )
            super().__init__(
                **tables,
                prefill_target={n: reference_repository.prefill_target(n) for n in PERFORMANCE_LENGTHS},
                decode_target={n: reference_repository.decode_target(n) for n in PERFORMANCE_LENGTHS},
                provenance=provenance,
            )
        except (KeyError, TypeError, IndexError, ValueError) as error:
            if isinstance(error, LUTError):
                raise
            raise LUTError(f'invalid compact LUT structure: {error}') from error


def load_lut_repository(path, *, reference_lut_dir=None):
    """Select one recognized distribution; compact arrays require explicit targets."""
    root = Path(path)
    sealed = (root / 'bundle-manifest.json').exists()
    compact = any((root / name).exists() for name in ('manifest.json', 'prefill.json', 'decode.json', 'quality.json'))
    if sealed and compact:
        raise LUTError('ambiguous LUT directory contains sealed and compact formats')
    if compact:
        if reference_lut_dir is None:
            raise LUTError('compact LUT requires explicit --reference-lut-dir with pinned AGX2 FP16 targets')
        reference = load_reference_repository(reference_lut_dir)
        return CompactLUTRepository(root, reference_repository=reference)
    if sealed:
        if reference_lut_dir is not None:
            raise LUTError('--reference-lut-dir applies only to compact LUTs; sealed bundle already contains targets')
        return MeasuredLUTRepository(root)
    raise LUTError('unknown LUT directory format: expected a sealed bundle or compact arrays with manifest.json')
