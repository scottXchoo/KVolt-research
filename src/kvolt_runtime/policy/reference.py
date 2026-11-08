"""Pinned, target-only FP16 references supplied separately from predictions."""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from .lut import MeasuredLUTRepository, _at, _bracket, _json, _number, _read
from .types import LUTError, LUTProvenance, PERFORMANCE_LENGTHS, length as validate_length


COMPACT_REFERENCE_MANIFEST_SHA256 = '00d5112eaf9b0cdad2420b7622522c5d0913196e47681a286c99e19eeeeae15c'
COMPACT_REFERENCE_REPEAT5_MANIFEST_SHA256 = 'd61c1b6a6ccd0aaa80b0cf4645702eaee09bd406dac084df70f151669a9b74ee'
COMPACT_REFERENCE_REPEAT10_MANIFEST_SHA256 = 'd332875ccce626dd99700da8c7a128bf4801577333e76ee40a987f4051311e40'
_CONTEXT = {
    'device_id': 'A2', 'model_id': 'meta-llama/Llama-3.1-8B-Instruct',
    'model_revision': '0e9e39f249a16976918f6564b8830bc894c89659',
    'weight_dtype': 'float16', 'kv_precision': 'FP16',
    'generation': {'force_length': False, 'max_new_tokens': 512},
    'cpu_khz': 2201600, 'emc_hz': 3199000000,
    'gpu_policy': 'default_governor_cap1122', 'gpu_governor': 'nvhost_podgov',
    'gpu_min_hz': 306000000, 'gpu_max_hz': 1122000000,
    'technical_repeats_per_length': 3, 'total_requests': 24,
    'documents_per_length': 1,
    'aggregation': 'independent_maximum_TTFT_and_request_mean_TPOT_per_length',
    'prefill_boundary': 't0..t1', 'decode_TPOT': '(tN-t1)/(N-1); only N>1',
}
_TRUSTED_REFERENCE_MANIFESTS = {
    COMPACT_REFERENCE_MANIFEST_SHA256: _CONTEXT,
    COMPACT_REFERENCE_REPEAT5_MANIFEST_SHA256: {
        **_CONTEXT, 'technical_repeats_per_length': 5, 'total_requests': 40,
    },
    COMPACT_REFERENCE_REPEAT10_MANIFEST_SHA256: {
        **_CONTEXT, 'technical_repeats_per_length': 10, 'total_requests': 80,
    },
}
COMPACT_REFERENCE_MANIFEST_SHA256S = frozenset(_TRUSTED_REFERENCE_MANIFESTS)


def _manifest_contract(manifest, expected_context):
    if (type(manifest) is not dict
            or manifest.get('schema') != 'kvolt.reference-lut.v1'
            or manifest.get('units') != {'length': 'tokens', 'r_p_ttft_s': 's', 'r_d_tpot_s': 's/token'}
            or manifest.get('interpolation') != {
                'axis': 'length', 'method': 'piecewise_linear', 'extrapolation': False}):
        raise LUTError('unsupported compact reference schema, units or interpolation')
    # Canonical JSON distinguishes boolean values from integer repeat counts.
    if json.dumps(manifest['context'], sort_keys=True, allow_nan=False) != json.dumps(expected_context, sort_keys=True):
        raise LUTError('compact reference source context does not match its trusted contract')
    if (json.dumps(manifest['policy_multipliers'], sort_keys=True, allow_nan=False)
            != json.dumps({'prefill': 1.0, 'decode': 1.0}, sort_keys=True)):
        raise LUTError('compact reference policy multipliers must remain unchanged')
    table = manifest['table']
    if (table['file'] != 'reference.json'
            or table['columns'] != ['length', 'r_p_ttft_s', 'r_d_tpot_s']
            or type(table['rows']) is not int or table['rows'] != len(PERFORMANCE_LENGTHS)
            or type(table['lengths']) is not list
            or any(type(n) is not int for n in table['lengths'])
            or table['lengths'] != list(PERFORMANCE_LENGTHS)):
        raise LUTError('invalid compact reference table contract')
    validation = manifest['validation']
    if (type(validation['canonical_requests_verified']) is not int
            or validation['canonical_requests_verified'] != expected_context['total_requests']
            or validation['raw_maxima_and_slo_values_preserved'] is not True
            or validation['controller_integration_validated'] is not False):
        raise LUTError('invalid compact reference source validation contract')
    if expected_context['technical_repeats_per_length'] in (5, 10):
        evidence = manifest['evidence_files']
        if type(evidence) is not dict or set(evidence) != {'source-validation.json'}:
            raise LUTError('compact reference requires source-validation.json evidence')
        if any(validation.get(key) is not True for key in (
                'canonical_hashes_checked', 'timestamp_metrics_checked', 'request_cpu_gpu_controls_checked')):
            raise LUTError('invalid compact reference canonical source validation contract')


def _bundle_file(root, name):
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise LUTError(f'compact reference file escapes the bundle or is a symlink: {name}')
    return path


class CompactReferenceRepository:
    """Read the eight frozen FP16 targets without importing performance cells.

    A known export and its measurement conditions are required. Source integrity
    does not establish controller integration or transfer an existing M4 grant.
    """

    def __init__(self, reference_path):
        root = Path(reference_path).resolve()
        manifest_data = _read(_bundle_file(root, 'manifest.json'))
        manifest_sha256 = hashlib.sha256(manifest_data).hexdigest()
        expected_context = _TRUSTED_REFERENCE_MANIFESTS.get(manifest_sha256)
        if expected_context is None:
            raise LUTError('unknown or modified compact reference manifest SHA-256')
        try:
            manifest = _json(manifest_data, 'manifest.json')
            _manifest_contract(manifest, expected_context)
            table = manifest['table']
            data = _read(_bundle_file(root, table['file']))
            if hashlib.sha256(data).hexdigest() != table['sha256']:
                raise LUTError('LUT SHA-256 mismatch: reference.json')
            if type(table['bytes']) is not int or len(data) != table['bytes']:
                raise LUTError('compact reference byte count mismatch')
            hashes = [('reference.json', table['sha256'])]
            if expected_context['technical_repeats_per_length'] in (5, 10):
                name = 'source-validation.json'
                spec = manifest['evidence_files'][name]
                evidence_data = _read(_bundle_file(root, name))
                if hashlib.sha256(evidence_data).hexdigest() != spec['sha256']:
                    raise LUTError(f'LUT SHA-256 mismatch: {name}')
                if type(spec['bytes']) is not int or len(evidence_data) != spec['bytes']:
                    raise LUTError(f'compact reference byte count mismatch: {name}')
                hashes.append((name, spec['sha256']))
            rows = _json(data, 'reference.json')
            if type(rows) is not list or len(rows) != len(PERFORMANCE_LENGTHS):
                raise LUTError('compact reference requires eight complete target rows')
            prefill, decode = {}, {}
            for row in rows:
                if (type(row) is not list or len(row) != 3 or type(row[0]) is not int
                        or row[0] not in PERFORMANCE_LENGTHS or row[0] in prefill):
                    raise LUTError('invalid or duplicate compact reference row')
                prefill[row[0]] = _number(row[1], 'Prefill target', positive=True)
                decode[row[0]] = _number(row[2], 'Decode target', positive=True)
            if set(prefill) != set(PERFORMANCE_LENGTHS):
                raise LUTError('incomplete compact reference length grid')
            self._prefill_target = MappingProxyType(prefill)
            self._decode_target = MappingProxyType(decode)
            repeats = {3: 'three', 5: 'five', 10: 'ten'}[expected_context['technical_repeats_per_length']]
            self.provenance = LUTProvenance(
                kind='measured', source_device='AGX2', source_commit=None,
                manifest_sha256=manifest_sha256,
                files_sha256=tuple(hashes),
                notes=(
                    f'AGX2 FP16 independent raw maxima over {repeats} requests per length; multipliers 1.0.',
                    'Source controller integration remains unvalidated; no automatic formal qualification.',
                ),
                metadata_json=json.dumps({'reference_manifest': manifest}, sort_keys=True, allow_nan=False),
            )
        except (KeyError, TypeError, IndexError, ValueError) as error:
            if isinstance(error, LUTError):
                raise
            raise LUTError(f'invalid compact reference structure: {error}') from error

    def preflight(self, length):
        validate_length(length)

    def prefill_target(self, length):
        bracket = _bracket(length, PERFORMANCE_LENGTHS)
        return _at(self._prefill_target[bracket.lower], self._prefill_target[bracket.upper], bracket)

    def decode_target(self, length):
        bracket = _bracket(length, PERFORMANCE_LENGTHS)
        return _at(self._decode_target[bracket.lower], self._decode_target[bracket.upper], bracket)


def load_reference_repository(path):
    """Accept only an explicit known sealed bundle or pinned compact reference."""
    root = Path(path)
    sealed = (root / 'bundle-manifest.json').exists()
    compact = any((root / name).exists() for name in ('manifest.json', 'reference.json'))
    if sealed and compact:
        raise LUTError('ambiguous reference directory contains sealed and compact formats')
    if compact:
        return CompactReferenceRepository(root)
    if sealed:
        return MeasuredLUTRepository(root)
    raise LUTError('unknown reference directory format: expected a sealed bundle or compact reference')
