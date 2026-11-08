"""Pinned, target-only FP16 references supplied separately from predictions."""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from .lut import MeasuredLUTRepository, _at, _bracket, _json, _number, _read
from .types import LUTError, LUTProvenance, PERFORMANCE_LENGTHS, length as validate_length


COMPACT_REFERENCE_MANIFEST_SHA256 = '00d5112eaf9b0cdad2420b7622522c5d0913196e47681a286c99e19eeeeae15c'
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


def _manifest_contract(manifest):
    if (type(manifest) is not dict
            or manifest.get('schema') != 'kvolt.reference-lut.v1'
            or manifest.get('units') != {'length': 'tokens', 'r_p_ttft_s': 's', 'r_d_tpot_s': 's/token'}
            or manifest.get('interpolation') != {
                'axis': 'length', 'method': 'piecewise_linear', 'extrapolation': False}):
        raise LUTError('unsupported compact reference schema, units or interpolation')
    # Canonical JSON distinguishes boolean values from integer repeat counts.
    if json.dumps(manifest['context'], sort_keys=True, allow_nan=False) != json.dumps(_CONTEXT, sort_keys=True):
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
            or validation['canonical_requests_verified'] != 24
            or validation['raw_maxima_and_slo_values_preserved'] is not True
            or validation['controller_integration_validated'] is not False):
        raise LUTError('invalid compact reference source validation contract')


class CompactReferenceRepository:
    """Read the eight frozen FP16 targets without importing performance cells.

    A known export and its measurement conditions are required. Source integrity
    does not establish controller integration or transfer an existing M4 grant.
    """

    def __init__(self, reference_path):
        root = Path(reference_path).resolve()
        manifest_data = _read(root / 'manifest.json')
        if hashlib.sha256(manifest_data).hexdigest() != COMPACT_REFERENCE_MANIFEST_SHA256:
            raise LUTError('unknown or modified compact reference manifest SHA-256')
        try:
            manifest = _json(manifest_data, 'manifest.json')
            _manifest_contract(manifest)
            table = manifest['table']
            path = root / table['file']
            if path.resolve().parent != root:
                raise LUTError('compact reference file escapes the bundle')
            data = _read(path)
            if hashlib.sha256(data).hexdigest() != table['sha256']:
                raise LUTError('LUT SHA-256 mismatch: reference.json')
            if type(table['bytes']) is not int or len(data) != table['bytes']:
                raise LUTError('compact reference byte count mismatch')
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
            self.provenance = LUTProvenance(
                kind='measured', source_device='AGX2', source_commit=None,
                manifest_sha256=COMPACT_REFERENCE_MANIFEST_SHA256,
                files_sha256=(('reference.json', table['sha256']),),
                notes=(
                    'AGX2 FP16 independent raw maxima over three requests per length; multipliers 1.0.',
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
