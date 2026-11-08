"""Independent checks for the tracked AGX2 target-only compact reference.

Semantic corruption tests explicitly patch only the test process's manifest pin
(and table hash where necessary). Production integrity tests never patch pins.
No optional sealed data or policy numerical dependencies are needed by defaults.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from kvolt_runtime.policy import reference as module
from kvolt_runtime.policy.lut import MeasuredLUTRepository
from kvolt_runtime.policy.reference import CompactReferenceRepository, load_reference_repository
from kvolt_runtime.policy.types import LUTError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/luts/agx2-reference-repeat3'
MANIFEST_PIN = '00d5112eaf9b0cdad2420b7622522c5d0913196e47681a286c99e19eeeeae15c'
TABLE_PIN = '0a1382b6a6a3b64626f4195902c3bdee325b58ff993714a4d4eb9d0693263b17'
EXPECTED = (
    (1024, 0.80568193, 0.10321148182102909),
    (2048, 1.974938814, 0.1079870083150685),
    (3072, 2.956504612, 0.11268618886497064),
    (4096, 3.894798365, 0.11743893422896282),
    (5120, 3.748924308, 0.12218092463405088),
    (6144, 4.624497683, 0.12700853362035225),
    (7168, 5.401990231, 0.1316925049373777),
    (8192, 6.313309433, 0.13651245762818004),
)


@pytest.fixture
def reference():
    return CompactReferenceRepository(SOURCE)


@pytest.fixture
def copied(tmp_path):
    return Path(shutil.copytree(SOURCE, tmp_path / 'reference'))


def semantic_reseal(monkeypatch, root, manifest, rows=None):
    """Bypass trusted hash only in semantic tests; never modify original files."""
    if rows is not None:
        data = json.dumps(rows).encode()
        (root / 'reference.json').write_bytes(data)
        manifest['table'].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    raw = json.dumps(manifest).encode()
    (root / 'manifest.json').write_bytes(raw)
    monkeypatch.setattr(module, 'COMPACT_REFERENCE_MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())


def test_original_hashes_bytes_and_all_eight_targets(reference):
    assert module.COMPACT_REFERENCE_MANIFEST_SHA256 == MANIFEST_PIN
    manifest = (SOURCE / 'manifest.json').read_bytes()
    data = (SOURCE / 'reference.json').read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == MANIFEST_PIN
    assert hashlib.sha256(data).hexdigest() == TABLE_PIN
    assert len(data) == 310
    assert json.loads(data) == [list(row) for row in EXPECTED]
    for length, prefill, decode in EXPECTED:
        assert reference.prefill_target(length) == prefill
        assert reference.decode_target(length) == decode
        assert reference.preflight(length) is None
    # The measured non-monotonic TTFT knot must not be smoothed away.
    assert reference.prefill_target(5120) < reference.prefill_target(4096)


@pytest.mark.parametrize('method,column', [('prefill_target', 1), ('decode_target', 2)])
def test_length_interpolation_each_adjacent_interval(reference, method, column):
    for lower, upper in zip(EXPECTED, EXPECTED[1:]):
        for offset in (1, 137, 512, 1023):
            weight = offset / 1024
            assert getattr(reference, method)(lower[0] + offset) == (
                (1 - weight) * lower[column] + weight * upper[column])


@pytest.mark.parametrize('method', ['preflight', 'prefill_target', 'decode_target'])
@pytest.mark.parametrize('bad', [True, False, None, '1024', 1024.0, 1023, 8193, -1])
def test_strict_length_types_and_no_extrapolation(reference, method, bad):
    with pytest.raises(ValueError):
        getattr(reference, method)(bad)


def test_provenance_preserves_entire_manifest_and_validation(reference):
    provenance = reference.provenance
    assert provenance.kind == 'measured'
    assert provenance.source_device == 'AGX2'
    assert provenance.source_commit is None
    assert provenance.manifest_sha256 == MANIFEST_PIN
    assert dict(provenance.files_sha256) == {'reference.json': TABLE_PIN}
    manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
    assert json.loads(provenance.metadata_json) == {'reference_manifest': manifest}
    assert manifest['context']['technical_repeats_per_length'] == 3
    assert manifest['context']['total_requests'] == 24
    assert manifest['policy_multipliers'] == {'prefill': 1.0, 'decode': 1.0}
    assert manifest['validation']['controller_integration_validated'] is False
    assert manifest['validation']['raw_maxima_and_slo_values_preserved'] is True
    assert 'independent raw maxima' in ' '.join(provenance.notes)


def test_target_maps_are_immutable(reference):
    for mapping in (reference._prefill_target, reference._decode_target):
        with pytest.raises(TypeError):
            mapping[1024] = 0
    assert reference.prefill_target(1024) == EXPECTED[0][1]


@pytest.mark.parametrize('name', ['manifest.json', 'reference.json'])
def test_modified_and_missing_original_file_rejected(copied, name):
    path = copied / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(LUTError, match='SHA-256'):
        CompactReferenceRepository(copied)
    path.unlink()
    with pytest.raises(LUTError, match='missing or unreadable'):
        CompactReferenceRepository(copied)


def test_rehashed_replacement_still_rejected_by_production_pin(copied):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    rows = json.loads((copied / 'reference.json').read_bytes())
    rows[0][1] *= 0.5
    raw = json.dumps(rows).encode()
    (copied / 'reference.json').write_bytes(raw)
    manifest['table'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (copied / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(LUTError, match='manifest SHA-256'):
        CompactReferenceRepository(copied)


@pytest.mark.parametrize('bad', [
    'missing', 'extra', 'duplicate', 'wrong_grid', 'float_length', 'bool_length',
    'string_length', 'short_row', 'extra_column', 'object_row', 'object_table',
    'bool_metric', 'str_metric', 'none_metric', 'zero', 'negative', 'nan', 'infinity',
])
@pytest.mark.parametrize('column', [1, 2])
def test_semantic_rows_grid_and_metrics(monkeypatch, copied, bad, column):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    rows = json.loads((copied / 'reference.json').read_bytes())
    if bad == 'missing': rows.pop()
    elif bad == 'extra': rows.append(rows[0].copy())
    elif bad == 'duplicate': rows[-1] = rows[0].copy()
    elif bad == 'wrong_grid': rows[0][0] = 1025
    elif bad == 'float_length': rows[0][0] = 1024.0
    elif bad == 'bool_length': rows[0][0] = True
    elif bad == 'string_length': rows[0][0] = '1024'
    elif bad == 'short_row': rows[0].pop()
    elif bad == 'extra_column': rows[0].append(1)
    elif bad == 'object_row': rows[0] = {'length': 1024}
    elif bad == 'object_table': rows = {'rows': rows}
    else:
        rows[0][column] = {'bool_metric': True, 'str_metric': '0.5', 'none_metric': None,
                           'zero': 0, 'negative': -1, 'nan': float('nan'),
                           'infinity': float('inf')}[bad]
    semantic_reseal(monkeypatch, copied, manifest, rows)
    with pytest.raises(LUTError):
        CompactReferenceRepository(copied)


CONTEXT_KEYS = tuple(json.loads((SOURCE / 'manifest.json').read_bytes())['context'])


@pytest.mark.parametrize('key', CONTEXT_KEYS)
def test_semantic_full_measurement_context_required(monkeypatch, copied, key):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    del manifest['context'][key]
    semantic_reseal(monkeypatch, copied, manifest)
    with pytest.raises(LUTError):
        CompactReferenceRepository(copied)


@pytest.mark.parametrize('bad', [
    'schema', 'units', 'axis', 'method', 'extrapolation', 'context_repeat_type',
    'context_generation_bool', 'multiplier_prefill', 'multiplier_decode', 'multiplier_type',
    'file', 'columns', 'rows', 'rows_type', 'lengths', 'length_type', 'bytes', 'bytes_type',
    'hash', 'verified', 'verified_type', 'preserved', 'integration', 'missing_context',
])
def test_semantic_manifest_contract(monkeypatch, copied, bad):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    if bad == 'schema': manifest['schema'] = 'unknown'
    elif bad == 'units': manifest['units']['r_p_ttft_s'] = 'ms'
    elif bad == 'axis': manifest['interpolation']['axis'] = 'frequency'
    elif bad == 'method': manifest['interpolation']['method'] = 'nearest'
    elif bad == 'extrapolation': manifest['interpolation']['extrapolation'] = True
    elif bad == 'context_repeat_type': manifest['context']['technical_repeats_per_length'] = 3.0
    elif bad == 'context_generation_bool': manifest['context']['generation']['force_length'] = 0
    elif bad.startswith('multiplier_'):
        manifest['policy_multipliers']['decode' if bad == 'multiplier_decode' else 'prefill'] = 1 if bad == 'multiplier_type' else 1.1
    elif bad == 'file': manifest['table']['file'] = '../reference.json'
    elif bad == 'columns': manifest['table']['columns'].reverse()
    elif bad == 'rows': manifest['table']['rows'] = 7
    elif bad == 'rows_type': manifest['table']['rows'] = 8.0
    elif bad == 'lengths': manifest['table']['lengths'].reverse()
    elif bad == 'length_type': manifest['table']['lengths'][0] = 1024.0
    elif bad == 'bytes': manifest['table']['bytes'] += 1
    elif bad == 'bytes_type': manifest['table']['bytes'] = 310.0
    elif bad == 'hash': manifest['table']['sha256'] = '0' * 64
    elif bad == 'verified': manifest['validation']['canonical_requests_verified'] = 23
    elif bad == 'verified_type': manifest['validation']['canonical_requests_verified'] = 24.0
    elif bad == 'preserved': manifest['validation']['raw_maxima_and_slo_values_preserved'] = 1
    elif bad == 'integration': manifest['validation']['controller_integration_validated'] = 0
    elif bad == 'missing_context': del manifest['context']
    semantic_reseal(monkeypatch, copied, manifest)
    with pytest.raises(LUTError):
        CompactReferenceRepository(copied)


def test_reference_file_cannot_escape_through_symlink(copied):
    (copied / 'reference.json').unlink()
    (copied / 'reference.json').symlink_to(SOURCE / 'reference.json')
    with pytest.raises(LUTError, match='escapes'):
        CompactReferenceRepository(copied)


def test_factory_recognized_unknown_incomplete_and_ambiguous(copied, tmp_path):
    assert type(load_reference_repository(SOURCE)) is CompactReferenceRepository
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(LUTError, match='unknown'):
        load_reference_repository(empty)
    (empty / 'reference.json').write_text('[]')
    with pytest.raises(LUTError, match='missing or unreadable'):
        load_reference_repository(empty)
    (copied / 'bundle-manifest.json').write_text('{}')
    with pytest.raises(LUTError, match='ambiguous'):
        load_reference_repository(copied)


def test_optional_old_sealed_targets_are_exact(reference):
    configured = os.environ.get('KVOLT_LUT_BUNDLE')
    if configured is not None and (not configured or not Path(configured).is_dir()):
        pytest.fail('KVOLT_LUT_BUNDLE must name an available sealed reference path')
    path = Path(configured) if configured is not None else ROOT.parent / 'KVolt-lut-source/data/luts/agx2-repeat5-20261103'
    if not path.is_dir():
        pytest.skip('optional sealed reference unavailable: set KVOLT_LUT_BUNDLE')
    sealed = load_reference_repository(path)
    assert type(sealed) is MeasuredLUTRepository
    for length, _, _ in EXPECTED:
        assert reference.prefill_target(length) == sealed.prefill_target(length)
        assert reference.decode_target(length) == sealed.decode_target(length)
