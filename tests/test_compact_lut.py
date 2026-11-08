"""Latest AGX1 compact predictions with explicitly separate sealed AGX2 targets."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from kvolt_runtime.policy.lut import MeasuredLUTRepository, MANIFEST_SHA256
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.types import LUTError, PERFORMANCE_LENGTHS as P, QUALITY_LENGTHS as Q
from kvolt_runtime.policy.compact import CompactLUTRepository, load_lut_repository
from test_performance_cli import performance, write_report


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / 'data/luts/latest'
AGX2 = ROOT / 'data/luts/agx2-runtime-repeat5'
NEW_REFERENCE = ROOT / 'data/luts/agx2-reference-repeat3'
AGX2_MANIFEST_PIN = 'b421fcd10ecb57df7d6962aa9a5f48e264e90defc0c76a3544218d31685f783b'
MANIFEST_PIN = '86b2343eed65432d1ee1f071fa8188a33789bdcf0309f325e689ec0bae37591a'
PINS = {
    'prefill.json': '26380e92c28fd1f570fe4e83f8829d24a80558cb3e7bb5a7c65792c97f81add5',
    'decode.json': '4deacacb51444f941d1b6263dbc667f4dc42cad4afa80c99d8a8745fe8566208',
    'quality.json': '7804c10ea01c386cc52fe9367568951715d47b1d017aa790d299837db569235c',
}


@pytest.fixture(scope='module')
def reference_path():
    configured = os.environ.get('KVOLT_LUT_BUNDLE')
    if configured is not None and (not configured or not Path(configured).is_dir()):
        pytest.fail('KVOLT_LUT_BUNDLE must name an available sealed reference path')
    path = Path(configured) if configured is not None else ROOT.parent / 'KVolt-lut-source/data/luts/agx2-repeat5-20261103'
    if not path.is_dir():
        pytest.skip('sealed measured reference unavailable: set KVOLT_LUT_BUNDLE')
    return path


@pytest.fixture(scope='module')
def reference(reference_path):
    return MeasuredLUTRepository(reference_path)


@pytest.fixture(scope='module')
def compact(reference):
    return CompactLUTRepository(LATEST, reference_repository=reference)


def test_original_manifest_and_raw_bytes_are_pinned():
    # The recovered source export is not distributed in fresh checkouts.
    # Its independently recorded byte count/hash pin the tracked copy exactly.
    raw_manifest = (LATEST / 'manifest.json').read_bytes()
    assert len(raw_manifest) == 4809
    assert hashlib.sha256(raw_manifest).hexdigest() == MANIFEST_PIN
    for name, pin in PINS.items():
        assert hashlib.sha256((LATEST / name).read_bytes()).hexdigest() == pin


def test_all_459_original_cells_are_exact_and_targets_are_separate(compact, reference):
    counts = []
    for name, query, width in [('prefill', compact.prefill, 2), ('decode', compact.decode, 3), ('quality', compact.quality, 2)]:
        rows = json.loads((LATEST / (name + '.json')).read_bytes())
        counts.append(len(rows))
        for row in rows:
            candidates = query(row[0])
            found = next(p for p in candidates if
                         ([p.frequency_mhz] if name == 'prefill' else
                          [p.group, p.frequency_mhz] if name == 'decode' else [p.group]) == row[1:width])
            actual = [found.ttft_s, found.energy_j] if name == 'prefill' else [found.tpot_s, found.energy_j_per_token] if name == 'decode' else [found.score]
            assert actual == row[width:]
            assert found.bracket.lower == found.bracket.upper == row[0]
    assert counts == [64, 320, 75]
    for n in P:
        assert compact.prefill_target(n) == reference.prefill_target(n)
        assert compact.decode_target(n) == reference.decode_target(n)
    # AGX1 performance measurements must never replace frozen FP16 targets.
    assert compact.decode_target(1024) != min(p.tpot_s for p in compact.decode(1024))


@pytest.mark.parametrize('query,grid,attrs', [
    ('prefill', P, ('ttft_s', 'energy_j')), ('decode', P, ('tpot_s', 'energy_j_per_token')),
    ('quality', Q, ('score',)),
])
def test_length_only_interpolation(compact, query, grid, attrs):
    for lo, hi in zip(grid, grid[1:]):
        n = lo + 137
        w = 137 / (hi - lo)
        for a, b, value in zip(getattr(compact, query)(lo), getattr(compact, query)(hi), getattr(compact, query)(n)):
            for attr in attrs:
                assert getattr(value, attr) == (1 - w) * getattr(a, attr) + w * getattr(b, attr)
            assert value.bracket.weight == w


@pytest.mark.parametrize('method', ['preflight', 'prefill', 'decode', 'quality', 'prefill_target', 'decode_target'])
@pytest.mark.parametrize('bad', [True, None, '1024', 1024.0, 1023, 8193])
def test_no_coercion_or_extrapolation(compact, method, bad):
    with pytest.raises(ValueError):
        getattr(compact, method)(bad)


def test_cross_device_target_provenance_preserves_source_flags(compact, reference):
    p = compact.provenance
    assert p.kind == 'measured' and p.source_device == 'agx1' and p.source_commit is None
    assert p.manifest_sha256 == MANIFEST_PIN and dict(p.files_sha256) == PINS
    metadata = json.loads(p.metadata_json)
    manifest = json.loads((LATEST / 'manifest.json').read_bytes())
    assert metadata['compact_manifest'] == manifest
    assert metadata['compact_manifest']['validation']['performance_runtime_accuracy_or_SLO_validated'] is False
    assert metadata['compact_manifest']['validation']['quality_runtime_eligible'] is False
    assert metadata['cross_device_reference'] is True
    assert metadata['target_source_device'] == 'AGX2'
    target = metadata['target_repository']
    assert target['manifest_sha256'] == MANIFEST_SHA256
    assert target['source_device'] == reference.provenance.source_device
    assert target['files_sha256']['fp16-reference-curves.json'] == dict(reference.provenance.files_sha256)['fp16-reference-curves.json']
    assert target['reference']['device'] == 'A2'
    assert target['reference']['policy_multiplier_d'] == target['reference']['policy_multiplier_p'] == 1.0
    assert 'requests' not in target['reference'] and 'knots' not in target['reference']
    assert len(p.metadata_json) < 20_000


def test_compact_repository_cannot_supply_its_own_measured_targets(compact):
    with pytest.raises(LUTError, match='sealed.*MeasuredLUTRepository'):
        CompactLUTRepository(LATEST, reference_repository=compact)


@pytest.mark.parametrize('source', [None, MockLUTRepository(), SimpleNamespace(provenance=SimpleNamespace(kind='measured'))])
def test_missing_synthetic_or_duck_reference_rejected(source):
    with pytest.raises(LUTError, match='sealed.*MeasuredLUTRepository'):
        CompactLUTRepository(LATEST, reference_repository=source)


@pytest.mark.parametrize('name', ['manifest.json', *PINS])
def test_every_compact_file_changed_or_missing_rejected(tmp_path, reference, name):
    root = tmp_path / 'latest'
    shutil.copytree(LATEST, root)
    path = root / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(LUTError, match='SHA-256'):
        CompactLUTRepository(root, reference_repository=reference)
    path.unlink()
    with pytest.raises(LUTError, match='missing or unreadable'):
        CompactLUTRepository(root, reference_repository=reference)


def reseal_for_semantic_test(monkeypatch, root, name, data):
    """Test structure independently of hash protection; production pin stays fixed."""
    from kvolt_runtime.policy import compact as module
    raw = json.dumps(data).encode()
    (root / name).write_bytes(raw)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    spec = manifest['tables'][name.removesuffix('.json')]
    spec.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    raw_manifest = json.dumps(manifest).encode()
    (root / 'manifest.json').write_bytes(raw_manifest)
    monkeypatch.setitem(module._TRUSTED_COMPACT_MANIFESTS, hashlib.sha256(raw_manifest).hexdigest(),
                        module._TRUSTED_COMPACT_MANIFESTS[MANIFEST_PIN])


def test_rehashed_modified_data_and_manifest_cannot_replace_original_pin(tmp_path, reference):
    root = tmp_path / 'latest'
    shutil.copytree(LATEST, root)
    rows = json.loads((root / 'decode.json').read_bytes())
    rows[0][-2] *= 0.5
    raw = json.dumps(rows).encode()
    (root / 'decode.json').write_bytes(raw)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    manifest['tables']['decode'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(LUTError, match='manifest SHA-256'):
        CompactLUTRepository(root, reference_repository=reference)


@pytest.mark.parametrize('bad', ['schema', 'units', 'interpolation', 'axis', 'float_axis', 'columns', 'rows', 'source_device'])
def test_manifest_contract_structure(monkeypatch, tmp_path, reference, bad):
    from kvolt_runtime.policy import compact as module
    root = tmp_path / 'latest'
    shutil.copytree(LATEST, root)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    if bad == 'schema': manifest['schema'] = 'unknown'
    elif bad == 'units': manifest['units']['tpot_actual_s'] = 'ms/token'
    elif bad == 'interpolation': manifest['interpolation']['axis'] = 'gpu_mhz'
    elif bad == 'axis': manifest['tables']['decode']['axes']['group'][0] = 7
    elif bad == 'float_axis': manifest['tables']['prefill']['axes']['length'][0] = 1024.0
    elif bad == 'columns': manifest['tables']['prefill']['columns'].reverse()
    elif bad == 'rows': manifest['tables']['quality']['rows'] = 74
    elif bad == 'source_device': manifest['context']['performance_device_id'] = 'A2'
    raw = json.dumps(manifest).encode()
    (root / 'manifest.json').write_bytes(raw)
    monkeypatch.setitem(module._TRUSTED_COMPACT_MANIFESTS, hashlib.sha256(raw).hexdigest(),
                        module._TRUSTED_COMPACT_MANIFESTS[MANIFEST_PIN])
    with pytest.raises(LUTError):
        CompactLUTRepository(root, reference_repository=reference)


@pytest.mark.parametrize('name', ['prefill.json', 'decode.json', 'quality.json'])
@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'extra', 'bool_id', 'float_id', 'str_id', 'bad_grid', 'short_row', 'object_row', 'bool_metric', 'str_metric', 'nan', 'negative', 'zero_or_high'])
def test_full_grids_numeric_rows_and_ranges(monkeypatch, tmp_path, reference, name, bad):
    root = tmp_path / 'latest'
    shutil.copytree(LATEST, root)
    rows = json.loads((root / name).read_bytes())
    if bad == 'missing': rows.pop()
    elif bad == 'duplicate': rows[-1] = rows[0].copy()
    elif bad == 'extra': rows.append(rows[0].copy())
    elif bad == 'bool_id': rows[0][0] = True
    elif bad == 'float_id': rows[0][0] = float(rows[0][0])
    elif bad == 'str_id': rows[0][0] = str(rows[0][0])
    elif bad == 'bad_grid': rows[0][0] = 1025
    elif bad == 'short_row': rows[0].pop()
    elif bad == 'object_row': rows[0] = {'length': 1024}
    elif bad == 'bool_metric': rows[0][-1] = True
    elif bad == 'str_metric': rows[0][-1] = '0.5'
    elif bad == 'nan': rows[0][-1] = float('nan')
    elif bad == 'negative': rows[0][-1] = -1
    elif bad == 'zero_or_high': rows[0][-1 if name == 'quality.json' else -2] = 1.1 if name == 'quality.json' else 0
    reseal_for_semantic_test(monkeypatch, root, name, rows)
    with pytest.raises(LUTError):
        CompactLUTRepository(root, reference_repository=reference)


def test_factory_requires_explicit_reference_and_recognized_unambiguous_format(tmp_path, reference_path):
    with pytest.raises(LUTError, match='reference-lut-dir'):
        load_lut_repository(LATEST)
    assert type(load_lut_repository(reference_path)) is MeasuredLUTRepository
    assert type(load_lut_repository(LATEST, reference_lut_dir=reference_path)) is CompactLUTRepository
    with pytest.raises(LUTError, match='unknown'):
        load_lut_repository(tmp_path)
    shutil.copytree(LATEST, tmp_path / 'both')
    (tmp_path / 'both/bundle-manifest.json').write_text('{}')
    with pytest.raises(LUTError, match='ambiguous'):
        load_lut_repository(tmp_path / 'both', reference_lut_dir=reference_path)
    with pytest.raises(LUTError, match='compact'):
        load_lut_repository(reference_path, reference_lut_dir=reference_path)


def test_cli_repository_defaults_keep_old_namespace_compatible(reference_path):
    from kvolt_runtime.cli import _build_repository
    from kvolt_runtime.run_cli import _repository
    assert type(_repository(SimpleNamespace(lut_dir=reference_path, mock=None))) is MeasuredLUTRepository
    assert type(_build_repository(SimpleNamespace(lut_bundle=reference_path, mock=None))) is MeasuredLUTRepository
    for factory, key in [(_repository, 'lut_dir'), (_build_repository, 'lut_bundle')]:
        assert type(factory(SimpleNamespace(**{key: None, 'mock': 'baseline'}))) is MockLUTRepository
        with pytest.raises(ValueError, match='--reference-lut-dir requires'):
            factory(SimpleNamespace(**{key: None, 'mock': 'baseline', 'reference_lut_dir': reference_path}))


def test_diagnose_cli_explicit_reference_and_skip_preserve_provenance(reference_path, capsys):
    from kvolt_runtime.cli import main
    args = ['diagnose', '--length', '1024', '--max-new-tokens', '1', '--pc-elapsed-ns', '0', '--lut-bundle', str(LATEST)]
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out)['error']['stage'] == 'lut'
    assert main([*args, '--reference-lut-dir', str(reference_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['source']['source_device'] == 'agx1'
    assert report['source']['metadata']['target_source_device'] == 'AGX2'
    assert report['decode']['status'] == 'skipped'


@pytest.mark.parametrize('bundle,source_device,mismatch', [(LATEST, 'agx1', True), (AGX2, 'agx2', False)])
def test_run_development_compact_source_keeps_explicit_agx2_targets(tmp_path, reference_path, capsys, bundle, source_device, mismatch):
    from kvolt_runtime.run_cli import _repository, main
    from test_run_cli import arguments, factories
    events = []
    seam, _ = factories(events)
    args = arguments(tmp_path, '--native-profile', 'agx2')
    index = args.index('--mock')
    args[index:index + 2] = ['--lut-dir', str(bundle), '--reference-lut-dir', str(reference_path)]
    assert main(args, factories=replace(seam, repository=_repository)) == 0
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    session = next(r for r in reports if r['record_type'] == 'session_result')
    provenance = json.loads((Path(session['artifact_root']) / 'session.json').read_text())['provenance']
    assert provenance['lut']['source_device'] == source_device
    assert provenance['lut']['metadata']['target_repository']['source_device'] == 'AGX2'
    assert provenance['lut']['metadata']['cross_device_reference'] is mismatch
    assert provenance['lut_device_conditions']['device_match'] is (not mismatch)
    assert provenance['formal_performance_qualified'] is False


def test_run_cli_new_source_records_device_mismatch_and_rejects_old_grant_before_gpu(performance, reference_path, capsys):
    from kvolt_runtime.run_cli import _repository, main
    args = list(performance.args)
    args[args.index('--lut-dir') + 1] = str(LATEST)
    args += ['--reference-lut-dir', str(reference_path)]
    seam = replace(performance.seam, repository=_repository)
    assert main(args, factories=seam) == 2
    report, = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert report['errors'][0]['stage'] == 'qualification'
    assert 'AGX2 LUTs' in report['errors'][0]['message']
    assert not {'lock', 'backend', 'runner'} & set(performance.events)
    saved = json.loads((Path(report['artifact_root']) / 'session.json').read_text())
    assert saved['provenance']['lut_device_conditions']['development_device_mismatch'] is True
    assert saved['provenance']['formal_performance_qualified'] is False


def test_run_mock_reference_conflict_rejected_even_with_injected_repository(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    from test_run_cli import arguments, factories
    events = []
    seam, _ = factories(events)
    assert main(arguments(tmp_path, '--reference-lut-dir', str(tmp_path / 'refs')), factories=seam) == 2
    report = json.loads(capsys.readouterr().out)
    assert '--reference-lut-dir requires' in report['errors'][0]['message']
    assert events == []


@pytest.mark.parametrize('reference_kind', ['compact', 'sealed'])
def test_agx2_runtime_bundle_preserves_all_cells_context_and_targets(request, reference_kind):
    from kvolt_runtime.policy.reference import CompactReferenceRepository
    reference = (CompactReferenceRepository(NEW_REFERENCE) if reference_kind == 'compact' else
                 request.getfixturevalue('reference'))
    repo = CompactLUTRepository(AGX2, reference_repository=reference)
    manifest = json.loads((AGX2 / 'manifest.json').read_bytes())
    assert repo.provenance.source_device == 'agx2'
    assert repo.provenance.manifest_sha256 == AGX2_MANIFEST_PIN
    metadata = json.loads(repo.provenance.metadata_json)
    assert metadata['compact_manifest'] == manifest
    assert metadata['cross_device_reference'] is False
    assert manifest['context']['prefill_aggregation'] == {
        'method': 'arithmetic_mean_over_all_group_request_observations',
        'groups': [8, 16, 32, 64, 128], 'repeats_per_group': 5,
        'samples_per_cell': 25, 'total_requests': 1600, 'documents_per_length': 1}
    assert manifest['context']['decode_aggregation'] == {
        'method': 'source_cell_means_preserved', 'repeats_per_cell': 5}
    assert manifest['context']['quality_source_device_id'] == 'A1'
    assert manifest['context']['quality_items_per_length'] == 100
    assert manifest['validation']['performance_runtime_accuracy_or_SLO_validated'] is False
    assert manifest['validation']['quality_runtime_eligible'] is False
    counts = []
    for name in ('prefill', 'decode', 'quality'):
        rows = json.loads((AGX2 / (name + '.json')).read_bytes())
        counts.append(len(rows))
        for row in rows:
            values = getattr(repo, name)(row[0])
            key = row[1:2] if name != 'decode' else row[1:3]
            found = next(p for p in values if
                         ([p.frequency_mhz] if name == 'prefill' else
                          [p.group, p.frequency_mhz] if name == 'decode' else [p.group]) == key)
            actual = ([found.ttft_s, found.energy_j] if name == 'prefill' else
                      [found.tpot_s, found.energy_j_per_token] if name == 'decode' else [found.score])
            assert actual == row[len(row) - len(actual):]
    assert counts == [64, 320, 75]
    for n in (*P, 1161, 4097, 8191):
        assert repo.prefill_target(n) == reference.prefill_target(n)
        assert repo.decode_target(n) == reference.decode_target(n)


@pytest.mark.parametrize('name', ['manifest.json', 'prefill.json', 'decode.json', 'quality.json'])
def test_agx2_modified_or_missing_file_rejected(tmp_path, name):
    from kvolt_runtime.policy.reference import CompactReferenceRepository
    reference = CompactReferenceRepository(NEW_REFERENCE)
    root = tmp_path / 'a2'
    shutil.copytree(AGX2, root)
    path = root / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(LUTError, match='SHA-256'):
        CompactLUTRepository(root, reference_repository=reference)
    path.unlink()
    with pytest.raises(LUTError, match='missing or unreadable'):
        CompactLUTRepository(root, reference_repository=reference)


def test_agx2_matching_device_does_not_inherit_existing_formal_grant(performance, reference, reference_path, capsys):
    from kvolt_runtime.run_cli import _repository, main
    args = list(performance.args)
    args[args.index('--lut-dir') + 1] = str(AGX2)
    args += ['--reference-lut-dir', str(reference_path)]
    from kvolt_runtime.qualification.contracts import qualification_binding
    binding = qualification_binding(
        repo=reference, device_profile=performance.device_profile,
        power_profile=performance.power_profile, native_profile='agx2', runtime=performance.runtime)
    report_path = write_report(performance.report.parent.parent / 'sealed-evidence', binding, {'1024': 4})
    args[args.index('--qualification') + 1] = str(report_path)
    seam = replace(performance.seam, repository=_repository)
    assert main(args, factories=seam) == 2
    report, = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert report['errors'][0]['stage'] == 'qualification'
    assert 'binding mismatch' in report['errors'][0]['message']
    assert not {'lock', 'backend', 'runner'} & set(performance.events)
    saved = json.loads((Path(report['artifact_root']) / 'session.json').read_text())
    assert saved['provenance']['lut_device_conditions']['device_match'] is True
    assert saved['provenance']['formal_performance_qualified'] is False


def test_agx2_default_lookup_original_bytes_are_pinned():
    pins = {'manifest.json': AGX2_MANIFEST_PIN,
            'prefill.json': 'bb9e800faf56b3d0d5e4d20081dd62c4638642908fbf121199d83c40ae445a47',
            'decode.json': '875c78e7af420b52719b3a29d4c43332a9e53581be1681fa1f5b79e8cda3bb4b',
            'quality.json': PINS['quality.json']}
    for name, pin in pins.items():
        assert hashlib.sha256((AGX2 / name).read_bytes()).hexdigest() == pin


@pytest.mark.parametrize('field,value', [
    ('performance_device_id', 'A1'), ('performance_repeats_per_group', 3),
    ('model_revision', 'wrong'), ('quality_source_device_id', 'A2'),
    ('decode_measurement_boundary', 'Decode-only without packing'),
    ('prefill_aggregation.samples_per_cell', 5), ('prefill_aggregation.repeats_per_group', 3),
    ('prefill_aggregation.groups', [8]), ('prefill_aggregation.total_requests', 320),
    ('prefill_aggregation.documents_per_length', True),
    ('prefill_aggregation.method', 'G8_only'), ('decode_aggregation.repeats_per_cell', 3),
    ('decode_aggregation.method', 'recomputed'), ('quality_items_per_length', 40),
])
def test_agx2_context_contract_rejects_semantic_drift(monkeypatch, tmp_path, field, value):
    from kvolt_runtime.policy.reference import CompactReferenceRepository
    reference = CompactReferenceRepository(NEW_REFERENCE)
    from kvolt_runtime.policy import compact as module
    root = tmp_path / 'a2'
    shutil.copytree(AGX2, root)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    context = manifest['context']
    parts = field.split('.')
    for part in parts[:-1]:
        context = context[part]
    context[parts[-1]] = value
    raw = json.dumps(manifest).encode()
    (root / 'manifest.json').write_bytes(raw)
    # Explicit test trust injection bypasses only hashing, preserving the exact
    # production A2 context contract so metadata regressions are exercised.
    monkeypatch.setitem(module._TRUSTED_COMPACT_MANIFESTS, hashlib.sha256(raw).hexdigest(),
                        module._TRUSTED_COMPACT_MANIFESTS[AGX2_MANIFEST_PIN])
    with pytest.raises(LUTError, match='source context'):
        CompactLUTRepository(root, reference_repository=reference)


def test_measured_sealed_contract_does_not_accept_new_compact_bundle():
    with pytest.raises(LUTError):
        MeasuredLUTRepository(AGX2)


def test_new_compact_reference_metadata_keeps_complete_original_manifest():
    repo = load_lut_repository(AGX2, reference_lut_dir=NEW_REFERENCE)
    target = json.loads(repo.provenance.metadata_json)['target_repository']
    manifest = json.loads((NEW_REFERENCE / 'manifest.json').read_bytes())
    assert target['reference'] == target['compact_reference_manifest'] == manifest
    assert target['source_conditions'] == manifest['context']
    assert target['manifest_sha256'] == '00d5112eaf9b0cdad2420b7622522c5d0913196e47681a286c99e19eeeeae15c'
    assert target['source_commit'] is None
    assert target['reference']['validation']['controller_integration_validated'] is False
    assert dict(target['files_sha256']) == {'reference.json': '0a1382b6a6a3b64626f4195902c3bdee325b58ff993714a4d4eb9d0693263b17'}


@pytest.mark.parametrize('bundle,device', [(LATEST, 'agx1'), (AGX2, 'agx2')])
def test_both_prediction_exports_load_against_new_explicit_reference(bundle, device):
    from kvolt_runtime.policy.reference import CompactReferenceRepository
    repo = load_lut_repository(bundle, reference_lut_dir=NEW_REFERENCE)
    target = CompactReferenceRepository(NEW_REFERENCE)
    assert repo.provenance.source_device == device
    assert json.loads(repo.provenance.metadata_json)['cross_device_reference'] is (device != 'agx2')
    for n in (*P, 1161, 4097, 8191):
        assert repo.prefill_target(n) == target.prefill_target(n)
        assert repo.decode_target(n) == target.decode_target(n)


@pytest.mark.parametrize('field,value', [('manifest_sha256', '0' * 64), ('kind', 'synthetic'), ('source_device', 'AGX1')])
def test_new_reference_provenance_mismatch_rejected(field, value):
    from kvolt_runtime.policy.reference import CompactReferenceRepository
    reference = CompactReferenceRepository(NEW_REFERENCE)
    reference.provenance = replace(reference.provenance, **{field: value})
    with pytest.raises(LUTError, match='explicit reference_repository'):
        CompactLUTRepository(AGX2, reference_repository=reference)


def test_new_reference_subclass_and_duck_type_rejected():
    from kvolt_runtime.policy.reference import CompactReferenceRepository
    class Subclass(CompactReferenceRepository):
        pass
    for reference in (Subclass(NEW_REFERENCE), SimpleNamespace(provenance=CompactReferenceRepository(NEW_REFERENCE).provenance)):
        with pytest.raises(LUTError, match='explicit reference_repository'):
            CompactLUTRepository(AGX2, reference_repository=reference)


def test_new_reference_diagnose_and_development_cli(tmp_path, capsys):
    from kvolt_runtime.cli import main as diagnose
    from kvolt_runtime.run_cli import _repository, main
    from test_run_cli import arguments, factories
    assert diagnose(['diagnose', '--length', '1024', '--max-new-tokens', '1', '--pc-elapsed-ns', '0',
                     '--lut-bundle', str(AGX2), '--reference-lut-dir', str(NEW_REFERENCE)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['source']['metadata']['target_repository']['compact_reference_manifest']['context']['technical_repeats_per_length'] == 3
    events = []
    seam, _ = factories(events)
    args = arguments(tmp_path, '--native-profile', 'agx2')
    index = args.index('--mock')
    args[index:index + 2] = ['--lut-dir', str(AGX2), '--reference-lut-dir', str(NEW_REFERENCE)]
    assert main(args, factories=replace(seam, repository=_repository)) == 0
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    session = next(r for r in reports if r['record_type'] == 'session_result')
    provenance = json.loads((Path(session['artifact_root']) / 'session.json').read_text())['provenance']
    assert provenance['lut']['metadata']['target_repository']['manifest_sha256'] == '00d5112eaf9b0cdad2420b7622522c5d0913196e47681a286c99e19eeeeae15c'
    assert provenance['lut_device_conditions']['device_match'] is True
    assert provenance['formal_performance_qualified'] is False


@pytest.mark.parametrize('old_compact_reference', [False, True])
def test_new_reference_does_not_transfer_old_reference_grant(performance, reference_path, capsys, old_compact_reference):
    from kvolt_runtime.run_cli import _repository, main
    from kvolt_runtime.qualification.contracts import qualification_binding
    old_reference, new_reference = ((NEW_REFERENCE, reference_path) if old_compact_reference else
                                    (reference_path, NEW_REFERENCE))
    old_repo = load_lut_repository(AGX2, reference_lut_dir=old_reference)
    binding = qualification_binding(repo=old_repo, device_profile=performance.device_profile,
                                    power_profile=performance.power_profile, native_profile='agx2', runtime=performance.runtime)
    path = write_report(performance.report.parent.parent / 'old-compact-evidence', binding, {'1024': 4})
    args = list(performance.args)
    args[args.index('--lut-dir') + 1] = str(AGX2)
    args[args.index('--qualification') + 1] = str(path)
    args += ['--reference-lut-dir', str(new_reference)]
    assert main(args, factories=replace(performance.seam, repository=_repository)) == 2
    report, = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert report['errors'][0]['stage'] == 'qualification'
    assert 'binding mismatch' in report['errors'][0]['message']
    assert not {'lock', 'backend', 'runner'} & set(performance.events)


def test_compact_qualification_identity_uses_both_pins_and_ignores_local_paths(tmp_path):
    from kvolt_runtime.qualification.contracts import _lut_binding
    first = load_lut_repository(AGX2, reference_lut_dir=NEW_REFERENCE)
    runtime = tmp_path / 'runtime'
    target = tmp_path / 'target'
    shutil.copytree(AGX2, runtime)
    shutil.copytree(NEW_REFERENCE, target)
    second = load_lut_repository(runtime, reference_lut_dir=target)
    binding = _lut_binding(first.provenance)
    assert binding == _lut_binding(second.provenance)
    assert binding['manifest_sha256'] == AGX2_MANIFEST_PIN
    assert binding['reference']['manifest_sha256'] == '00d5112eaf9b0cdad2420b7622522c5d0913196e47681a286c99e19eeeeae15c'
    assert binding['reference']['files_sha256'] == {'reference.json': '0a1382b6a6a3b64626f4195902c3bdee325b58ff993714a4d4eb9d0693263b17'}


def test_sealed_qualification_identity_is_unchanged(reference):
    from kvolt_runtime.qualification.contracts import _lut_binding
    p = reference.provenance
    assert _lut_binding(p) == {'kind': p.kind, 'source_device': p.source_device,
                               'source_commit': p.source_commit, 'manifest_sha256': p.manifest_sha256,
                               'files_sha256': dict(p.files_sha256)}


@pytest.mark.parametrize('corruption', [
    'missing', 'null', 'list', 'field_missing', 'kind', 'device', 'commit',
    'manifest', 'files_list', 'files_empty', 'files_bad_hash', 'files_bad_name',
])
def test_qualification_rejects_missing_or_malformed_compact_target(corruption):
    from kvolt_runtime.qualification.contracts import _lut_binding
    repo = load_lut_repository(AGX2, reference_lut_dir=NEW_REFERENCE)
    metadata = json.loads(repo.provenance.metadata_json)
    target = metadata['target_repository']
    if corruption == 'missing': del metadata['target_repository']
    elif corruption == 'null': metadata['target_repository'] = None
    elif corruption == 'list': metadata['target_repository'] = []
    elif corruption == 'field_missing': del target['source_commit']
    elif corruption == 'kind': target['kind'] = 'synthetic'
    elif corruption == 'device': target['source_device'] = True
    elif corruption == 'commit': target['source_commit'] = []
    elif corruption == 'manifest': target['manifest_sha256'] = '0' * 63
    elif corruption == 'files_list': target['files_sha256'] = []
    elif corruption == 'files_empty': target['files_sha256'] = {}
    elif corruption == 'files_bad_hash': target['files_sha256'] = {'reference.json': True}
    elif corruption == 'files_bad_name': target['files_sha256'] = {'': '0' * 64}
    provenance = replace(repo.provenance, metadata_json=json.dumps(metadata))
    with pytest.raises(ValueError, match='target identity'):
        _lut_binding(provenance)
