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
from test_performance_cli import performance


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / 'data/luts/latest'
MANIFEST_PIN = '86b2343eed65432d1ee1f071fa8188a33789bdcf0309f325e689ec0bae37591a'
PINS = {
    'prefill.json': '26380e92c28fd1f570fe4e83f8829d24a80558cb3e7bb5a7c65792c97f81add5',
    'decode.json': '4deacacb51444f941d1b6263dbc667f4dc42cad4afa80c99d8a8745fe8566208',
    'quality.json': '7804c10ea01c386cc52fe9367568951715d47b1d017aa790d299837db569235c',
}


@pytest.fixture(scope='module')
def reference_path():
    configured = os.environ.get('KVOLT_LUT_BUNDLE')
    path = Path(configured) if configured else ROOT.parent / 'KVolt-lut-source/data/luts/agx2-repeat5-20261103'
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
    monkeypatch.setattr(module, 'COMPACT_MANIFEST_SHA256', hashlib.sha256(raw_manifest).hexdigest())


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
    monkeypatch.setattr(module, 'COMPACT_MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())
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


def test_run_development_compact_source_keeps_explicit_agx2_targets(tmp_path, reference_path, capsys):
    from kvolt_runtime.run_cli import _repository, main
    from test_run_cli import arguments, factories
    events = []
    seam, _ = factories(events)
    args = arguments(tmp_path, '--native-profile', 'agx2')
    index = args.index('--mock')
    args[index:index + 2] = ['--lut-dir', str(LATEST), '--reference-lut-dir', str(reference_path)]
    assert main(args, factories=replace(seam, repository=_repository)) == 0
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    session = next(r for r in reports if r['record_type'] == 'session_result')
    provenance = json.loads((Path(session['artifact_root']) / 'session.json').read_text())['provenance']
    assert provenance['lut']['source_device'] == 'agx1'
    assert provenance['lut']['metadata']['target_repository']['source_device'] == 'AGX2'
    assert provenance['lut']['metadata']['cross_device_reference'] is True
    assert provenance['lut_device_conditions']['device_match'] is False
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
