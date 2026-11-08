"""Repeat-five targets bind the tracked canonical validation evidence."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from kvolt_runtime.policy import reference as module
from kvolt_runtime.policy.compact import load_lut_repository
from kvolt_runtime.policy.reference import CompactReferenceRepository
from kvolt_runtime.policy.types import LUTError
from test_compact_reference import (
    semantic_reseal, test_semantic_rows_grid_and_metrics,
    test_semantic_full_measurement_context_required, test_semantic_manifest_contract,
    test_strict_length_types_and_no_extrapolation,
)
from test_performance_cli import performance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/luts/agx2-reference-repeat5'
OLD = ROOT / 'data/luts/agx2-reference-repeat3'
RUNTIME = ROOT / 'data/luts/agx2-runtime-repeat5'
MANIFEST_PIN = 'd61c1b6a6ccd0aaa80b0cf4645702eaee09bd406dac084df70f151669a9b74ee'
TABLE_PIN = '0892b309e9124553f0b5a1410241060648b68f5be7c9c02db7e1d3b0fd640151'
EVIDENCE_PIN = 'a51bcff64f8e291d4121b0578c23ef211bad5c3e772f396ca31098d55400aa93'


@pytest.fixture
def reference():
    return CompactReferenceRepository(SOURCE)


@pytest.fixture
def copied(tmp_path):
    return Path(shutil.copytree(SOURCE, tmp_path / 'reference'))


def test_repeat5_known_pin_all_targets_and_complete_provenance(reference):
    assert module.COMPACT_REFERENCE_REPEAT5_MANIFEST_SHA256 == MANIFEST_PIN
    manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
    assert hashlib.sha256((SOURCE / 'manifest.json').read_bytes()).hexdigest() == MANIFEST_PIN
    assert reference.provenance.manifest_sha256 == MANIFEST_PIN
    assert dict(reference.provenance.files_sha256) == {
        'reference.json': TABLE_PIN, 'source-validation.json': EVIDENCE_PIN}
    assert json.loads(reference.provenance.metadata_json) == {'reference_manifest': manifest}
    assert 'locally generated' in manifest['provenance']['manifest_origin']
    assert manifest['context']['technical_repeats_per_length'] == 5
    assert manifest['context']['total_requests'] == manifest['validation']['canonical_requests_verified'] == 40
    assert manifest['validation']['controller_integration_validated'] is False
    rows = json.loads((SOURCE / 'reference.json').read_bytes())
    for length, prefill, decode in rows:
        assert reference.prefill_target(length) == prefill
        assert reference.decode_target(length) == decode
    for lower, upper in zip(rows, rows[1:]):
        for offset in (1, 137, 512, 1023):
            weight = offset / 1024
            for method, column in [('prefill_target', 1), ('decode_target', 2)]:
                assert getattr(reference, method)(lower[0] + offset) == (
                    (1 - weight) * lower[column] + weight * upper[column])


def test_tracked_evidence_recomputes_40_timestamps_and_independent_maxima(reference):
    evidence = json.loads((SOURCE / 'source-validation.json').read_bytes())
    records = evidence['records']
    assert len(records) == 40
    assert len({row['id'] for row in records}) == 40
    for row in records:
        assert row['controls_verified'] is True
        assert row['n'] > 1
        assert row['ttft_s'] == (row['t1_ns'] - row['t0_ns']) / 1e9
        assert row['tpot_actual_s'] == (row['tN_ns'] - row['t1_ns']) / 1e9 / (row['n'] - 1)
        assert len(row['result_sha256']) == 64
    old = CompactReferenceRepository(OLD)
    for length in range(1024, 8193, 1024):
        group = [r for r in records if r['length'] == length]
        assert sorted(r['repeat'] for r in group) == [1, 2, 3, 4, 5]
        assert reference.prefill_target(length) == max(r['ttft_s'] for r in group)
        assert reference.decode_target(length) == max(r['tpot_actual_s'] for r in group)
        assert reference.prefill_target(length) >= old.prefill_target(length)
        assert reference.decode_target(length) >= old.decode_target(length)
    assert reference.decode_target(8192) > old.decode_target(8192)


@pytest.mark.parametrize('name', ['manifest.json', 'reference.json', 'source-validation.json'])
def test_files_modified_missing_or_symlinked_rejected(copied, name):
    path = copied / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(LUTError, match='SHA-256'):
        CompactReferenceRepository(copied)
    path.unlink()
    with pytest.raises(LUTError, match='missing or unreadable'):
        CompactReferenceRepository(copied)
    path.symlink_to(SOURCE / name)
    with pytest.raises(LUTError, match='symlink|escapes'):
        CompactReferenceRepository(copied)


@pytest.mark.parametrize('field,value', [('technical_repeats_per_length', 3), ('total_requests', 24)])
def test_repeat3_context_cannot_replace_repeat5_contract(monkeypatch, copied, field, value):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    manifest['context'][field] = value
    semantic_reseal(monkeypatch, copied, manifest)
    with pytest.raises(LUTError, match='context'):
        CompactReferenceRepository(copied)


@pytest.mark.parametrize('bad', ['missing', 'escape', 'extra', 'bytes', 'bytes_type', 'hash'])
def test_evidence_contract_required(monkeypatch, copied, bad):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    spec = manifest['evidence_files']['source-validation.json']
    if bad == 'missing': del manifest['evidence_files']
    elif bad == 'escape': manifest['evidence_files'] = {'../source-validation.json': spec}
    elif bad == 'extra': manifest['evidence_files']['extra.json'] = spec.copy()
    elif bad == 'bytes': spec['bytes'] += 1
    elif bad == 'bytes_type': spec['bytes'] = float(spec['bytes'])
    elif bad == 'hash': spec['sha256'] = '0' * 64
    semantic_reseal(monkeypatch, copied, manifest)
    with pytest.raises(LUTError):
        CompactReferenceRepository(copied)


@pytest.mark.parametrize('bundle', [RUNTIME, ROOT / 'data/luts/latest'])
def test_both_prediction_bundles_use_repeat5_targets_with_evidence_binding(bundle, reference):
    repo = load_lut_repository(bundle, reference_lut_dir=SOURCE)
    target = json.loads(repo.provenance.metadata_json)['target_repository']
    assert target['manifest_sha256'] == MANIFEST_PIN
    assert target['files_sha256']['source-validation.json'] == EVIDENCE_PIN
    assert target['source_conditions']['technical_repeats_per_length'] == 5
    assert repo.decode_target(8192) == reference.decode_target(8192)


@pytest.mark.parametrize('old,new', [(OLD, SOURCE), (SOURCE, OLD)])
def test_repeat3_repeat5_grants_refused_both_directions_before_gpu(performance, capsys, old, new):
    from dataclasses import replace
    from kvolt_runtime.run_cli import _repository, main
    from kvolt_runtime.qualification.contracts import qualification_binding
    from test_performance_cli import write_report
    old_repo = load_lut_repository(RUNTIME, reference_lut_dir=old)
    binding = qualification_binding(repo=old_repo, device_profile=performance.device_profile,
                                    power_profile=performance.power_profile,
                                    native_profile='agx2', runtime=performance.runtime)
    report = write_report(performance.report.parent / 'repeat-binding', binding, {'1024': 4})
    args = list(performance.args)
    args[args.index('--lut-dir') + 1] = str(RUNTIME)
    args[args.index('--qualification') + 1] = str(report)
    args += ['--reference-lut-dir', str(new)]
    assert main(args, factories=replace(performance.seam, repository=_repository)) == 2
    result, = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert result['errors'][0]['stage'] == 'qualification'
    assert 'binding mismatch' in result['errors'][0]['message']
    assert not {'lock', 'backend', 'runner'} & set(performance.events)


@pytest.mark.parametrize('name', ['reference.json', 'source-validation.json'])
def test_reference_failure_precedes_tokenizer_lock_and_backend(copied, tmp_path, capsys, name):
    from dataclasses import replace
    from kvolt_runtime.run_cli import _repository, main
    from test_run_cli import arguments, factories
    (copied / name).unlink()
    events = []
    seam, _ = factories(events)
    args = arguments(tmp_path, '--native-profile', 'agx2')
    index = args.index('--mock')
    args[index:index + 2] = ['--lut-dir', str(RUNTIME), '--reference-lut-dir', str(copied)]
    assert main(args, factories=replace(seam, repository=_repository)) == 2
    result, = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert 'missing or unreadable' in result['errors'][0]['message']
    assert result['errors'][0]['stage'] == 'lut'
    assert not {'tokenizer', 'lock', 'backend', 'runner'} & set(events)


@pytest.mark.parametrize('source,repeats,total', [(SOURCE, 3, 24), (OLD, 5, 40)])
def test_context_cannot_cross_trusted_versions(monkeypatch, tmp_path, source, repeats, total):
    root = Path(shutil.copytree(source, tmp_path / 'cross-version'))
    manifest = json.loads((root / 'manifest.json').read_bytes())
    manifest['context'].update(technical_repeats_per_length=repeats, total_requests=total)
    manifest['validation']['canonical_requests_verified'] = total
    semantic_reseal(monkeypatch, root, manifest)
    with pytest.raises(LUTError, match='context'):
        CompactReferenceRepository(root)


@pytest.mark.parametrize('key', ['canonical_hashes_checked', 'timestamp_metrics_checked',
                                 'request_cpu_gpu_controls_checked'])
@pytest.mark.parametrize('value', [False, 1, None])
def test_canonical_validation_claims_are_strict(monkeypatch, copied, key, value):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    manifest['validation'][key] = value
    semantic_reseal(monkeypatch, copied, manifest)
    with pytest.raises(LUTError, match='source validation'):
        CompactReferenceRepository(copied)


def test_rehashing_repeat5_table_and_envelope_cannot_replace_known_pin(copied):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    table = copied / 'reference.json'
    rows = json.loads(table.read_bytes())
    rows[0][1] /= 2
    data = json.dumps(rows).encode()
    table.write_bytes(data)
    manifest['table'].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (copied / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(LUTError, match='manifest SHA-256'):
        CompactReferenceRepository(copied)
