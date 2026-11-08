"""Repeat-ten targets bind the tracked canonical validation evidence."""
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
SOURCE = ROOT / 'data/luts/agx2-reference-repeat10'
OLD = ROOT / 'data/luts/agx2-reference-repeat3'
PREVIOUS = ROOT / 'data/luts/agx2-reference-repeat5'
RUNTIME = ROOT / 'data/luts/agx2-runtime-repeat5'
MANIFEST_PIN = 'd332875ccce626dd99700da8c7a128bf4801577333e76ee40a987f4051311e40'
TABLE_PIN = '359fd02c609738755c87847f27b19286f45a3a7cb8d6bef8acd7c369e1c30c2b'
EVIDENCE_PIN = 'f954289b28ac94e3945d7ffeb0efa26e851fdfdc8025988181a94bdeabe8e8ea'


EXPECTED = (
    (1024, 0.817132383, 0.10339920563534676),
    (2048, 1.978160811, 0.10827898330724071),
    (3072, 2.963832699, 0.11288540909197652),
    (4096, 3.951638709, 0.11761419499021526),
    (5120, 3.796079475, 0.1225068992113503),
    (6144, 4.624497683, 0.12733681808219177),
    (7168, 5.44257839, 0.1322296097514677),
    (8192, 6.313309433, 0.13676398722700586),
)


@pytest.fixture
def reference():
    return CompactReferenceRepository(SOURCE)


@pytest.fixture
def copied(tmp_path):
    return Path(shutil.copytree(SOURCE, tmp_path / 'reference'))


def test_repeat10_known_pin_all_targets_and_complete_provenance(reference):
    assert module.COMPACT_REFERENCE_REPEAT10_MANIFEST_SHA256 == MANIFEST_PIN
    manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
    assert hashlib.sha256((SOURCE / 'manifest.json').read_bytes()).hexdigest() == MANIFEST_PIN
    assert reference.provenance.manifest_sha256 == MANIFEST_PIN
    assert dict(reference.provenance.files_sha256) == {
        'reference.json': TABLE_PIN, 'source-validation.json': EVIDENCE_PIN}
    assert json.loads(reference.provenance.metadata_json) == {'reference_manifest': manifest}
    assert 'locally generated' in manifest['provenance']['manifest_origin']
    assert manifest['context']['technical_repeats_per_length'] == 10
    assert manifest['context']['total_requests'] == manifest['validation']['canonical_requests_verified'] == 80
    assert manifest['validation']['controller_integration_validated'] is False
    assert manifest['validation']['original_component_manifests_rechecked'] is False
    assert manifest['policy_multipliers'] == {'prefill': 1.0, 'decode': 1.0}
    evidence_data = (SOURCE / 'source-validation.json').read_bytes()
    assert hashlib.sha256(evidence_data).hexdigest() == EVIDENCE_PIN
    assert manifest['evidence_files']['source-validation.json'] == {
        'sha256': EVIDENCE_PIN, 'bytes': len(evidence_data)}
    data = (SOURCE / 'reference.json').read_bytes()
    assert len(data) == 310
    assert hashlib.sha256(data).hexdigest() == TABLE_PIN
    rows = json.loads(data)
    assert rows == [list(row) for row in EXPECTED]
    for length, prefill, decode in rows:
        assert reference.prefill_target(length) == prefill
        assert reference.decode_target(length) == decode
    for lower, upper in zip(rows, rows[1:]):
        for offset in (1, 137, 512, 1023):
            weight = offset / 1024
            for method, column in [('prefill_target', 1), ('decode_target', 2)]:
                assert getattr(reference, method)(lower[0] + offset) == (
                    (1 - weight) * lower[column] + weight * upper[column])


def test_tracked_evidence_recomputes_80_timestamps_and_independent_maxima(reference):
    evidence = json.loads((SOURCE / 'source-validation.json').read_bytes())
    assert evidence['targets'] == [list(row) for row in EXPECTED]
    assert evidence['previous_repeat5_records_preserved'] == 40
    assert evidence['previous_repeat5_evidence_sha256'] == hashlib.sha256(
        (PREVIOUS / 'source-validation.json').read_bytes()).hexdigest()
    assert sorted(source['canonical_count'] for source in evidence['sources']) == [16, 24, 40]
    assert all(source['original_directory_available'] is False for source in evidence['sources'])
    assert 'not newly rechecked' in evidence['scope']
    records = evidence['records']
    assert len(records) == 80
    assert len({row['id'] for row in records}) == 80
    for row in records:
        assert row['controls_verified'] is True
        assert row['n'] > 1
        assert row['ttft_s'] == (row['t1_ns'] - row['t0_ns']) / 1e9
        assert row['tpot_actual_s'] == (row['tN_ns'] - row['t1_ns']) / 1e9 / (row['n'] - 1)
        assert len(row['result_sha256']) == 64
    old = CompactReferenceRepository(PREVIOUS)
    for length in range(1024, 8193, 1024):
        group = [r for r in records if r['length'] == length]
        assert sorted(r['repeat'] for r in group) == list(range(1, 11))
        assert reference.prefill_target(length) == max(r['ttft_s'] for r in group)
        assert reference.decode_target(length) == max(r['tpot_actual_s'] for r in group)
        assert reference.prefill_target(length) >= old.prefill_target(length)
        assert reference.decode_target(length) >= old.decode_target(length)
    assert reference.decode_target(8192) > old.decode_target(8192)
    # Independent maxima need not come from one request; do not select a paired row.
    assert any(
        not any(r['ttft_s'] == reference.prefill_target(length)
                and r['tpot_actual_s'] == reference.decode_target(length) for r in records
                if r['length'] == length)
        for length in range(1024, 8193, 1024))
    previous_records = json.loads((PREVIOUS / 'source-validation.json').read_bytes())['records']
    by_id = {row['id']: row for row in records}
    for previous in previous_records:
        assert {key: by_id[previous['id']][key] for key in previous} == previous


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


@pytest.mark.parametrize('field,value', [('technical_repeats_per_length', 3),
                                         ('technical_repeats_per_length', 5),
                                         ('total_requests', 24), ('total_requests', 40)])
def test_old_context_cannot_replace_repeat10_contract(monkeypatch, copied, field, value):
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
def test_both_prediction_bundles_use_repeat10_targets_with_evidence_binding(bundle, reference):
    repo = load_lut_repository(bundle, reference_lut_dir=SOURCE)
    target = json.loads(repo.provenance.metadata_json)['target_repository']
    assert target['manifest_sha256'] == MANIFEST_PIN
    assert target['files_sha256']['source-validation.json'] == EVIDENCE_PIN
    assert target['source_conditions']['technical_repeats_per_length'] == 10
    assert repo.decode_target(8192) == reference.decode_target(8192)


@pytest.mark.parametrize('old,new', [(old, new) for prior in (OLD, PREVIOUS)
                                      for old, new in ((prior, SOURCE), (SOURCE, prior))])
def test_repeat3_repeat5_repeat10_grants_refused_both_directions_before_gpu(performance, capsys, old, new):
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


@pytest.mark.parametrize('source,repeats,total', [(SOURCE, 3, 24), (SOURCE, 5, 40), (OLD, 10, 80), (PREVIOUS, 10, 80)])
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


@pytest.mark.parametrize('name', ['reference.json', 'source-validation.json'])
def test_rehashing_repeat10_file_and_envelope_cannot_replace_known_pin(copied, name):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    path = copied / name
    value = json.loads(path.read_bytes())
    if name == 'reference.json':
        value[0][1] /= 2
        spec = manifest['table']
    else:
        value['records'][0]['ttft_s'] /= 2
        spec = manifest['evidence_files'][name]
    data = json.dumps(value).encode()
    path.write_bytes(data)
    spec.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (copied / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(LUTError, match='manifest SHA-256'):
        CompactReferenceRepository(copied)


@pytest.mark.parametrize('field,value', [
    ('technical_repeats_per_length', 10.0), ('total_requests', 80.0),
    ('cpu_khz', 2201600.0), ('emc_hz', 3199000000.0),
    ('gpu_min_hz', 306000000.0), ('gpu_max_hz', 1122000000.0),
    ('documents_per_length', True), ('kv_precision', 'K2V2'),
    ('gpu_policy', 'fixed_1122'), ('gpu_governor', 'performance'),
    ('aggregation', 'mean_TTFT_and_TPOT_per_length'),
    ('decode_TPOT', '(tN-t1)/N'),
    ('generation', {'force_length': False, 'max_new_tokens': 512.0}),
])
def test_repeat10_measurement_conditions_are_exact(monkeypatch, copied, field, value):
    manifest = json.loads((copied / 'manifest.json').read_bytes())
    manifest['context'][field] = value
    semantic_reseal(monkeypatch, copied, manifest)
    with pytest.raises(LUTError, match='context'):
        CompactReferenceRepository(copied)
