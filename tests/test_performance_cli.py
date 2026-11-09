"""Real evidence loading and CLI gates with CPU-only hardware substitutes.

Fixture reports exercise validation mechanics; they are never GPU evidence.
"""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvolt_runtime.device import DeviceProfile
from kvolt_runtime.execution.runner import SessionRunner
from kvolt_runtime.policy.decode import DecodeController
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.qualification.contracts import (
    REQUIRED_CHECKS, SCHEMA, QualificationGrant, canonical_json, qualification_binding,
    value_sha256,
)
from kvolt_runtime.telemetry import PowerProfile, RAILS
from test_device_controller import profile_data
from test_run_cli import arguments, factories
from test_session_runner import Adapter, Backend, Clock, Collector, Device


def write_report(root, binding, coverage):
    """Build a self-contained valid loader fixture without replacing the loader."""
    root.mkdir()
    raw = root / 'cpu-fixture.log'
    raw.write_text('CPU regression fixture only; no GPU qualification claimed.\n')
    report = {'schema_version': SCHEMA, 'binding': copy.deepcopy(binding),
              'coverage': coverage, 'checks': {},
              'artifacts': {raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()}}
    for check in REQUIRED_CHECKS:
        summary = {'schema_version': 'kvolt-qualification-check-v1', 'check': check,
                   'binding_sha256': value_sha256(binding), 'coverage': coverage,
                   'command': ['cpu-fixture', '--check', check], 'exit_code': 0,
                   'tests': {'run': 1, 'passed': 1, 'failed': 0, 'skipped': 0},
                   'artifacts': [raw.name]}
        path = root / (check + '.json')
        path.write_text(canonical_json(summary))
        report['artifacts'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['checks'][check] = {'status': 'passed', 'artifact': path.name}
    path = root / 'qualification.json'
    path.write_text(canonical_json(report))
    return path


@pytest.fixture
def performance(tmp_path):
    events = []
    seam, lock = factories(events)
    clock = Clock()
    repo = MockLUTRepository()
    repo.provenance = replace(repo.provenance, kind='measured', source_device='AGX2',
                              source_commit='measured-fixture', manifest_sha256='c' * 64)
    device_profile = DeviceProfile.from_dict(profile_data(tmp_path))
    device_profile.identity_path.write_text('agx2-test\x00\n')
    device_profile.boot_id_path.write_text('boot-one\n')
    power_profile = PowerProfile(
        rails={rail: {'label_path': str(tmp_path / rail / 'label'),
                      'current_path': str(tmp_path / rail / 'current'),
                      'voltage_path': str(tmp_path / rail / 'voltage')} for rail in RAILS},
        interval_ns=1_000_000, max_gap_ns=10**12, ready_timeout_ns=10**9,
        coverage_timeout_ns=10**9, join_timeout_ns=10**9)
    versions = {'numpy': '1.26.4', 'scipy': '1.15.3', 'kneed': '0.9.0'}
    from kvolt_runtime.policy.tpot import tpot_policy
    runtime = {'tpot_policy': tpot_policy(), 'source_worktree_dirty': False, 'source_sha256': {'fixture.py': 'a' * 64},
               'python': 'cpu-fixture', 'platform': 'cpu-fixture',
               'required_policy_versions': versions, 'installed_distribution_versions': dict(versions)}
    binding = qualification_binding(repo=repo, device_profile=device_profile,
                                    power_profile=power_profile, native_profile='agx2', runtime=runtime)
    report = write_report(tmp_path / 'evidence', binding, {'1024': 4})
    observed = {}

    class LockedDevice(Device):
        profile = device_profile
        def receipt(self, operation):
            lock.assert_held()
            return {**super().receipt(operation), 'identity': binding['identity']}

    def backend_factory(args, shared_clock):
        lock.assert_held()
        events.append('backend')
        backend = Backend(shared_clock, events)
        observed['backend'] = backend
        return backend

    def device_factory(profile, held, shared_clock):
        held.assert_held()
        assert profile is device_profile
        events.append('device')
        device = LockedDevice(shared_clock, events)
        observed['device'] = device
        return device

    def collector_factory(profile, shared_clock):
        lock.assert_held()
        collector = Collector(shared_clock, events)
        collector.profile = profile
        return collector

    def runner_factory(**kwargs):
        lock.assert_held()
        observed['runner_kwargs'] = kwargs
        return SessionRunner(**kwargs)

    def repository(args):
        events.append('repo')
        return repo

    seam = replace(seam, repository=repository,
                   device_profile=lambda path: device_profile, power_profile=lambda path: power_profile,
                   runtime_metadata=lambda: runtime, clock=clock, backend=backend_factory,
                   device=device_factory, collector=collector_factory, runner=runner_factory,
                   decode_controller=lambda source, prepared, shared_clock:
                       DecodeController(source, Adapter(events), clock=shared_clock))
    args = arguments(tmp_path)
    index = args.index('--mock')
    del args[index:index + 2]
    args += ['--lut-dir', str(tmp_path / 'lut'), '--mode', 'performance',
             '--collect-power', '--power-profile', str(tmp_path / 'power.json'),
             '--native-profile', 'agx2', '--warmup', 'formal',
             '--qualification', str(report), '--max-new-tokens', '4']
    return SimpleNamespace(args=args, seam=seam, events=events, lock=lock, binding=binding,
                           report=report, repo=repo, runtime=runtime, observed=observed,
                           device_profile=device_profile, power_profile=power_profile)


def stdout_records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.fixture(params=['current', 'explicit_compact'])
def real_compact_performance(performance, request):
    """Real pinned tables, producer and loader; hardware alone stays CPU-only."""
    from kvolt_runtime.run_cli import _repository
    root = Path(__file__).resolve().parents[1]/'data/luts'
    route = request.param
    args = list(performance.args)
    lut_dir = root/('current' if route == 'current' else 'agx2-runtime-repeat5')
    reference = root/'agx2-reference-repeat10' if route == 'explicit_compact' else None
    args[args.index('--lut-dir')+1] = str(lut_dir)
    if reference is not None:
        args += ['--reference-lut-dir', str(reference)]
    repo = _repository(SimpleNamespace(lut_dir=lut_dir, reference_lut_dir=reference, mock=None))
    binding = qualification_binding(repo=repo, device_profile=performance.device_profile,
        power_profile=performance.power_profile, native_profile='agx2', runtime=performance.runtime)
    report = write_report(performance.report.parent.parent/('real-'+route), binding, {'1024': 4})
    args[args.index('--qualification')+1] = str(report)
    def repository(cli_args):
        performance.events.append('repo')
        loaded = _repository(cli_args)
        performance.observed['real_repository'] = loaded
        return loaded
    performance.args, performance.repo, performance.binding, performance.report = args, repo, binding, report
    performance.seam = replace(performance.seam, repository=repository)
    return performance


@pytest.mark.parametrize('explicit_frozen', [False, True])
def test_real_current_and_compact_grant_reaches_real_performance_runner(real_compact_performance, capsys, explicit_frozen):
    from kvolt_runtime.policy.compact import CompactLUTRepository
    from kvolt_runtime.qualification.contracts import _lut_binding
    from kvolt_runtime.qualification.evidence import load_evidence
    from kvolt_runtime.run_cli import main
    fixture = real_compact_performance
    if explicit_frozen:
        fixture.args.append('--no-online-lut')
    grant = load_evidence(fixture.report, fixture.binding)
    assert grant.binding['lut']['reference']['manifest_sha256'] == 'd332875ccce626dd99700da8c7a128bf4801577333e76ee40a987f4051311e40'
    assert main(fixture.args, factories=fixture.seam) == 0
    assert stdout_records(capsys)[-1]['exit_code'] == 0
    actual = fixture.observed['real_repository']
    assert isinstance(actual, CompactLUTRepository)
    assert _lut_binding(actual.provenance) == grant.binding['lut']
    assert fixture.observed['backend'].loads == 1
    assert isinstance(fixture.observed['runner_kwargs']['qualification'], QualificationGrant)
    saved = json.loads((fixture.report.parent.parent/'out/session.json').read_text())
    assert saved['provenance']['formal_evidence_accepted'] is True
    assert saved['provenance']['formal_performance_qualified'] is True
    assert saved['warmup']['formal_coverage_qualified'] is True
    assert saved['results'][0]['generation']['status'] == 'completed'
    assert saved['results'][0]['energy']['valid'] is True
    assert saved['provenance']['online_lut']['enabled'] is False
    assert not list((fixture.report.parent.parent/'out').glob('online-lut*.json'))
    assert not fixture.lock.held


@pytest.mark.parametrize('flags', [['--online-lut-state', 'not-opened.json'], ['--online-lut-alpha', '0.1']])
def test_valid_compact_performance_rejects_explicit_online_options_before_source(
        real_compact_performance, capsys, flags):
    from kvolt_runtime.run_cli import main
    fixture = real_compact_performance
    assert main([*fixture.args, *flags], factories=fixture.seam) == 2
    record = stdout_records(capsys)[-1]
    assert 'omit online state/alpha' in record['errors'][0]['message']
    assert fixture.events == []


def test_valid_compact_grant_does_not_qualify_direct_adaptive_runner(real_compact_performance, capsys):
    from kvolt_runtime.policy.online import OnlineLUTRepository
    from kvolt_runtime.run_cli import main
    fixture = real_compact_performance
    original = fixture.seam.runner
    def adaptive_runner(**kwargs):
        kwargs['repo'] = OnlineLUTRepository(kwargs['repo'],
            state_path=fixture.report.parent/'not-qualified-state.json', context={'cpu_fixture': True})
        return original(**kwargs)
    assert main(fixture.args, factories=replace(fixture.seam, runner=adaptive_runner)) == 2
    record = stdout_records(capsys)[-1]
    assert record['errors'][0]['message'] == 'adaptive LUT has no formal performance qualification'
    assert not any(event in fixture.events for event in ('capture', 'load', 'power_start'))
    saved = json.loads((fixture.report.parent.parent/'out/session.json').read_text())
    assert saved['provenance']['formal_evidence_accepted'] is True
    assert saved['provenance']['formal_performance_qualified'] is False


def test_real_compact_development_feedback_and_cross_session_state_reuse(real_compact_performance, capsys):
    from kvolt_runtime.policy.online import OnlineLUTRepository
    from kvolt_runtime.run_cli import main
    fixture = real_compact_performance
    root = fixture.report.parent.parent
    manifest = root/'online-manifest.json'
    manifest.write_text(canonical_json({'schema_version': 'kvolt-request-manifest-v1',
        'requests': [{'request_id': str(i), 'prompt_file': 'prompt.txt', 'nmax': 3} for i in range(2)]}))
    args = list(fixture.args)
    args[args.index('--mode')+1] = 'development'
    for option in ('--qualification', '--prompt-file', '--max-new-tokens'):
        index = args.index(option)
        del args[index:index+2]
    args += ['--manifest', str(manifest)]
    assert main(args, factories=fixture.seam) == 0
    stdout_records(capsys)
    first = json.loads((root/'out/session.json').read_text())
    assert isinstance(fixture.observed['runner_kwargs']['repo'], OnlineLUTRepository)
    assert first['provenance']['online_lut']['enabled'] is True
    assert [r['online_lut']['used']['revision'] for r in first['results']] == [0, 1]
    final = json.loads((root/'out/online-lut.final.json').read_text())
    assert final['revision'] == 2 and final['alpha'] == 0.1
    baseline = fixture.observed['runner_kwargs']['repo'].base
    expected = {'prefill': {}, 'decode': {}}
    for row in first['results']:
        original = json.loads((root/'out'/row['online_lut']['observation_path']).read_text())
        assert 'feedback' not in original['online_lut']
        assert row['online_lut']['feedback']['status'] == 'updated'
        length = row['request']['length']
        pc, dc = row['decisions']['prefill']['selected'], row['decisions']['decode']['selected']
        p = next(p for p in baseline.prefill(length) if p.frequency_mhz == pc['frequency_mhz'])
        d = next(p for p in baseline.decode(length) if (p.group, p.frequency_mhz) == (dc['group'], dc['frequency_mhz']))
        observations = [
            ('prefill', f"{length}:{p.frequency_mhz}", 'ttft_s', p.ttft_s, row['timing']['ttft_s']),
            ('prefill', f"{length}:{p.frequency_mhz}", 'prefill_energy_j', p.energy_j, row['energy']['prefill_energy_j']),
            ('decode', f"{length}:{d.group}:{d.frequency_mhz}", 'tpot_s', d.tpot_s, row['timing']['mean_tpot_s']),
            ('decode', f"{length}:{d.group}:{d.frequency_mhz}", 'energy_j_per_token', d.energy_j_per_token, row['energy']['decode_j_per_token'])]
        for phase, key, metric, initial, actual in observations:
            cell = expected[phase].setdefault(key, {})
            previous = cell.get(metric)
            cell[metric] = {'ema': 0.9*(previous['ema'] if previous else initial)+0.1*actual,
                            'count': previous['count']+1 if previous else 1}
            if phase == 'decode':
                cell[metric].update(last_n=3, min_n=3, max_n=3)
    assert final['cells'] == expected
    args[args.index('--output-dir')+1] = str(root/'second-online')
    args += ['--online-lut-state', str(root/'out/online-lut.json')]
    assert main(args, factories=fixture.seam) == 0
    stdout_records(capsys)
    second = json.loads((root/'second-online/session.json').read_text())
    assert [r['online_lut']['used']['revision'] for r in second['results']] == [2, 3]
    assert json.loads((root/'second-online/online-lut.final.json').read_text())['revision'] == 4
    assert json.loads((root/'out/online-lut.final.json').read_text())['revision'] == 2


@pytest.mark.parametrize('change', ['reference3', 'reference5', 'reference_sha', 'runtime_manifest',
    'runtime_sha', 'boot', 'device_profile', 'power_profile', 'kernel', 'source', 'scope'])
def test_real_compact_stale_grant_refuses_before_hardware(real_compact_performance, capsys, change):
    from kvolt_runtime.run_cli import main
    fixture = real_compact_performance
    args = list(fixture.args)
    if change in ('reference3', 'reference5'):
        root = Path(__file__).resolve().parents[1]/'data/luts'
        args[args.index('--lut-dir')+1] = str(root/'agx2-runtime-repeat5')
        reference = str(root/('agx2-reference-repeat'+change[-1]))
        if '--reference-lut-dir' in args:
            args[args.index('--reference-lut-dir')+1] = reference
        else:
            args += ['--reference-lut-dir', reference]
    else:
        binding = copy.deepcopy(fixture.binding)
        coverage = {'1024': 3} if change == 'scope' else {'1024': 4}
        if change == 'reference_sha': binding['lut']['reference']['files_sha256']['reference.json'] = '0'*64
        elif change == 'runtime_manifest': binding['lut']['manifest_sha256'] = '0'*64
        elif change == 'runtime_sha': binding['lut']['files_sha256']['decode.json'] = '0'*64
        elif change == 'boot': binding['identity']['boot_id'] = 'stale'
        elif change in ('device_profile', 'power_profile'): binding[change]['name'] = 'stale'
        elif change == 'kernel': binding['native_binaries'][next(iter(binding['native_binaries']))]['sha256'] = '0'*64
        elif change == 'source': binding['runtime_source_sha256']['fixture.py'] = '0'*64
        report = write_report(fixture.report.parent/('stale-'+change), binding, coverage)
        args[args.index('--qualification')+1] = str(report)
    assert main(args, factories=fixture.seam) == 2
    record = stdout_records(capsys)[-1]
    assert record['errors'][0]['stage'] == 'qualification'
    assert 'lock' not in fixture.events and 'backend' not in fixture.events and 'load' not in fixture.events


@pytest.mark.parametrize('change', ['missing_reference', 'reference_manifest', 'reference_files',
                                    'runtime_manifest', 'runtime_files'])
def test_real_runner_rechecks_compact_identity_after_cli_grant_loading(real_compact_performance, capsys, change):
    """A change after the CLI accepts evidence must still block model loading."""
    from kvolt_runtime.run_cli import main
    fixture = real_compact_performance
    original = fixture.seam.runner
    def changed_runner(**kwargs):
        repo = kwargs['repo']
        metadata = json.loads(repo.provenance.metadata_json)
        if change == 'missing_reference': del metadata['target_repository']
        elif change == 'reference_manifest': metadata['target_repository']['manifest_sha256'] = '0'*64
        elif change == 'reference_files': metadata['target_repository']['files_sha256']['reference.json'] = '0'*64
        elif change == 'runtime_manifest': repo.provenance = replace(repo.provenance, manifest_sha256='0'*64)
        else: repo.provenance = replace(repo.provenance, files_sha256=(('decode.json', '0'*64),))
        repo.provenance = replace(repo.provenance, metadata_json=canonical_json(metadata))
        return original(**kwargs)
    assert main(fixture.args, factories=replace(fixture.seam, runner=changed_runner)) == 2
    record = stdout_records(capsys)[-1]
    assert record['errors'][0]['stage'] == 'qualification'
    assert 'load' not in fixture.events and 'capture' not in fixture.events
    saved = json.loads((fixture.report.parent.parent/'out/session.json').read_text())
    assert saved['provenance']['formal_evidence_accepted'] is True
    assert saved['provenance']['formal_performance_qualified'] is False
    assert all(row['generation']['status'] == 'not_run' for row in saved['results'])


@pytest.mark.parametrize('change', ['reference_bytes', 'runtime_bytes', 'selector_pin'])
def test_real_compact_source_bytes_and_selector_pins_refuse_before_hardware(
        real_compact_performance, capsys, change):
    from kvolt_runtime.run_cli import main
    fixture = real_compact_performance
    source = Path(__file__).resolve().parents[1]/'data/luts'
    copied = fixture.report.parent/'copied-luts'
    for directory in ('agx2-runtime-repeat5', 'agx2-reference-repeat10', 'current'):
        shutil.copytree(source/directory, copied/directory)
    if change == 'reference_bytes':
        path = copied/'agx2-reference-repeat10/reference.json'
        path.write_bytes(path.read_bytes()+b'\n')
    elif change == 'runtime_bytes':
        path = copied/'agx2-runtime-repeat5/decode.json'
        path.write_bytes(path.read_bytes()+b'\n')
    else:
        path = copied/'current/selection.json'
        selection = json.loads(path.read_text())
        selection['runtime']['manifest_sha256'] = '0'*64
        path.write_text(canonical_json(selection))
    args = list(fixture.args)
    args[args.index('--lut-dir')+1] = str(copied/'current')
    if '--reference-lut-dir' in args:
        index = args.index('--reference-lut-dir')
        del args[index:index+2]
    assert main(args, factories=fixture.seam) == 2
    record = stdout_records(capsys)[-1]
    assert record['errors'][0]['stage'] == 'lut'
    assert 'lock' not in fixture.events and 'backend' not in fixture.events and 'load' not in fixture.events


def test_performance_without_evidence_refuses_before_output_or_lock(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    events = []
    seam, lock = factories(events)
    assert main(arguments(tmp_path, '--mode', 'performance'), factories=seam) == 2
    record, = stdout_records(capsys)
    assert record['errors'][0]['code'] == 'M4_QUALIFICATION_REQUIRED'
    assert record['artifact_root'] is record['session_path'] is None
    assert not (tmp_path / 'out').exists()
    assert events == [] and not lock.held


@pytest.mark.parametrize('field', ['runtime_source_sha256', 'device_profile', 'power_profile',
                                  'native_profile', 'identity', 'lut'])
def test_real_report_wrong_binding_rejects_on_cpu(performance, capsys, field):
    from kvolt_runtime.run_cli import main
    report = json.loads(performance.report.read_text())
    report['binding'][field] = {'changed': True}
    performance.report.write_text(canonical_json(report))
    assert main(performance.args, factories=performance.seam) == 2
    record, = stdout_records(capsys)
    assert record['errors'][0]['stage'] == 'qualification'
    assert 'binding mismatch' in record['errors'][0]['message']
    assert 'lock' not in performance.events and 'backend' not in performance.events


@pytest.mark.parametrize('coverage', [{'1024': 3}, {'2048': 512}])
def test_actual_request_outside_valid_report_coverage_refuses_before_lock(performance, capsys, coverage):
    from kvolt_runtime.run_cli import main
    report = write_report(performance.report.parent / 'scoped', performance.binding, coverage)
    args = list(performance.args)
    args[args.index('--qualification') + 1] = str(report)
    assert main(args, factories=performance.seam) == 2
    record, = stdout_records(capsys)
    assert 'outside qualified coverage' in record['errors'][0]['message']
    assert 'lock' not in performance.events and 'backend' not in performance.events


@pytest.mark.parametrize('artifact', ['backend_exact.json', 'cpu-fixture.log', 'qualification.json'])
def test_tampered_real_report_or_artifact_refuses_before_lock(performance, capsys, artifact):
    from kvolt_runtime.run_cli import main
    path = performance.report.parent / artifact
    if artifact == 'qualification.json':
        path.write_text('{"schema_version": "forged"}')
    else:
        path.write_text(path.read_text() + '\ntampered\n')
    assert main(performance.args, factories=performance.seam) == 2
    record, = stdout_records(capsys)
    assert record['errors'][0]['stage'] == 'qualification'
    assert 'lock' not in performance.events and 'backend' not in performance.events


@pytest.mark.parametrize('change', ['dirty', 'installed-version', 'boot', 'serial', 'lut-source'])
def test_current_environment_change_invalidates_real_evidence_before_lock(performance, capsys, change):
    from kvolt_runtime.run_cli import main
    if change == 'dirty': performance.runtime['source_worktree_dirty'] = True
    if change == 'installed-version': performance.runtime['installed_distribution_versions']['scipy'] = '0'
    if change == 'boot': performance.device_profile.boot_id_path.write_text('next-boot')
    if change == 'serial': performance.device_profile.identity_path.write_text('wrong-device')
    if change == 'lut-source': performance.repo.provenance = replace(performance.repo.provenance, source_commit='changed')
    assert main(performance.args, factories=performance.seam) == 2
    record, = stdout_records(capsys)
    assert record['errors'][0]['stage'] == 'qualification'
    assert 'lock' not in performance.events and 'backend' not in performance.events


def test_real_grant_flows_through_locked_formal_runner_with_power(performance, capsys):
    from kvolt_runtime.run_cli import main
    assert main(performance.args, factories=performance.seam) == 0
    records = stdout_records(capsys)
    assert [record['record_type'] for record in records] == ['request_result', 'session_result']
    kwargs = performance.observed['runner_kwargs']
    assert isinstance(kwargs['qualification'], QualificationGrant)
    assert kwargs['warmup'] == 'formal' and callable(kwargs['collector_factory'])
    assert not performance.lock.held and performance.events[-1] == 'unlock'
    assert performance.events.index('lock') < performance.events.index('backend')
    assert performance.events.index('capture') < performance.events.index('load')
    assert performance.events.index('restore') < performance.events.index('power_start')
    assert performance.observed['backend'].loads == 1
    saved = json.loads((performance.report.parent.parent / 'out' / 'session.json').read_text())
    assert saved['mode'] == 'performance' and saved['provenance']['formal_performance_qualified'] is True
    assert saved['qualification']['report_sha256'] == hashlib.sha256(performance.report.read_bytes()).hexdigest()
    assert saved['warmup']['formal_coverage_qualified'] is True
    assert [case['decode_forwards'] for case in saved['warmup']['cases']] == [3] * 5
    assert saved['results'][0]['energy']['valid'] is True
    assert saved['warmup']['ended_ns'] < saved['results'][0]['timing']['t0_ns']
    assert saved['provenance']['formal_evidence_accepted'] is True


def test_locked_identity_recheck_blocks_model_load_even_with_valid_cpu_evidence(performance, capsys):
    from kvolt_runtime.run_cli import main
    factory = performance.seam.device
    def mismatched_device(profile, lock, clock):
        device = factory(profile, lock, clock)
        capture = device.capture
        device.capture = lambda: {**capture(), 'identity': {'boot_id': 'changed-under-lock'}}
        return device
    seam = replace(performance.seam, device=mismatched_device)
    assert main(performance.args, factories=seam) == 2
    record = stdout_records(capsys)[-1]
    assert 'identity differs' in record['errors'][0]['message']
    assert 'capture' in performance.events and 'load' not in performance.events
    assert performance.events[-1] == 'unlock'


@pytest.mark.parametrize('failure,code', [('busy', 3), ('preparation', 3), ('warmup', 1), ('power', 1)])
def test_accepted_evidence_does_not_qualify_failed_session(performance, capsys, failure, code):
    from kvolt_runtime.run_cli import main
    seam = performance.seam
    if failure == 'busy':
        performance.lock.busy = True
    elif failure == 'preparation':
        original = seam.device
        def failed_device(*args):
            device = original(*args)
            device.fail = 'verify'
            return device
        seam = replace(seam, device=failed_device)
    elif failure == 'warmup':
        original = seam.backend
        def failed_backend(*args):
            backend = original(*args)
            backend.fail = 'pack'
            return backend
        seam = replace(seam, backend=failed_backend)
    else:
        original = seam.collector
        def failed_collector(*args):
            collector = original(*args)
            collector.failure = 'start'
            return collector
        seam = replace(seam, collector=failed_collector)
    assert main(performance.args, factories=seam) == code
    assert stdout_records(capsys)[-1]['exit_code'] == code
    saved = json.loads((performance.report.parent.parent / 'out' / 'session.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['provenance']['formal_evidence_accepted'] is True
    assert saved['provenance']['formal_performance_qualified'] is False


def test_unlock_failure_clears_final_performance_qualification(performance, capsys):
    from kvolt_runtime.run_cli import main
    lock_type = type(performance.lock)
    class FailingUnlock(lock_type):
        def __exit__(self, *args):
            super().__exit__(*args)
            raise OSError('fixture unlock failure')
    performance.lock.__class__ = FailingUnlock
    assert main(performance.args, factories=performance.seam) == 1
    assert stdout_records(capsys)[-1]['exit_code'] == 1
    saved = json.loads((performance.report.parent.parent / 'out' / 'session.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['provenance']['formal_performance_qualified'] is False


def test_qualification_is_rejected_in_development_before_any_output(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events)
    assert main(arguments(tmp_path, '--qualification', str(tmp_path / 'report.json')), factories=seam) == 2
    record, = stdout_records(capsys)
    assert '--qualification is only used' in record['errors'][0]['message']
    assert events == [] and not (tmp_path / 'out').exists()


def test_development_formal_passes_without_qualification_grant(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    events, observed = [], {}
    seam, _ = factories(events)
    base = seam.runner
    class InspectRunner(base):
        def __init__(self, **kwargs):
            observed.update(kwargs)
            super().__init__(**kwargs)
    assert main(arguments(tmp_path, '--warmup', 'formal'), factories=replace(seam, runner=InspectRunner)) == 0
    assert observed['warmup'] == 'formal' and 'qualification' not in observed
    assert stdout_records(capsys)[-1]['exit_code'] == 0


def test_performance_imports_and_help_remain_cpu_only():
    code = """import sys
from kvolt_runtime.qualification.contracts import QualificationGrant, qualification_binding
from kvolt_runtime.qualification.evidence import load_evidence
from kvolt_runtime.execution.runner import SessionRunner
from kvolt_runtime.cli import main
try: main(['run', '--help'])
except SystemExit as error: assert error.code == 0
assert not ({'torch','transformers','triton','flash_attn','numpy','scipy','kneed'} & set(sys.modules))
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert '--qualification' in result.stdout and 'formal' in result.stdout


@pytest.mark.parametrize('stale_ratio', [None, 0.0, .1, .3])
def test_tpot_policy_stale_grant_rejected_before_hardware(performance, capsys, stale_ratio):
    from kvolt_runtime.policy.tpot import tpot_policy
    from kvolt_runtime.run_cli import main
    fixture = performance
    binding = copy.deepcopy(fixture.binding)
    if stale_ratio is None:
        binding.pop('tpot_policy')
    else:
        binding['tpot_policy'] = tpot_policy(stale_ratio)
    report = write_report(fixture.report.parent.parent/'stale-tpot-evidence', binding, {'1024': 4})
    args = list(fixture.args)
    args[args.index('--qualification')+1] = str(report)
    assert main(args, factories=fixture.seam) == 2
    assert not any(event in fixture.events for event in ('lock', 'capture', 'load', 'power_start'))
    assert stdout_records(capsys)[-1]['errors'][0]['stage'] == 'qualification'


def test_global_tpot_ratio_change_invalidates_old_runtime_evidence(performance, capsys, monkeypatch):
    from kvolt_runtime import config
    from kvolt_runtime.run_cli import main
    fixture = performance
    monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', .3)
    assert main(fixture.args, factories=fixture.seam) == 2
    assert not any(event in fixture.events for event in ('lock', 'capture', 'load', 'power_start'))
    assert 'TPOT buffer policy' in stdout_records(capsys)[-1]['errors'][0]['message']


def test_direct_runner_refuses_grant_after_global_policy_change(performance, monkeypatch):
    from kvolt_runtime import config
    from kvolt_runtime.run_cli import main
    fixture = performance
    original = fixture.seam.runner
    def changed_after_preflight(**kwargs):
        # Simulate a global edit after CLI qualification binding was accepted.
        runner = original(**kwargs)
        monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', .3)
        return runner
    assert main(fixture.args, factories=replace(fixture.seam, runner=changed_after_preflight),
                emit=lambda _: None) == 2
    assert not any(event in fixture.events for event in ('capture', 'load', 'power_start'))
