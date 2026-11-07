"""Real evidence loading and CLI gates with CPU-only hardware substitutes.

Fixture reports exercise validation mechanics; they are never GPU evidence.
"""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
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
    runtime = {'source_worktree_dirty': False, 'source_sha256': {'fixture.py': 'a' * 64},
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
