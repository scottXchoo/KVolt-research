"""CPU tests for reviewed planning, control cleanup and report publication."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvolt_runtime.qualification.contracts import REQUIRED_CHECKS, file_sha256
from kvolt_runtime.qualification.evidence import load_evidence


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location('qualify_m4_test_driver', Path(__file__).parents[1]/'tools/qualify_m4.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plan_bundle(tmp_path, driver, monkeypatch):
    reference = tmp_path.parent/(tmp_path.name+'-reference')
    helper = reference/'scripts/reference.py'
    helper.parent.mkdir(parents=True)
    helper.write_text('pinned independent reference\n')
    monkeypatch.setattr(driver, 'BASELINE_HELPERS', {'scripts/reference.py': file_sha256(helper)})
    profile = {
        'gpu_path': '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu',
        'emc_rate_path': '/sys/kernel/debug/bpmp/debug/clk/emc/rate',
        'emc_lock_path': '/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked',
        'emc_max_rate_path': '/sys/kernel/debug/bpmp/debug/clk/emc/max_rate',
        'emc_cap_path': '/sys/kernel/nvpmodel_clk_cap/emc',
        'emc_halt_path': '/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt',
        'fan_path': '/sys/class/hwmon/hwmon2/pwm1',
        'fan_policy_path': '/sys/class/thermal/thermal_zone1/policy',
        'cpu_policy_paths': ['/sys/devices/system/cpu/cpufreq/policy'+str(i) for i in (0, 4, 8)],
    }
    driver.write(tmp_path/'device-profile.json', profile)
    driver.write(tmp_path/'power-profile.json', {'record': 'power'})
    driver.write(tmp_path/'cohort/manifest.json', {'requests': []})
    driver.write(tmp_path/'eos-cohort/manifest.json', {'requests': []})
    (tmp_path/'cpu').mkdir()
    cpu_xml = ('<testsuite><testcase/><testcase/>'
        + '<testcase classname="tests.integration.test_cache_equivalence_gpu"><skipped/></testcase>'*101
        + '<testcase classname="tests.integration.test_backend_prefill_gpu"><skipped/></testcase>'
        + '<testcase classname="tests.integration.test_greedy_equivalence_gpu"><skipped/></testcase></testsuite>')
    gpu_xml = '<testsuite>'+'<testcase/>'*102+'</testsuite>'
    for name, value in (('tests.xml', cpu_xml), ('gpu.xml', gpu_xml), ('pytest.log', '2 passed, 1 skipped'), ('gpu.log', '2 passed')):
        (tmp_path/'cpu'/name).write_text(value)
    # Plan-path/count validation does not need to duplicate the external pinned
    # M1 fixture. Stub only its content pin, while retaining real per-file hashes.
    (tmp_path/'cpu/m1-cohort.json').write_text('pinned M1 fixture')
    (tmp_path/'cpu/b1-prefill-report.json').write_text('{"status":"passed"}')
    driver.write(tmp_path/'cpu/reference-helpers.json', driver.baseline_sources(reference))
    actual_sha = driver.file_sha256
    monkeypatch.setattr(driver, 'file_sha256', lambda path: 'e9f0c05e0d21e13356f6a64d1176b03bcff613c708b0b0fd72228f711aadc20d'
        if Path(path) == tmp_path/'cpu/m1-cohort.json' else actual_sha(path))
    plan = {'schema_version': 'kvolt-m4-plan-v1', 'source_commit': 'commit', 'source_sha256': {'src/a.py': 'a'*64},
            'root': str(tmp_path), 'model_dir': '/model', 'kivi_root': '/kivi', 'baseline_root': str(reference), 'lut_dir': '/lut',
            'device_profile': str(tmp_path/'device-profile.json'), 'power_profile': str(tmp_path/'power-profile.json'),
            'manifest': str(tmp_path/'cohort/manifest.json'), 'eos_manifest': str(tmp_path/'eos-cohort/manifest.json'),
            'coverage': {'1024': 512, '4096': 512, '8192': 512}, 'cpu': driver.junit(tmp_path/'cpu/tests.xml'),
            'gpu': driver.junit(tmp_path/'cpu/gpu.xml'), 'requests': [], 'eos_requests': [],
            'formal_evaluation_run': False, 'prepared_ns': 1}
    def save(*, refresh_files=True):
        if refresh_files:
            plan['frozen_files'] = {str(path.relative_to(tmp_path)): driver.file_sha256(path)
                for path in tmp_path.rglob('*') if path.is_file() and path.name != 'plan.json'}
        driver.write(tmp_path/'plan.json', plan)
        return file_sha256(tmp_path/'plan.json')
    monkeypatch.setattr(driver, 'clean_source', lambda: 'commit')
    monkeypatch.setattr(driver, 'source_hashes', lambda: {'src/a.py': 'a'*64})
    save()
    return SimpleNamespace(root=tmp_path, path=tmp_path/'plan.json', plan=plan, save=save)


def test_validate_frozen_plan_without_executing_recorded_commands(driver, plan_bundle, monkeypatch):
    monkeypatch.setattr(driver.subprocess, 'run', lambda *a, **kw: pytest.fail('recorded command executed'))
    assert driver.validate_plan(plan_bundle.path, file_sha256(plan_bundle.path)) == plan_bundle.plan


@pytest.mark.parametrize('mutation', ['digest', 'escape', 'symlink', 'source', 'counts', 'coverage',
                                     'profile_target', 'profile_path', 'missing_file', 'unknown_key', 'baseline_changed'])
def test_unsafe_or_stale_plan_denied(driver, plan_bundle, monkeypatch, mutation):
    bundle = plan_bundle
    if mutation == 'digest':
        with pytest.raises(ValueError, match='SHA-256'):
            driver.validate_plan(bundle.path, '0'*64)
        return
    if mutation == 'escape':
        bundle.plan['frozen_files']['../outside'] = 'a'*64
    elif mutation == 'symlink':
        (bundle.root/'cpu/gpu.log').unlink()
        (bundle.root/'cpu/gpu.log').symlink_to('pytest.log')
    elif mutation == 'source':
        monkeypatch.setattr(driver, 'source_hashes', lambda: {'src/a.py': 'b'*64})
    elif mutation == 'counts':
        bundle.plan['gpu']['skipped'] = 1
    elif mutation == 'coverage':
        bundle.plan['coverage']['8192'] = 128
    elif mutation == 'profile_target':
        profile = driver.read_json(bundle.root/'device-profile.json')
        profile['emc_rate_path'] = '/etc/passwd'
        driver.write(bundle.root/'device-profile.json', profile)
        bundle.save()
    elif mutation == 'profile_path':
        bundle.plan['device_profile'] = '/etc/profile'
    elif mutation == 'missing_file':
        (bundle.root/'cpu/gpu.log').unlink()
    elif mutation == 'unknown_key':
        bundle.plan['command'] = ['arbitrary', 'root', 'command']
    elif mutation == 'baseline_changed':
        (Path(bundle.plan['baseline_root'])/'scripts/reference.py').write_text('changed reference')
    digest = bundle.save(refresh_files=False)
    with pytest.raises(ValueError):
        driver.validate_plan(bundle.path, digest)


def test_freeze_preserves_diagnostics_and_adds_full_cap_for_every_prompt(driver, tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    for length in (1024, 4096, 8192):
        (source/f'{length}.txt').write_text('a'*length)
    original = [dict(request_id=str(i), prompt_file=f'{length}.txt', nmax=cap)
                for i, (length, cap) in enumerate(((1024, 1), (1024, 2), (4096, 128), (8192, 128), (1024, 128)))]
    driver.write(source/'manifest.json', {'schema_version': 'kvolt-request-manifest-v1', 'requests': original})
    manifest = driver.freeze_manifest(source/'manifest.json', tmp_path/'frozen', full_caps=True)
    requests = driver.read_json(manifest)['requests']
    assert [row['nmax'] for row in requests[:5]] == [1, 2, 128, 128, 128]
    assert len(requests) == 8
    full_lengths = {len((manifest.parent/row['prompt_file']).read_bytes()) for row in requests if row['nmax'] == 512}
    assert full_lengths == {1024, 4096, 8192}


def control_snapshots():
    gpu = {'verified': True, 'identity': {'physical_id_sha256': 'a'*64, 'boot_id': 'boot'},
           'observed': {'governor': 'nvhost_podgov', 'min_hz': 306000000, 'max_hz': 1122000000, 'cur_hz': 408000000}}
    external = {'cpu': [{'path': 'cpu0', 'minimum': 1, 'maximum': 2, 'governor': 'schedutil'}],
        'emc': {'lock': 0, 'halt': 0, 'rate': 100}, 'mode': {'name': 'MAXN', 'mode_id': 0},
        'tpc': '0', 'fan_service': 'active', 'fan_policy': 'step_wise', 'fan_pwm': 100}
    return gpu, external


@pytest.mark.parametrize('failure', ['external_capture', 'prepare', 'gpu_restore', 'body', 'after_cpu'])
def test_hardware_scope_always_attempts_both_restores(driver, tmp_path, monkeypatch, failure):
    import kvolt_runtime.device as device_module
    import kvolt_runtime.qualification.session as preparation_module
    events = []
    gpu, external = control_snapshots()
    class Device:
        def __init__(self, *a, **kw): pass
        def capture(self):
            events.append('gpu_capture')
            return copy.deepcopy(gpu)
        def restore(self, original):
            events.append('gpu_restore')
            if failure == 'gpu_restore':
                raise RuntimeError('GPU restore failed')
            return copy.deepcopy(gpu)
    class Preparation:
        def __init__(self, *a, **kw): self.captures = 0
        def capture(self):
            self.captures += 1
            events.append('external_capture')
            if failure == 'external_capture' and self.captures == 1:
                raise RuntimeError('external capture failed')
            result = copy.deepcopy(external)
            if failure == 'after_cpu' and self.captures == 2:
                result['cpu'][0]['minimum'] = 999
            return result
        def prepare(self):
            events.append('prepare')
            if failure == 'prepare':
                raise RuntimeError('prepare failed after write')
            return {'prepared': True}
        def restore(self):
            events.append('external_restore')
            return {'verified': True}
    monkeypatch.setattr(device_module, 'DeviceController', Device)
    monkeypatch.setattr(preparation_module, 'ExperimentPreparation', Preparation)
    with pytest.raises((ValueError, RuntimeError)):
        with driver.hardware_scope(tmp_path, '', object(), object()):
            if failure == 'body':
                raise RuntimeError('generation failed')
    assert events.count('gpu_restore') == events.count('external_restore') == 1
    assert (tmp_path/'restoration-audit.json').is_file()


@pytest.mark.parametrize('failure', ['performance', 'source', 'final_loader'])
def test_failed_performance_or_publication_removes_every_usable_report(driver, tmp_path, monkeypatch, failure):
    import kvolt_runtime.qualification.evidence as evidence
    plan = {'root': str(tmp_path), 'coverage': {'1024': 512}, 'source_commit': 'commit', 'source_sha256': {'a': 'hash'}}
    driver.write(tmp_path/'plan.json', plan)
    driver.write(tmp_path/'raw.json', {'actual_checks': 'executed'})
    original = file_sha256(tmp_path/'raw.json')
    summaries = {key: {'status': 'passed'} for key in REQUIRED_CHECKS}
    binding = {'trusted_test_scope': 1}
    def performance(*args):
        assert (tmp_path/'qualification.candidate.json').is_file()
        assert not (tmp_path/'qualification.json').exists()
        assert load_evidence(tmp_path/'qualification.candidate.json', binding)
        if failure == 'performance':
            raise RuntimeError('real performance failed')
    monkeypatch.setattr(driver, 'validate_performance', performance)
    monkeypatch.setattr(driver, 'clean_source', lambda: 'changed' if failure == 'source' else 'commit')
    monkeypatch.setattr(driver, 'source_hashes', lambda: {'a': 'hash'})
    original_loader = evidence.load_evidence
    def loader(path, binding):
        if failure == 'final_loader' and Path(path).name == 'qualification.json':
            raise RuntimeError('final loader failed')
        return original_loader(path, binding)
    monkeypatch.setattr(evidence, 'load_evidence', loader)
    with pytest.raises((RuntimeError, ValueError)):
        driver.publish_evidence(plan, binding, summaries, object())
    assert not (tmp_path/'qualification.json').exists()
    assert not (tmp_path/'qualification.candidate.json').exists()
    assert file_sha256(tmp_path/'raw.json') == original


def test_publication_preserves_frozen_bytes_and_exercises_candidate_first(driver, tmp_path, monkeypatch):
    plan = {'root': str(tmp_path), 'coverage': {'1024': 512}, 'source_commit': 'commit', 'source_sha256': {'a': 'hash'}}
    driver.write(tmp_path/'plan.json', plan)
    driver.write(tmp_path/'raw.json', {'actual_checks': 'executed'})
    binding, seen = {'trusted_test_scope': 1}, []
    def performance(*args):
        seen.append(file_sha256(tmp_path/'qualification.candidate.json'))
        assert not (tmp_path/'qualification.json').exists()
        driver.write(tmp_path/'performance-proof.json', {'actual_performance': 'passed'})
    monkeypatch.setattr(driver, 'validate_performance', performance)
    monkeypatch.setattr(driver, 'clean_source', lambda: 'commit')
    monkeypatch.setattr(driver, 'source_hashes', lambda: {'a': 'hash'})
    grant = driver.publish_evidence(plan, binding, {key: {'status': 'passed'} for key in REQUIRED_CHECKS}, object())
    assert grant.report_path == str(tmp_path/'qualification.json')
    assert grant.report_sha256 == seen[0]
    assert not (tmp_path/'qualification.candidate.json').exists()
    assert 'performance-proof.json' not in driver.read_json(tmp_path/'qualification.json')['artifacts']


def test_execute_parser_requires_reviewed_plan_digest(driver):
    with pytest.raises(SystemExit) as error:
        driver.main(['execute', 'plan.json'])
    assert error.value.code == 2


def test_skipped_cpu_data_check_is_not_a_passing_suite(driver, tmp_path):
    path = tmp_path/'tests.xml'
    path.write_text('<testsuite><testcase/><testcase classname="tests.test_policy_lut"><skipped/></testcase></testsuite>')
    with pytest.raises(ValueError, match='103 GPU'):
        driver.junit(path, cpu_suite=True)


def test_gpu_stage_explicitly_enables_both_gates_and_freezes_paths(driver, tmp_path):
    args = SimpleNamespace(kivi_root=tmp_path/'kivi-source/third_party/KIVI', model_dir=tmp_path/'model',
                           baseline_root=tmp_path/'independent-reference')
    env = driver.gpu_environment({'KVOLT_HARDWARE_LOCK': '/tmp/unsafe', 'KVOLT_MODEL_SNAPSHOT': '/wrong'}, args, tmp_path)
    assert env['KVOLT_RUN_GPU'] == env['KVOLT_RUN_B1_GPU'] == '1'
    assert env['KVOLT_HARDWARE_LOCK'] == '/run/lock/kvolt-p3-hardware.lock'
    assert env['KVOLT_MODEL_SNAPSHOT'] == str(tmp_path/'model')
    assert env['KVOLT_BASELINE'] == str(tmp_path/'independent-reference')
    assert env['KVOLT_M1_COHORT'] == str(tmp_path/'cpu/m1-cohort.json')
    assert env['KVOLT_B1_REPORT'] == str(tmp_path/'cpu/b1-prefill-report.json')


def test_cpu_stage_disables_inherited_gpu_opt_ins_and_uses_actual_luts(driver, tmp_path, monkeypatch):
    monkeypatch.setenv('KVOLT_RUN_GPU', '1')
    monkeypatch.setenv('KVOLT_RUN_B1_GPU', '1')
    monkeypatch.setenv('KVOLT_LUT_BUNDLE', '/wrong/lut')
    environment = driver.cpu_environment(tmp_path/'measured')
    assert 'KVOLT_RUN_GPU' not in environment and 'KVOLT_RUN_B1_GPU' not in environment
    assert environment['KVOLT_LUT_BUNDLE'] == str(tmp_path/'measured')


def test_progress_channel_failure_cannot_abort_finally_cleanup(driver, monkeypatch):
    class Closed:
        def write(self, value): raise OSError('progress pipe closed')
    monkeypatch.setattr(driver.sys, 'stderr', Closed())
    driver.progress('final restoration starting')


@pytest.mark.parametrize('state', ['valid', 'missing', 'tampered'])
def test_independent_baseline_helpers_are_pinned_before_gpu_stage(driver, tmp_path, monkeypatch, state):
    helper = tmp_path/'scripts/reference.py'
    helper.parent.mkdir()
    contents = b'independent pinned reference\n'
    monkeypatch.setattr(driver, 'BASELINE_HELPERS', {'scripts/reference.py': hashlib.sha256(contents).hexdigest()})
    if state != 'missing':
        helper.write_bytes(contents if state == 'valid' else b'changed source')
    if state == 'valid':
        assert driver.baseline_sources(tmp_path)['files_sha256'] == driver.BASELINE_HELPERS
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            driver.baseline_sources(tmp_path)


def test_actual_performance_validation_uses_new_lock_and_candidate(driver, tmp_path, monkeypatch):
    import kvolt_runtime.device as device_module
    import kvolt_runtime.policy.lut as lut_module
    events = []
    lock = object()
    @contextmanager
    def locked():
        events.append('lock')
        yield lock
        events.append('unlock')
    @contextmanager
    def scope(root, prefix, profile, owned_lock):
        assert owned_lock is lock and prefix == 'performance.'
        events.append('prepare')
        try:
            yield object()
        finally:
            events.append('restore')
    plan = {'root': str(tmp_path), 'manifest': str(tmp_path/'manifest.json'),
            'requests': [{'frozen': 'request'}], 'lut_dir': '/sealed'}
    def session(plan, name, manifest, **kwargs):
        assert events == ['lock', 'prepare']
        assert name == 'run-performance' and kwargs['mode'] == 'performance'
        assert kwargs['qualification'] == tmp_path/'qualification.candidate.json'
        assert kwargs['lock'] is lock
        events.append('performance')
        return {'mode': 'performance', 'provenance': {'formal_performance_qualified': True}}
    def audit(path, **kwargs):
        assert kwargs['require_formal_warmup'] is True
        assert kwargs['expected_requests'] == plan['requests']
        events.append('audit')
        return {'status': 'passed'}
    monkeypatch.setattr(device_module, 'HardwareLock', locked)
    monkeypatch.setattr(lut_module, 'MeasuredLUTRepository', lambda path: object())
    monkeypatch.setattr(driver, 'hardware_scope', scope)
    monkeypatch.setattr(driver, 'run_session', session)
    monkeypatch.setattr(driver, 'audit_session', audit)
    driver.validate_performance(plan, {'trusted': 'scope'}, object())
    assert events == ['lock', 'prepare', 'performance', 'audit', 'restore', 'unlock']
    assert driver.read_json(tmp_path/'performance-proof.json')['status'] == 'passed'
