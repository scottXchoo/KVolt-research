"""CPU tests for reviewed planning, control cleanup and report publication."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
from pathlib import Path
import signal
import sys
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
    import kvolt_runtime.policy.compact as lut_module
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
    monkeypatch.setattr(lut_module, 'load_lut_repository', lambda path: object())
    monkeypatch.setattr(driver, 'hardware_scope', scope)
    monkeypatch.setattr(driver, 'run_session', session)
    monkeypatch.setattr(driver, 'audit_session', audit)
    driver.validate_performance(plan, {'trusted': 'scope'}, object())
    assert events == ['lock', 'prepare', 'performance', 'audit', 'restore', 'unlock']
    assert driver.read_json(tmp_path/'performance-proof.json')['status'] == 'passed'


@pytest.mark.parametrize('signal_number', [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize('boundary', ['cap_two', 'first_eos', 'decode_eos', 'no_active_case', 'only_terminal_decode'])
def test_signal_injection_preserves_terminal_boundaries_and_targets_real_active_decode(
        driver, tmp_path, monkeypatch, signal_number, boundary):
    """Use the real Runner and real OS signal handlers with a CPU-only Backend."""
    import kvolt_runtime.run_cli as cli
    spec = importlib.util.spec_from_file_location('signal_runner_fixture', Path(__file__).with_name('test_session_runner.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    caps = [1,2,3,3] if boundary == 'cap_two' else [3,3,3]
    if boundary in ('no_active_case', 'only_terminal_decode'):
        caps = [1,2,3]
    runner, _, options, backend, events, emitted, _ = fixture.setup(tmp_path, requests=len(caps))
    fixture_manifest = {'schema_version': 'kvolt-request-manifest-v1', 'requests': [
        {'request_id': f'request-{i}', 'prompt_file': 'prompt.txt', 'nmax': cap} for i, cap in enumerate(caps)]}
    driver.write(tmp_path/'manifest.json', fixture_manifest)
    prepared = fixture.preflight_inputs(prompt_file=None, manifest=tmp_path/'manifest.json',
                                       tokenizer=fixture.Tokenizer(), repo=runner.repo)
    runner.recorder.save_preflight(prepared)
    current = {}
    prefill, decode = backend.prefill_raw, backend.decode_one
    def contextual_prefill(request):
        current['id'] = request.request_id
        backend.first = 128009 if ((boundary == 'first_eos' and request.request_id == 'request-0')
            or (boundary == 'no_active_case' and request.request_id == 'request-2')) else 17
        return prefill(request)
    def contextual_decode(cache, previous):
        token = decode(cache, previous)
        terminal = ((boundary == 'decode_eos' and current['id'] == 'request-0')
                    or (boundary == 'only_terminal_decode' and current['id'] == 'request-2'))
        return fixture.TokenResult(128009, token.ready_ns, token.cache) if terminal else token
    backend.prefill_raw, backend.decode_one = contextual_prefill, contextual_decode
    real_factories = cli.RunFactories
    monkeypatch.setattr(cli, 'RunFactories', lambda: real_factories(backend=lambda args, clock: backend))
    def run_real_runner(args, *, lock, factories, emit):
        assert factories.backend(args, fixture.Clock()) is backend
        return runner.run(prepared, options)['exit_code']
    monkeypatch.setattr(cli, 'run', run_real_runner)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    actual_raise, signals = signal.raise_signal, []
    def observed_raise(number):
        signals.append((number, current['id']))
        actual_raise(number)
    monkeypatch.setattr(driver.signal, 'raise_signal', observed_raise)
    previous_handler = signal.getsignal(signal_number)
    signal.signal(signal_number, lambda received, frame: runner.stop.request(received))
    plan = {'root': str(tmp_path), 'model_dir': '/unused-model', 'lut_dir': '/unused-lut',
            'device_profile': '/unused-device', 'power_profile': '/unused-power', 'kivi_root': '/unused-kivi'}
    try:
        if boundary in ('no_active_case', 'only_terminal_decode'):
            with pytest.raises(ValueError, match='no continuing Decode boundary'):
                driver.run_session(plan, 'out', tmp_path/'manifest.json', lock=object(), signal_number=signal_number)
            assert signals == []
            session = driver.read_json(tmp_path/'out/session.json')
            assert session['status'] == 'completed'
            assert all(row['generation']['status'] == 'completed' for row in session['results'])
            captured = driver.read_json(tmp_path/'out.stdout.json')
            assert captured['exit_code'] == 0 and captured['signal_injection'] == {
                'signal_number': int(signal_number), 'injected': False}
        else:
            session = driver.run_session(plan, 'out', tmp_path/'manifest.json', lock=object(), signal_number=signal_number)
            target = 2 if boundary == 'cap_two' else 1
            assert signals == [(signal_number, f'request-{target}')]
            assert [row['generation']['status'] for row in session['results']] == (
                ['completed']*target+['interrupted']+['not_run']*(len(caps)-target-1))
            prefix = session['results'][target]
            assert prefix['generation']['token_ids'] == [17,23]
            assert prefix['generation']['initial_packs'] == prefix['generation']['decode_forwards'] == 1
            assert prefix['generation']['final_cache_length'] == 1025
            assert prefix['timing']['valid'] is False and prefix['energy']['valid'] is False
            assert prefix['cleanup']['cache_released'] is True and prefix['cleanup']['gpu_restored'] is True
            assert session['exit_code'] == 128+signal_number
            captured = driver.read_json(tmp_path/'out.stdout.json')
            assert captured['signal_injection'] == {'signal_number': int(signal_number), 'injected': True,
                'request_id': f'request-{target}', 'output_tokens': 2, 'nmax': 3}
            assert all(row['timing']['valid'] is True for row in session['results'][:target])
            if boundary in ('first_eos', 'decode_eos'):
                assert session['results'][0]['generation']['stop_reason'] == 'eos'
    finally:
        signal.signal(signal_number, previous_handler)


def current_lut_dir():
    return Path(__file__).resolve().parents[1]/'data/luts/current'


def test_prepare_current_runtime_uses_separate_frozen_cpu_fixture(driver, tmp_path, monkeypatch):
    """The legacy data suite must not mistake current's compact data for sealed data."""
    import kvolt_runtime.run_cli as cli
    import kvolt_runtime.execution.inputs as inputs
    import kvolt_runtime.recording as recording
    import kvolt_runtime.policy.lut as lut
    from kvolt_runtime.policy.types import LUTProvenance
    output, sealed = tmp_path/'attempt', tmp_path/'historical-sealed'
    source = tmp_path/'input.json'
    source.write_text('{}')
    args = SimpleNamespace(output_dir=output, model_dir=tmp_path/'model', lut_dir=current_lut_dir(),
                           cpu_lut_dir=sealed, device_profile=source, power_profile=source,
                           manifest=source, eos_manifest=source, m1_cohort=source,
                           baseline_root=tmp_path/'baseline', kivi_root=tmp_path/'kivi')
    cpu_repo = SimpleNamespace(provenance=LUTProvenance(kind='measured', source_device='AGX2',
        manifest_sha256='a'*64, files_sha256=(('fixture.json', 'b'*64),)))
    def sealed_repository(path):
        assert Path(path) == sealed
        return cpu_repo
    monkeypatch.setattr(lut, 'MeasuredLUTRepository', sealed_repository)
    monkeypatch.setattr(driver.os, 'geteuid', lambda: 501)
    monkeypatch.setattr(driver, 'clean_source', lambda: 'reviewed')
    monkeypatch.setattr(driver, 'source_hashes', lambda: {'source': 'c'*64})
    monkeypatch.setattr(driver, 'baseline_sources', lambda root: {'root': str(root), 'files_sha256': {}})
    monkeypatch.setattr(cli, '_tokenizer', lambda path: (object(), {}))
    monkeypatch.setattr(cli, '_device_profile', lambda path: object())
    monkeypatch.setattr(cli, '_power_profile', lambda path: object())
    def freeze(source, destination, **kwargs):
        driver.write(destination/'manifest.json', {'requests': []})
        return destination/'manifest.json'
    monkeypatch.setattr(driver, 'freeze_manifest', freeze)
    def preflight(*, manifest, **kwargs):
        assert kwargs['repo'].provenance.manifest_sha256 == 'b421fcd10ecb57df7d6962aa9a5f48e264e90defc0c76a3544218d31685f783b'
        lengths = [1024]*8 if manifest.parent.name == 'eos-cohort' else [1024, 4096, 8192]
        return SimpleNamespace(requests=[SimpleNamespace(spec=SimpleNamespace(length=n, nmax=512)) for n in lengths])
    monkeypatch.setattr(inputs, 'preflight_inputs', preflight)
    monkeypatch.setattr(recording, 'request_record', lambda item, index: {'length': item.spec.length})
    original_sha = driver.file_sha256
    monkeypatch.setattr(driver, 'file_sha256', lambda path: 'e9f0c05e0d21e13356f6a64d1176b03bcff613c708b0b0fd72228f711aadc20d'
        if Path(path).name == 'm1-cohort.json' else original_sha(path))
    environments = []
    def command(argv, log, *, env):
        environments.append(env)
        log.write_text('simulated process output')
        for item in argv:
            if str(item).startswith('--junitxml='):
                path = Path(str(item).split('=', 1)[1])
                if path.name == 'tests.xml':
                    cases = '<testcase/>' + ''.join(
                        f'<testcase classname="tests.integration.{name}"><skipped/></testcase>'*count
                        for name, count in [('test_cache_equivalence_gpu', 101),
                                            ('test_backend_prefill_gpu', 1), ('test_greedy_equivalence_gpu', 1)])
                else:
                    cases = '<testcase/>'*102
                    driver.write(output/'cpu/b1-prefill-report.json', {'status': 'passed'})
                path.write_text('<testsuite>'+cases+'</testsuite>')
    monkeypatch.setattr(driver, 'command', command)
    driver.prepare(args)
    plan = driver.read_json(output/'plan.json')
    assert plan['lut_dir'] == str(current_lut_dir())
    assert all(env['KVOLT_LUT_BUNDLE'] == str(sealed) for env in environments)
    assert plan['cpu']['skipped'] == 103 and plan['gpu']['passed'] == 102 and plan['gpu']['skipped'] == 0
    cpu = driver.read_json(output/'cpu/lut-provenance.json')
    assert cpu['lut_dir'] == str(sealed) and cpu['provenance']['manifest_sha256'] == 'a'*64
    runtime = driver.read_json(output/'lut-provenance.json')
    assert runtime['lut_dir'] == str(current_lut_dir())
    metadata = driver.strict_json(runtime['provenance']['metadata_json'])
    assert metadata['current_selection']['selection']['reference']['directory'] == 'agx2-reference-repeat10'
    assert {'lut-provenance.json', 'cpu/lut-provenance.json'} <= plan['frozen_files'].keys()


def test_execute_loads_current_reference_identity_before_hardware(driver, tmp_path, monkeypatch):
    import kvolt_runtime.device as device
    import kvolt_runtime.telemetry as telemetry
    import kvolt_runtime.cli as cli
    from kvolt_runtime.policy.compact import load_lut_repository
    from kvolt_runtime.qualification.contracts import _lut_binding
    expected = _lut_binding(load_lut_repository(current_lut_dir()).provenance)
    plan = {'root': str(tmp_path), 'lut_dir': str(current_lut_dir()),
            'device_profile': 'device.json', 'power_profile': 'power.json'}
    monkeypatch.setattr(driver.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(driver, 'validate_plan', lambda *args: plan)
    monkeypatch.setattr(device.DeviceProfile, 'from_json', lambda path: object())
    monkeypatch.setattr(telemetry.PowerProfile, 'from_json', lambda path: object())
    monkeypatch.setattr(driver, 'verify_frozen_requests', lambda plan, repo: None)
    monkeypatch.setattr(cli, '_runtime_metadata', lambda: {})
    class BoundBeforeHardware(Exception): pass
    def binding(**kwargs):
        assert _lut_binding(kwargs['repo'].provenance) == expected
        assert 'reference' in expected
        raise BoundBeforeHardware
    monkeypatch.setattr(driver, 'qualification_binding', binding)
    monkeypatch.setattr(device, 'HardwareLock', lambda: pytest.fail('hardware accessed before binding'))
    with pytest.raises(BoundBeforeHardware):
        driver.execute(SimpleNamespace(plan=tmp_path/'plan.json', plan_sha256='d'*64))


def test_performance_audit_loads_same_current_identity_as_run(driver, tmp_path, monkeypatch):
    import kvolt_runtime.device as device
    from kvolt_runtime.policy.compact import load_lut_repository
    from kvolt_runtime.qualification.contracts import _lut_binding
    expected = _lut_binding(load_lut_repository(current_lut_dir()).provenance)
    plan = {'root': str(tmp_path), 'lut_dir': str(current_lut_dir()), 'manifest': 'manifest.json', 'requests': []}
    @contextmanager
    def scope(*args):
        yield object()
    monkeypatch.setattr(device, 'HardwareLock', scope)
    monkeypatch.setattr(driver, 'hardware_scope', scope)
    monkeypatch.setattr(driver, 'run_session', lambda *args, **kwargs: {
        'mode': 'performance', 'provenance': {'formal_performance_qualified': True}})
    def audit(path, **kwargs):
        assert _lut_binding(kwargs['repository'].provenance) == expected
        return {'status': 'passed'}
    monkeypatch.setattr(driver, 'audit_session', audit)
    driver.validate_performance(plan, {'lut': expected}, object())
    assert driver.read_json(tmp_path/'performance-proof.json')['status'] == 'passed'


@pytest.mark.parametrize('mutation', [None, 'runtime_changed', 'cpu_changed', 'missing_runtime',
                                    'missing_cpu', 'cpu_unknown_key', 'cpu_path'])
def test_frozen_plan_binds_current_and_cpu_sources(driver, plan_bundle, monkeypatch, mutation):
    import kvolt_runtime.policy.lut as lut
    from kvolt_runtime.policy.compact import load_lut_repository
    from kvolt_runtime.policy.types import LUTProvenance
    bundle = plan_bundle
    bundle.plan['lut_dir'] = str(current_lut_dir())
    runtime = driver.lut_source(current_lut_dir(), load_lut_repository(current_lut_dir()))
    cpu_repo = SimpleNamespace(provenance=LUTProvenance(kind='measured', source_device='AGX2',
        manifest_sha256='a'*64, files_sha256=(('data.json', 'b'*64),)))
    cpu_dir = bundle.root/'sealed'
    def sealed_repository(path):
        assert Path(path) == cpu_dir
        return cpu_repo
    monkeypatch.setattr(lut, 'MeasuredLUTRepository', sealed_repository)
    cpu = driver.lut_source(cpu_dir, cpu_repo)
    if mutation == 'runtime_changed':
        # A different selector/target must fail even with correctly rehashed
        # local evidence and an unchanged runtime prediction manifest.
        metadata = driver.strict_json(runtime['provenance']['metadata_json'])
        metadata['target_repository']['manifest_sha256'] = 'f'*64
        runtime['provenance']['metadata_json'] = driver.canonical_json(metadata)
    elif mutation == 'cpu_changed':
        cpu['provenance']['files_sha256'] = [('data.json', 'c'*64)]
    elif mutation == 'cpu_unknown_key':
        cpu['runtime_override'] = str(cpu_dir)
    elif mutation == 'cpu_path':
        cpu['lut_dir'] = None
    if mutation != 'missing_runtime':
        driver.write(bundle.root/'lut-provenance.json', runtime)
    if mutation != 'missing_cpu':
        driver.write(bundle.root/'cpu/lut-provenance.json', cpu)
    digest = bundle.save()
    if mutation is None:
        assert driver.validate_plan(bundle.path, digest) == bundle.plan
    else:
        with pytest.raises(ValueError, match='LUT'):
            driver.validate_plan(bundle.path, digest)


def test_current_plan_cannot_omit_both_frozen_sources(driver, plan_bundle):
    bundle = plan_bundle
    bundle.plan['lut_dir'] = str(current_lut_dir())
    with pytest.raises(ValueError, match='requires frozen LUT provenance'):
        driver.validate_plan(bundle.path, bundle.save())


@pytest.mark.parametrize('cpu_option', [[], ['--cpu-lut-dir', '/sealed-cpu']])
def test_prepare_cpu_fixture_cli_is_optional_and_never_replaces_runtime(driver, monkeypatch, cpu_option):
    seen = []
    monkeypatch.setattr(driver, 'prepare', seen.append)
    required = ['model-dir', 'lut-dir', 'device-profile', 'power-profile', 'kivi-root',
                'baseline-root', 'manifest', 'eos-manifest', 'm1-cohort', 'output-dir']
    argv = ['prepare'] + [item for name in required for item in ['--'+name, '/'+name]] + cpu_option
    assert driver.main(argv) == 0
    assert seen[0].lut_dir == Path('/lut-dir')
    assert seen[0].cpu_lut_dir == (Path('/sealed-cpu') if cpu_option else None)
