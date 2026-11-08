#!/usr/bin/env python3
"""Freeze, execute and audit M4 qualification; never run the formal evaluation.

prepare runs CPU and opt-in correctness tests without administrator access.
execute needs explicit administrator invocation for temporary hardware controls;
all settings are captured and restored under the existing common hardware lock.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASELINE_HELPERS = {
    'scripts/k2_decode_lut/backend.py': '2702735c01b64bd84f146a8b042b96960aa7540ada52b98208caf7421295c859',
    'scripts/p6/cache.py': 'f0de50eae45cf204f63dd29837ba5b7dd33f2b21b0dc9544473e4c8c5258f24f',
}
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tools'))

from kvolt_runtime.qualification.contracts import (
    REQUIRED_CHECKS, SCHEMA, canonical_json, file_sha256, qualification_binding,
    strict_json, value_sha256,
)
from kvolt_runtime.qualification.audit import (
    artifact, audit_device, audit_native, audit_restoration, audit_session,
    audit_warmup, read_json, require,
)


def progress(message):
    # A closed progress channel must never prevent finally restoration.
    try:
        print('M4: '+message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not any(part.is_symlink() for part in path.parent.parents)
            and not path.parent.is_symlink(), 'unsafe M4 output directory')
    temporary = path.with_name('.'+path.name+'.'+secrets.token_hex(16)+'.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(canonical_json(value)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hashes():
    paths = [ROOT/'pyproject.toml']
    for folder in ('src', 'tests', 'tools'):
        paths.extend(sorted((ROOT/folder).rglob('*.py')))
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def clean_source():
    require(not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--',
                                         'src', 'tests', 'tools', 'pyproject.toml'], text=True).strip(),
            'commit reviewed source/tests/tools before freezing an M4 plan')
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def freeze_manifest(source, destination, *, full_caps=False):
    document = read_json(source)
    destination.mkdir()
    requests = []
    for index, row in enumerate(document['requests']):
        row = dict(row)
        original = Path(source).parent/row['prompt_file']
        target = destination/f'{index:03d}.txt'
        target.write_bytes(original.read_bytes())
        row['prompt_file'] = target.name
        requests.append(row)
    if full_caps:
        # Preserve every diagnostic and add a maximum-cap variant of every
        # distinct prompt. Tokenization below verifies all three lengths.
        seen = set()
        for row in list(requests):
            digest = file_sha256(destination/row['prompt_file'])
            if digest in seen:
                continue
            seen.add(digest)
            if row.get('nmax') != 512:
                requests.append({**row, 'request_id': 'm4-cap512-'+str(len(seen)), 'nmax': 512})
    path = destination/'manifest.json'
    write(path, {'schema_version': 'kvolt-request-manifest-v1', 'requests': requests})
    return path


def command(command, log, *, env=None):
    started = time.monotonic_ns()
    with Path(log).open('x') as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    result = {'command': list(map(str, command)), 'exit_code': completed.returncode,
              'started_ns': started, 'ended_ns': time.monotonic_ns(), 'log_sha256': file_sha256(log)}
    write(str(log)+'.result.json', result)
    require(completed.returncode == 0, 'M4 command failed; preserved log: '+str(log))
    return result


def junit(path, *, cpu_suite=False):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    failed = sum(bool(list(case.iter('failure')) or list(case.iter('error'))) for case in cases)
    skipped = sum(bool(list(case.iter('skipped'))) for case in cases)
    require(cases and failed == 0, 'CPU/GPU correctness tests failed')
    if cpu_suite:
        expected = {'tests.integration.test_cache_equivalence_gpu': 101,
                    'tests.integration.test_backend_prefill_gpu': 1,
                    'tests.integration.test_greedy_equivalence_gpu': 1}
        observed = Counter(case.get('classname', '') for case in cases if list(case.iter('skipped')))
        require(observed == expected, 'CPU suite must skip only the frozen 103 GPU checks')
    return {'run': len(cases), 'passed': len(cases)-skipped, 'failed': failed, 'skipped': skipped}


def gpu_environment(environment, args, root):
    return dict(environment, KVOLT_RUN_GPU='1', KVOLT_RUN_B1_GPU='1', KVOLT_NATIVE_PROFILE='agx2',
                KVOLT_HARDWARE_LOCK='/run/lock/kvolt-p3-hardware.lock',
                KVOLT_BASELINE=str(args.baseline_root.resolve()),
                KVOLT_KIVI_ROOT=str(args.kivi_root.resolve()), KVOLT_SNAPSHOT=str(args.model_dir.resolve()),
                KVOLT_MODEL_SNAPSHOT=str(args.model_dir.resolve()),
                KVOLT_M1_COHORT=str(root/'cpu/m1-cohort.json'), KVOLT_B1_REPORT=str(root/'cpu/b1-prefill-report.json'))


def cpu_environment(lut_dir):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', KVOLT_LUT_BUNDLE=str(lut_dir.resolve()))
    for name in ('KVOLT_RUN_GPU', 'KVOLT_RUN_B1_GPU'):
        environment.pop(name, None)
    return environment


def lut_source(lut_dir, repo):
    """Freeze exact source identity, including selector and separate targets."""
    return {'lut_dir': str(Path(lut_dir).resolve()), 'provenance': asdict(repo.provenance)}


def validate_lut_sources(plan):
    from kvolt_runtime.policy.compact import load_lut_repository
    from kvolt_runtime.policy.lut import MeasuredLUTRepository
    root = Path(plan['root'])
    runtime_file, cpu_file = 'lut-provenance.json', 'cpu/lut-provenance.json'
    frozen = plan['frozen_files']
    # Historical sealed plans retain their exact v1 schema. Current/compact
    # plans need identities captured before correctness tests, not at execute.
    lut_dir = Path(plan['lut_dir'])
    compact = any((lut_dir/name).exists() for name in ('selection.json', 'manifest.json'))
    if runtime_file not in frozen and cpu_file not in frozen:
        require(not compact, 'current/compact M4 plan requires frozen LUT provenance')
        return
    require(runtime_file in frozen and cpu_file in frozen, 'missing frozen runtime/CPU LUT provenance')
    runtime = read_json(root/runtime_file)
    require(canonical_json(runtime) == canonical_json(lut_source(lut_dir, load_lut_repository(lut_dir))),
            'runtime LUT source changed after preparation')
    cpu = read_json(root/cpu_file)
    require(isinstance(cpu, dict) and set(cpu) == {'lut_dir', 'provenance'}
            and isinstance(cpu['lut_dir'], str), 'invalid frozen CPU LUT provenance')
    require(canonical_json(cpu) == canonical_json(lut_source(cpu['lut_dir'], MeasuredLUTRepository(cpu['lut_dir']))),
            'CPU LUT fixture changed after preparation')


def baseline_sources(root):
    """Identify the independent reference helpers without inferring KIVI layout."""
    root = Path(root).resolve()
    require(root.is_dir(), 'explicit baseline root is missing')
    for relative, expected in BASELINE_HELPERS.items():
        require(file_sha256(root/relative) == expected, 'pinned reference helper changed: '+relative)
    return {'root': str(root), 'files_sha256': dict(BASELINE_HELPERS)}


def prepare(args):
    from kvolt_runtime.run_cli import _tokenizer, _device_profile, _power_profile
    from kvolt_runtime.policy.compact import load_lut_repository
    from kvolt_runtime.policy.lut import MeasuredLUTRepository
    from kvolt_runtime.execution.inputs import preflight_inputs
    from kvolt_runtime.recording import request_record
    require(os.geteuid() != 0, 'prepare must run as the ordinary project user')
    commit = clean_source()
    root = args.output_dir.resolve()
    root.mkdir()
    profile_path, power_path = root/'device-profile.json', root/'power-profile.json'
    profile_path.write_bytes(args.device_profile.read_bytes())
    power_path.write_bytes(args.power_profile.read_bytes())
    manifest = freeze_manifest(args.manifest.resolve(), root/'cohort', full_caps=True)
    eos_manifest = freeze_manifest(args.eos_manifest.resolve(), root/'eos-cohort')
    repo = load_lut_repository(args.lut_dir)
    cpu_lut_dir = getattr(args, 'cpu_lut_dir', None) or args.lut_dir
    cpu_repo = MeasuredLUTRepository(cpu_lut_dir)
    write(root/'lut-provenance.json', lut_source(args.lut_dir, repo))
    tokenizer, _ = _tokenizer(args.model_dir)
    prepared = preflight_inputs(manifest=manifest, prompt_file=None, tokenizer=tokenizer, repo=repo)
    eos = preflight_inputs(manifest=eos_manifest, prompt_file=None, tokenizer=tokenizer, repo=repo)
    coverage = {}
    for item in (*prepared.requests, *eos.requests):
        key = str(item.spec.length)
        coverage[key] = max(coverage.get(key, 0), item.spec.nmax)
    require(coverage == {'1024': 512, '4096': 512, '8192': 512}, 'M4 requires full caps for all three cohort lengths')
    require(len(eos.requests) == 8 and all(item.spec.length == 1024 for item in eos.requests), 'M4 requires eight frozen first-EOS contexts')
    _device_profile(profile_path)
    _power_profile(power_path)
    (root/'cpu').mkdir()
    write(root/'cpu/lut-provenance.json', lut_source(cpu_lut_dir, cpu_repo))
    reference = baseline_sources(args.baseline_root)
    write(root/'cpu/reference-helpers.json', reference)
    m1 = root/'cpu/m1-cohort.json'
    m1.write_bytes(args.m1_cohort.read_bytes())
    require(file_sha256(m1) == 'e9f0c05e0d21e13356f6a64d1176b03bcff613c708b0b0fd72228f711aadc20d', 'pinned M1 correctness cohort changed')
    environment = cpu_environment(cpu_lut_dir)
    command([sys.executable, '-m', 'pytest', '-q', '--junitxml='+str(root/'cpu/tests.xml')],
            root/'cpu/pytest.log', env=environment)
    cpu = junit(root/'cpu/tests.xml', cpu_suite=True)
    command([sys.executable, '-m', 'compileall', '-q', str(ROOT/'src'), str(ROOT/'tools')],
            root/'cpu/compile.log', env=environment)
    # GPU fixture owns the same nonblocking lock before importing CUDA. This
    # stage is numerical correctness only and makes no timing/energy claim.
    gpu_env = gpu_environment(environment, args, root)
    command([sys.executable, '-m', 'pytest', '-q', 'tests/integration/test_cache_equivalence_gpu.py',
             'tests/integration/test_backend_prefill_gpu.py', '--junitxml='+str(root/'cpu/gpu.xml')],
            root/'cpu/gpu.log', env=gpu_env)
    gpu = junit(root/'cpu/gpu.xml')
    require(gpu['run'] == gpu['passed'] == 102 and gpu['skipped'] == 0, 'all 102 GPU correctness checks must execute')
    plan = {'schema_version': 'kvolt-m4-plan-v1', 'source_commit': commit,
            'source_sha256': source_hashes(), 'root': str(root),
            'model_dir': str(args.model_dir.resolve()), 'kivi_root': str(args.kivi_root.resolve()),
            'baseline_root': reference['root'],
            'lut_dir': str(args.lut_dir.resolve()), 'device_profile': str(profile_path),
            'power_profile': str(power_path), 'manifest': str(manifest), 'eos_manifest': str(eos_manifest),
            'coverage': coverage, 'cpu': cpu, 'gpu': gpu,
            'requests': [request_record(item, index=index) for index, item in enumerate(prepared.requests)],
            'eos_requests': [request_record(item, index=index) for index, item in enumerate(eos.requests)],
            'formal_evaluation_run': False, 'prepared_ns': time.monotonic_ns()}
    plan['frozen_files'] = {str(path.relative_to(root)): file_sha256(path)
                            for path in sorted(root.rglob('*')) if path.is_file()}
    write(root/'plan.json', plan)
    digest = file_sha256(root/'plan.json')
    print(canonical_json({'status': 'prepared', 'plan': str(root/'plan.json'),
                          'plan_sha256': file_sha256(root/'plan.json'), 'cpu': cpu, 'gpu': gpu,
                          'next': [sys.executable, str(Path(__file__).resolve()), 'execute', str(root/'plan.json'), '--plan-sha256', digest]}))


def validate_plan(path, expected_digest):
    path = Path(os.path.abspath(path))
    require(isinstance(expected_digest, str) and re.fullmatch('[0-9a-f]{64}', expected_digest), 'explicit frozen plan SHA-256 required')
    artifact(path.parent, path.name)
    require(file_sha256(path) == expected_digest, 'frozen plan SHA-256 mismatch')
    plan = read_json(path)
    fields = {'schema_version', 'source_commit', 'source_sha256', 'root', 'model_dir', 'kivi_root', 'baseline_root',
              'lut_dir', 'device_profile', 'power_profile', 'manifest', 'eos_manifest', 'coverage',
              'cpu', 'gpu', 'requests', 'eos_requests', 'formal_evaluation_run', 'prepared_ns', 'frozen_files'}
    require(isinstance(plan, dict) and set(plan) == fields and plan['schema_version'] == 'kvolt-m4-plan-v1', 'unknown frozen plan schema')
    require(path == Path(plan['root'])/'plan.json' and path.parent == Path(plan['root']).resolve(), 'frozen plan root/path mismatch')
    require(plan['formal_evaluation_run'] is False, 'M4 plan must not run formal evaluation')
    require(clean_source() == plan['source_commit'] and source_hashes() == plan['source_sha256'], 'source changed after frozen plan')
    root = path.parent
    fixed = {'device_profile': 'device-profile.json', 'power_profile': 'power-profile.json',
             'manifest': 'cohort/manifest.json', 'eos_manifest': 'eos-cohort/manifest.json'}
    require(isinstance(plan['frozen_files'], dict) and plan['frozen_files'], 'missing frozen M4 files')
    for key, relative in fixed.items():
        require(plan[key] == str(root/relative) and relative in plan['frozen_files'], 'frozen M4 input path mismatch')
    for relative, digest in plan['frozen_files'].items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'invalid frozen file digest')
        require(file_sha256(artifact(root, relative)) == digest, 'frozen M4 evidence changed: '+relative)
    for relative in ('cpu/tests.xml', 'cpu/gpu.xml', 'cpu/pytest.log', 'cpu/gpu.log', 'cpu/m1-cohort.json', 'cpu/b1-prefill-report.json', 'cpu/reference-helpers.json'):
        require(relative in plan['frozen_files'], 'missing frozen correctness output')
    require(junit(root/'cpu/tests.xml', cpu_suite=True) == plan['cpu'] and junit(root/'cpu/gpu.xml') == plan['gpu'], 'recorded correctness counts changed')
    require(plan['cpu']['passed'] > 0 and plan['gpu']['run'] == plan['gpu']['passed'] == 102
            and plan['gpu']['skipped'] == 0, 'correctness evidence was not fully executed')
    require(plan['coverage'] == {'1024': 512, '4096': 512, '8192': 512}, 'M4 full-cap coverage mismatch')
    require(file_sha256(root/'cpu/m1-cohort.json') == 'e9f0c05e0d21e13356f6a64d1176b03bcff613c708b0b0fd72228f711aadc20d', 'pinned M1 correctness cohort changed')
    require(read_json(root/'cpu/reference-helpers.json') == baseline_sources(plan['baseline_root']),
            'independent baseline helpers changed after preparation')
    validate_lut_sources(plan)
    # Sysfs controls are trusted project inputs, not arbitrary root write targets.
    profile = read_json(root/'device-profile.json')
    patterns = {
        'gpu_path': r'/sys/devices/platform/(?:bus@0/)?17000000\.gpu/devfreq/17000000\.gpu',
        'emc_rate_path': r'/sys/kernel/debug/bpmp/debug/clk/emc/rate',
        'emc_lock_path': r'/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked',
        'emc_max_rate_path': r'/sys/kernel/debug/bpmp/debug/clk/emc/max_rate',
        'emc_cap_path': r'/sys/kernel/nvpmodel_clk_cap/emc',
        'emc_halt_path': r'/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt',
        'fan_path': r'/sys/class/hwmon/hwmon[0-9]+/pwm1',
        'fan_policy_path': r'/sys/class/thermal/thermal_zone[0-9]+/policy',
    }
    for key, pattern in patterns.items():
        require(isinstance(profile.get(key), str) and re.fullmatch(pattern, profile[key]), 'unsafe experiment control: '+key)
    require(all(re.fullmatch(r'/sys/devices/system/cpu/cpufreq/policy[0-9]+', value)
                for value in profile['cpu_policy_paths']), 'unsafe CPU experiment control')
    return plan


@contextmanager
def hardware_scope(root, prefix, profile, lock):
    """Restore both control groups independently, including capture/write errors."""
    from kvolt_runtime.device import DeviceController
    from kvolt_runtime.qualification.session import ExperimentPreparation
    device, preparation = DeviceController(profile, lock=lock), ExperimentPreparation(profile, lock=lock)
    original = device.capture()
    external = None
    primary = None
    try:
        progress(prefix+'external preparation starting')
        write(root/(prefix+'device.before.json'), original)
        external = preparation.capture()
        write(root/(prefix+'external.before.json'), external)
        write(root/(prefix+'external.prepare.json'), preparation.prepare())
        yield device
    except BaseException as error:
        primary = error
        raise
    finally:
        progress(prefix+'final restoration starting')
        errors = []
        def attempt(name, work):
            try:
                result = work()
                write(root/(prefix+name+'.json'), result)
                return result
            except BaseException as error:
                errors.append({'stage': name, 'type': type(error).__name__, 'message': str(error)})
                return None
        gpu = attempt('device.restore', lambda: device.restore(original))
        other = attempt('external.restore', preparation.restore)
        after_gpu = attempt('device.after', device.capture)
        after_external = attempt('external.after', preparation.capture) if external is not None else None
        try:
            require(gpu is not None and gpu['verified'] is True
                    and other is not None and other['verified'] is True, 'experiment restoration failed')
            audit_restoration(original, after_gpu)
            if external is not None:
                audit_restoration(external, after_external, external=True)
        except BaseException as error:
            errors.append({'stage': 'restoration_audit', 'type': type(error).__name__, 'message': str(error)})
        write(root/(prefix+'restoration-audit.json'), {'status': 'failed' if errors else 'passed', 'errors': errors,
              'primary_error': None if primary is None else {'type': type(primary).__name__, 'message': str(primary)}})
        progress(prefix+'final restoration '+('failed' if errors else 'passed'))
        if errors and primary is None:
            raise ValueError('M4 final restoration failed: '+canonical_json(errors))


def invalidate_reports(root):
    for name in ('qualification.candidate.json', 'qualification.json'):
        (Path(root)/name).unlink(missing_ok=True)


def run_session(plan, name, manifest, *, lock, mode='development', qualification=None, signal_number=None):
    progress(name+' session starting ('+mode+')')
    from kvolt_runtime.cli import _parser
    from kvolt_runtime.run_cli import RunFactories, run
    argv = ['run', '--manifest', str(manifest), '--model-dir', plan['model_dir'],
            '--lut-dir', plan['lut_dir'], '--output-dir', str(Path(plan['root'])/name),
            '--device-profile', plan['device_profile'], '--power-profile', plan['power_profile'],
            '--collect-power', '--native-profile', 'agx2', '--kivi-root', plan['kivi_root'],
            '--warmup', 'none' if signal_number else 'formal', '--mode', mode, '--no-online-lut']
    if qualification is not None:
        argv += ['--qualification', str(qualification)]
    parsed = _parser().parse_args(argv)
    parsed._original = [sys.executable, '-m', 'kvolt_runtime', *argv]
    factories = RunFactories()
    injection = None
    if signal_number:
        from kvolt_runtime.config import EOS_TOKEN_IDS
        factory = factories.backend
        injection = {'signal_number': int(signal_number), 'injected': False}
        def interrupted(args, clock):
            backend = factory(args, clock)
            original_prefill, original_decode = backend.prefill_raw, backend.decode_one
            current = {}
            def prefill(request):
                token = original_prefill(request)
                current.update(request=request, count=1)
                return token
            def decode(cache, previous):
                token = original_decode(cache, previous)
                current['count'] += 1
                request = current['request']
                # Preserve natural EOS/cap completion. A later request must
                # provide an actual continuing Decode boundary for this check.
                if (not injection['injected'] and token.token_id not in EOS_TOKEN_IDS
                        and current['count'] < request.nmax):
                    injection.update(injected=True, request_id=request.request_id,
                                     output_tokens=current['count'], nmax=request.nmax)
                    signal.raise_signal(signal_number)
                return token
            backend.prefill_raw, backend.decode_one = prefill, decode
            return backend
        factories = replace(factories, backend=interrupted)
    records = []
    code = run(parsed, lock=lock, factories=factories, emit=records.append)
    captured = {'argv': argv, 'exit_code': code, 'records': records}
    if injection is not None:
        captured['signal_injection'] = injection
    write(Path(plan['root'])/(name+'.stdout.json'), captured)
    if signal_number:
        require(injection['injected'], 'no continuing Decode boundary available for requested signal injection')
    require(code == (128+signal_number if signal_number else 0), 'actual CLI returned unexpected exit code')
    progress(name+' session ended with exit '+str(code))
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return read_json(Path(plan['root'])/name/'session.json')


def reference_comparison(plan, sessions, binding):
    progress('native exact comparison and repeat starting')
    from kvolt_runtime.backend.model import load_model
    from kvolt_runtime.config import RuntimeConfig
    from kvolt_runtime.contracts import RequestSpec
    from kvolt_runtime.qualification.reference import generate_reference
    loaded = load_model(Path(plan['model_dir']), RuntimeConfig(), kivi_root=Path(plan['kivi_root']),
                        raw_prefill=False, native_profile='agx2')
    proof = {'provenance': loaded.provenance, 'cases': [], 'status': 'failed'}
    try:
        audit_native(loaded.provenance, binding, raw_prefill=False)
        for session in sessions:
            for result in session['results']:
                request, generation = result['request'], result['generation']
                selection = result['decisions']['decode']['selected']
                group = selection['group'] if selection else 32
                spec = RequestSpec(request['request_id'], tuple(request['token_ids']), request['prompt_sha256'], request['nmax'])
                first = generate_reference(loaded, spec, group)
                second = generate_reference(loaded, spec, group)
                expected = {'token_ids': generation['token_ids'], 'stop_reason': generation['stop_reason'],
                            'decode_forwards': generation['decode_forwards'], 'final_cache_length': generation['final_cache_length']}
                row = {'request_id': request['request_id'], 'group': group, 'expected': expected,
                       'reference': asdict(first), 'reference_repeat': asdict(second)}
                proof['cases'].append(row)
                require(canonical_json(expected) == canonical_json(asdict(first)) == canonical_json(asdict(second)),
                        'native/reference repeat/runtime token or cache length mismatch')
        proof['status'] = 'passed'
        progress('native exact comparison and repeat passed')
    finally:
        write(Path(plan['root'])/'native-exact.json', proof)
        torch = loaded.torch
        loaded = None
        gc.collect()
        torch.cuda.empty_cache()
    return proof


def make_evidence(plan, binding, summaries):
    root = Path(plan['root'])
    report = {'schema_version': SCHEMA, 'binding': binding, 'coverage': plan['coverage'], 'checks': {}, 'artifacts': {}}
    # Include actual source/test commands and outputs. The check summaries are
    # computed by the auditor after assertions, never copied from the plan.
    raw = [path for path in sorted(root.rglob('*')) if path.is_file()
           and path.name not in ('qualification.json', 'qualification.candidate.json', 'execution-status.json') and 'checks' not in path.parts]
    relative = [str(path.relative_to(root)) for path in raw if path.stat().st_size]
    for path in raw:
        if path.stat().st_size:
            report['artifacts'][str(path.relative_to(root))] = file_sha256(path)
    for name in REQUIRED_CHECKS:
        require(name in summaries and summaries[name]['status'] == 'passed', 'missing passing M4 audit: '+name)
        check_path = 'checks/'+name+'.json'
        summary = {'schema_version': 'kvolt-qualification-check-v1', 'check': name,
                   'binding_sha256': value_sha256(binding), 'coverage': plan['coverage'],
                   'command': [sys.executable, str(Path(__file__).resolve()), 'execute', str(root/'plan.json'),
                               '--plan-sha256', file_sha256(root/'plan.json')],
                   'exit_code': 0, 'tests': {'run': 1, 'passed': 1, 'failed': 0, 'skipped': 0},
                   'artifacts': relative}
        write(root/check_path, summary)
        report['checks'][name] = {'status': 'passed', 'artifact': check_path}
        report['artifacts'][check_path] = file_sha256(root/check_path)
    write(root/'qualification.candidate.json', report)
    from kvolt_runtime.qualification.evidence import load_evidence
    return load_evidence(root/'qualification.candidate.json', binding)


def verify_frozen_requests(plan, repo):
    from kvolt_runtime.execution.inputs import preflight_inputs
    from kvolt_runtime.run_cli import _tokenizer
    from kvolt_runtime.recording import request_record
    tokenizer, _ = _tokenizer(Path(plan['model_dir']))
    coverage = {}
    for manifest_key, requests_key in (('manifest', 'requests'), ('eos_manifest', 'eos_requests')):
        prepared = preflight_inputs(manifest=Path(plan[manifest_key]), prompt_file=None, tokenizer=tokenizer, repo=repo)
        actual = [request_record(item, index=index) for index, item in enumerate(prepared.requests)]
        require(canonical_json(actual) == canonical_json(plan[requests_key]), 'frozen input/token IDs/cap/metadata changed')
        for item in prepared.requests:
            relative = str(item.prompt.path.relative_to(Path(plan['root'])))
            require(relative in plan['frozen_files'], 'unhashed frozen prompt')
            key = str(item.spec.length)
            coverage[key] = max(coverage.get(key, 0), item.spec.nmax)
    require(coverage == plan['coverage'] == {'1024': 512, '4096': 512, '8192': 512}, 'frozen actual coverage mismatch')
    require(len(plan['eos_requests']) == 8 and all(row['length'] == 1024 for row in plan['eos_requests']), 'first-EOS cohort mismatch')


def validate_performance(plan, binding, profile):
    """Exercise the candidate through the actual performance CLI and new lock."""
    from kvolt_runtime.device import HardwareLock
    from kvolt_runtime.policy.compact import load_lut_repository
    root = Path(plan['root'])
    with HardwareLock() as lock:
        with hardware_scope(root, 'performance.', profile, lock):
            session = run_session(plan, 'run-performance', plan['manifest'], lock=lock,
                                  mode='performance', qualification=root/'qualification.candidate.json')
            require(session['mode'] == 'performance' and session['provenance']['formal_performance_qualified'] is True,
                    'actual performance qualification gate was not exercised')
            proof = audit_session(root/'run-performance/session.json', require_formal_warmup=True,
                                  binding=binding, expected_requests=plan['requests'],
                                  repository=load_lut_repository(plan['lut_dir']))
            write(root/'performance-proof.json', proof)


def publish_evidence(plan, binding, summaries, profile):
    from kvolt_runtime.qualification.evidence import load_evidence
    root = Path(plan['root'])
    try:
        make_evidence(plan, binding, summaries)
        validate_performance(plan, binding, profile)
        require(clean_source() == plan['source_commit'] and source_hashes() == plan['source_sha256'],
                'source changed during M4 verification')
        # The immutable candidate bytes remain unchanged during performance
        # verification. Its proof has separate new artifacts outside this hash set.
        (root/'qualification.candidate.json').replace(root/'qualification.json')
        return load_evidence(root/'qualification.json', binding)
    except BaseException:
        invalidate_reports(root)
        raise


def execute(args):
    from kvolt_runtime.device import DeviceProfile, HardwareLock
    from kvolt_runtime.policy.compact import load_lut_repository
    from kvolt_runtime.telemetry import PowerProfile
    from kvolt_runtime.cli import _runtime_metadata
    from probe_device import qualify
    require(os.geteuid() == 0, 'execute requires explicit administrator invocation')
    plan = validate_plan(args.plan, args.plan_sha256)
    root = Path(plan['root'])
    require(not (root/'execution-status.json').exists(), 'M4 execution is single-attempt; freeze a new plan for another attempt')
    require(not any((root/name).exists() for name in ('qualification.json', 'qualification.candidate.json')), 'new M4 plan cannot contain an existing qualification report')
    status = {'status': 'running', 'started_ns': time.monotonic_ns(), 'errors': [], 'formal_evaluation_run': False}
    write(root/'execution-status.json', status)
    summaries = {}
    try:
        profile = DeviceProfile.from_json(plan['device_profile'])
        power = PowerProfile.from_json(plan['power_profile'])
        repo = load_lut_repository(plan['lut_dir'])
        verify_frozen_requests(plan, repo)
        binding = qualification_binding(repo=repo, device_profile=profile, power_profile=power,
                                        native_profile='agx2', runtime=_runtime_metadata())
        with HardwareLock() as lock:
            with hardware_scope(root, '', profile, lock) as device:
                progress('eight-frequency apply/no-op round trip starting')
                transitions = qualify(device)
                write(root/'device-transitions.json', transitions)
                summaries['device_control'] = audit_device(transitions)
                progress('eight-frequency apply/no-op round trip passed')
                sessions = []
                for name, manifest in (('run-first', plan['manifest']), ('run-repeat', plan['manifest']),
                                       ('run-first-eos', plan['eos_manifest'])):
                    sessions.append(run_session(plan, name, manifest, lock=lock))
                    summaries[name] = audit_session(root/name/'session.json', require_formal_warmup=True,
                        binding=binding, expected_requests=plan['eos_requests'] if name == 'run-first-eos' else plan['requests'], repository=repo)
                exact = reference_comparison(plan, (sessions[0], sessions[2]), binding)
                summaries['backend_exact'] = {'status': exact['status'], 'pairs': len(exact['cases'])}
                require(len(sessions[0]['results']) == len(sessions[1]['results']), 'repeat request count mismatch')
                for first, second in zip(sessions[0]['results'], sessions[1]['results']):
                    require(first['request'] == second['request'], 'repeat frozen request identity mismatch')
                    require(first['generation']['token_ids'] == second['generation']['token_ids']
                            and first['decisions']['decode']['selected'] == second['decisions']['decode']['selected'],
                            'identical-condition Runtime repeat did not reproduce tokens/selections')
                summaries['reproducibility'] = {'status': 'passed', 'requests': len(sessions[0]['results'])}
                eos_results = sessions[2]['results']
                require(all(row['generation']['n_output_tokens'] == 1
                            and row['generation']['stop_reason'] == 'eos' for row in eos_results),
                        'frozen natural first-EOS branch was not reached')
                for number, name in ((signal.SIGINT, 'interrupt'), (signal.SIGTERM, 'terminate')):
                    interrupted = run_session(plan, name, plan['manifest'], lock=lock, signal_number=number)
                    require(interrupted['status'] == 'interrupted' and interrupted['counts']['not_run'] > 0,
                            'signal did not stop remaining requests')
                    prefix = [row for row in interrupted['results'] if row['generation']['status'] == 'interrupted']
                    require(len(prefix) == 1 and prefix[0]['cleanup']['cache_released'] is True
                            and prefix[0]['cleanup']['gpu_restored'] is True
                            and prefix[0]['timing']['valid'] is False and prefix[0]['energy']['valid'] is False,
                            'signal lost prefix/cleanup/invalid measurement semantics')
                summaries['failure_recovery'] = {'status': 'passed', 'actual_signals': [2, 15], 'cpu_suite': plan['cpu']}
                summaries['cpu_policy'] = {'status': 'passed', 'cpu_suite': plan['cpu'], 'independent_policy_audits': True}
                summaries['data'] = {'status': 'passed', 'coverage': plan['coverage'], 'sealed_manifest_sha256': repo.provenance.manifest_sha256}
                summaries['timing_power'] = {'status': 'passed', 'sessions': [summaries[name] for name in ('run-first','run-repeat','run-first-eos')],
                                              'physical_sensor_response_validated': False}
                summaries['warmup'] = {'status': 'passed', 'ledgers': [audit_warmup(session) for session in sessions]}
                write(root/'audits.json', summaries)
                # Eligibility is still conditional on final external restoration.
        grant = publish_evidence(plan, binding, summaries, profile)
        status.update(status='qualified', qualification=grant.as_dict(), ended_ns=time.monotonic_ns())
    except BaseException as error:
        invalidate_reports(root)
        status.update(status='failed', ended_ns=time.monotonic_ns(), errors=[{'type': type(error).__name__, 'message': str(error)}])
        raise
    finally:
        try:
            write(root/'execution-status.json', status)
        except BaseException:
            invalidate_reports(root)
            raise
    print(canonical_json(status))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare_parser = sub.add_parser('prepare', help='freeze inputs and run CPU/GPU correctness without admin writes')
    for name in ('model-dir','lut-dir','device-profile','power-profile','kivi-root','baseline-root','manifest','eos-manifest','m1-cohort','output-dir'):
        prepare_parser.add_argument('--'+name, type=Path, required=True)
    prepare_parser.add_argument('--cpu-lut-dir', type=Path,
                                help='pinned sealed measured fixture for correctness tests (default: --lut-dir); separate from runtime LUT source')
    execute_parser = sub.add_parser('execute', help='explicit temporary device preparation and integrated M4 verification')
    execute_parser.add_argument('plan', type=Path)
    execute_parser.add_argument('--plan-sha256', required=True, help='SHA-256 emitted by the reviewed non-root prepare step')
    args = parser.parse_args(argv)
    (prepare if args.action == 'prepare' else execute)(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
