#!/usr/bin/env python3
"""Development-only latency attribution using an unchanged frozen cohort.

The wrappers add small host overhead. These observations are diagnostics, never
qualification evidence or a formal evaluation. Nested durations must not be
added to token-ready timing or interpreted as exclusive CUDA kernel time.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


class Trace:
    def __init__(self, clock=time.monotonic_ns):
        self.clock, self.rows, self.request_id = clock, [], None

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            row = {'component': name, 'request_id': self.request_id,
                   'start_ns': self.clock(), 'end_ns': None, 'status': 'failed'}
            try:
                result = function(*args, **kwargs)
                row['status'] = 'completed'
                return result
            finally:
                row['end_ns'] = self.clock()
                self.rows.append(row)
        return measured

    def backend(self, factory):
        def construct(args, clock):
            backend = factory(args, clock)
            original = backend.prefill_raw
            measured = self.wrap('prefill_raw', original)
            def prefill(request):
                self.request_id = request.request_id
                return measured(request)
            backend.prefill_raw = prefill
            for name in ('pack_initial', 'decode_one'):
                setattr(backend, name, self.wrap(name, getattr(backend, name)))
            return backend
        return construct

    @contextmanager
    def validators(self):
        # Both call sites hold their own reference to the same pure metadata
        # validator. Patch and restore each, retaining every original check.
        modules = [importlib.import_module('kvolt_runtime.backend.'+name)
                   for name in ('cache', 'decode')]
        originals = []
        try:
            for module in modules:
                original = module.validate_packed_cache
                originals.append((module, original))
                module.validate_packed_cache = self.wrap('validate_packed_cache', original)
            yield
        finally:
            for module, original in originals:
                module.validate_packed_cache = original


def source_commit(root):
    dirty = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--', 'src', 'tests', 'tools', 'pyproject.toml'],
        text=True)
    if dirty.strip():
        raise ValueError('commit reviewed source before development profiling')
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def runtime_modules(runtime):
    """Reject cached or redirected Runtime imports outside the selected tree."""
    package = runtime.resolve() / 'src/kvolt_runtime'
    paths = {}
    for name, module in list(sys.modules.items()):
        if name != 'kvolt_runtime' and not name.startswith('kvolt_runtime.'):
            continue
        filename = getattr(module, '__file__', None)
        path = Path(filename).resolve() if filename else None
        if path is None or not path.is_relative_to(package):
            raise ValueError('runtime module outside selected checkout: '+name)
        if name == 'kvolt_runtime' and path != package / '__init__.py':
            raise ValueError('runtime module outside selected checkout: '+name)
        paths[name] = str(path)
    return paths


def pin_runtime(runtime):
    """Load the package before helpers can prepend their own checkout's src."""
    sys.path.insert(0, str(runtime / 'src'))
    importlib.import_module('kvolt_runtime')
    runtime_modules(runtime)
    # These imports are CPU safe. Pin the actual execution entry points as well
    # as the package __path__, and reject preexisting mixed-checkout modules.
    for name in ('execution.runner', 'backend.model', 'cli'):
        importlib.import_module('kvolt_runtime.'+name)
    return runtime_modules(runtime)


def verify_session_source(session, commit):
    runtime = session.get('provenance', {}).get('runtime', {})
    if runtime.get('source_commit') != commit or runtime.get('source_worktree_dirty') is not False:
        raise ValueError('session Runtime source mismatch: requested '+commit+
                         ', observed '+str(runtime.get('source_commit'))+
                         ', dirty='+str(runtime.get('source_worktree_dirty')))
    return runtime


def validate_inputs(path, digest):
    """Bind diagnostic inputs to a reviewed plan, independently of its source."""
    from kvolt_runtime.qualification.audit import artifact, read_json, require
    from kvolt_runtime.qualification.contracts import file_sha256
    path = Path(os.path.abspath(path))
    artifact(path.parent, path.name)
    require(file_sha256(path) == digest, 'diagnostic plan SHA-256 mismatch')
    plan = read_json(path)
    require(plan['schema_version'] == 'kvolt-m4-plan-v1'
            and plan['formal_evaluation_run'] is False, 'expected non-evaluation M4 input plan')
    require(path == Path(plan['root'])/'plan.json'
            and path.parent == Path(plan['root']).resolve(), 'diagnostic plan root/path mismatch')
    fixed = {'device_profile': 'device-profile.json', 'power_profile': 'power-profile.json',
             'manifest': 'cohort/manifest.json', 'eos_manifest': 'eos-cohort/manifest.json'}
    for key, relative in fixed.items():
        require(plan[key] == str(path.parent/relative)
                and relative in plan['frozen_files'], 'diagnostic frozen input path mismatch')
    for relative, expected in plan['frozen_files'].items():
        require(file_sha256(artifact(path.parent, relative)) == expected,
                'diagnostic frozen input changed: '+relative)
    return plan


def main(argv=None):
    invocation = [sys.executable, str(Path(__file__).resolve()),
                  *(sys.argv[1:] if argv is None else argv)]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='existing frozen M4 plan; inputs only')
    parser.add_argument('--plan-sha256', required=True, help='reviewed frozen input-plan SHA-256')
    parser.add_argument('--runtime-root', type=Path, default=ROOT,
                        help='clean Git checkout to diagnose; imported in a fresh process')
    parser.add_argument('--output-dir', type=Path, required=True, help='new diagnostic directory')
    parser.add_argument('--repeats', type=int, choices=range(1, 4), default=1)
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        parser.error('explicit administrator invocation is required for temporary hardware preparation')
    runtime = args.runtime_root.resolve()
    commit, driver_commit = source_commit(runtime), source_commit(ROOT)
    # Source choice precedes every Runtime import, allowing the same diagnostic
    # driver to profile a baseline Git worktree without copying modified code.
    sys.path.insert(0, str(ROOT/'tools'))
    pin_runtime(runtime)
    from qualify_m4 import hardware_scope, progress, read_json, write
    from kvolt_runtime.cli import _parser
    from kvolt_runtime.device import DeviceProfile, HardwareLock
    from kvolt_runtime.qualification.contracts import file_sha256
    from kvolt_runtime.run_cli import RunFactories, run
    imported_runtime = runtime_modules(runtime)
    plan = validate_inputs(args.plan, args.plan_sha256)
    profile = DeviceProfile.from_json(plan['device_profile'])
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status = {'schema_version': 'kvolt-tpot-profile-v1', 'status': 'running',
              'formal_evaluation_run': False, 'formal_performance_qualified': False,
              'runtime_commit': commit, 'driver_commit': driver_commit,
              'runtime_modules': imported_runtime,
              'runtime_root': str(runtime), 'plan_sha256': file_sha256(args.plan),
              'plan_source_commit': plan['source_commit'], 'repeats': args.repeats,
              'instrumentation': 'host call wrappers; includes wrapper overhead; nested durations overlap',
              'sessions': [], 'errors': []}
    write(root/'profile-status.json', status)
    try:
        with HardwareLock() as lock:
            with hardware_scope(root, '', profile, lock):
                for index in range(args.repeats):
                    trace, records = Trace(), []
                    output = root/f'run-{index+1}'
                    argv = ['run', '--manifest', plan['manifest'], '--model-dir', plan['model_dir'],
                            '--lut-dir', plan['lut_dir'], '--output-dir', str(output),
                            '--device-profile', plan['device_profile'], '--power-profile', plan['power_profile'],
                            '--collect-power', '--native-profile', 'agx2', '--kivi-root', plan['kivi_root'],
                            '--warmup', 'formal', '--mode', 'development']
                    parsed = _parser().parse_args(argv)
                    parsed._original = invocation
                    factories = RunFactories()
                    factories = replace(factories, backend=trace.backend(factories.backend))
                    progress(f'diagnostic session {index+1}/{args.repeats} source {commit[:7]}')
                    try:
                        with trace.validators():
                            code = run(parsed, lock=lock, factories=factories, emit=records.append)
                    finally:
                        write(root/f'run-{index+1}.profile.json', {'rows': trace.rows})
                        write(root/f'run-{index+1}.stdout.json', {'records': records})
                    session = {'path': str(output), 'exit_code': code, 'source_verified': False}
                    status['sessions'].append(session)
                    if code != 0:
                        raise ValueError(f'diagnostic session failed with exit {code}')
                    actual = read_json(output/'session.json')
                    session['runtime'] = actual.get('provenance', {}).get('runtime')
                    verify_session_source(actual, commit)
                    session['source_verified'] = True
                    status['runtime_modules'] = runtime_modules(runtime)
            status['status'] = 'completed'
    except BaseException as error:
        status['status'] = 'failed'
        status['errors'].append({'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        write(root/'profile-status.json', status)
    progress('development profiling completed; restoration verified')


if __name__ == '__main__':
    main()
