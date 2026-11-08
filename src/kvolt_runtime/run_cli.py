"""Standard M3 execution CLI; CPU preflight precedes exclusive GPU ownership."""
import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
from importlib import metadata
import os
import math
from pathlib import Path
import signal
import sys
import time
import uuid

from .config import MISSING, RuntimeConfig, SNAPSHOT
from .execution.contracts import ExecutionError, PreparationError, SessionOptions, StopController, error_record
from .recording import SessionRecorder, json_line, to_json_native

SCENARIOS = ('baseline', 'flat_quality', 'nonmonotonic', 'exact_ties')
TOKENIZERS_VERSION = '0.19.1'


class RunArgumentError(ValueError):
    pass


class RunParser(argparse.ArgumentParser):
    def error(self, message):
        raise RunArgumentError(message)


def configure_parser(parser):
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prompt-file', type=Path, help='UTF-8 prompt; no template or truncation')
    inputs.add_argument('--manifest', type=Path, help='kvolt-request-manifest-v1 JSON')
    parser.add_argument('--model-dir', type=Path, required=True, help='fixed pinned model snapshot directory')
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument('--lut-dir', type=Path,
                        help='override data/luts/current/selection.json with a selection directory, sealed measured bundle or pinned compact LUT; no synthetic fallback')
    source.add_argument('--mock', choices=SCENARIOS, help='explicit synthetic LUT; development only')
    parser.add_argument('--reference-lut-dir', type=Path,
                        help='FP16 targets for explicit compact --lut-dir; omit all source options to use data/luts/current/selection.json')
    parser.add_argument('--output-dir', type=Path, required=True, help='new directory; existing paths are refused')
    parser.add_argument('--max-new-tokens', type=int, default=MISSING, help='output cap 1..512; item > CLI > 512')
    parser.add_argument('--mode', choices=('development', 'performance'), default='development')
    parser.add_argument('--collect-power', action='store_true')
    parser.add_argument('--device-profile', type=Path, required=True, help='explicit device preparation/verification JSON')
    parser.add_argument('--power-profile', type=Path, help='explicit INA rails/timings JSON; required with --collect-power')
    parser.add_argument('--kivi-root', type=Path, default=Path('/home/agx3/kvolt/third_party/KIVI'))
    parser.add_argument('--native-profile', choices=('agx1', 'agx2'), default='agx1')
    parser.add_argument('--warmup', choices=('basic', 'none', 'formal'), default='basic',
                        help='basic development or full natural-generation formal coverage (default: basic)')
    parser.add_argument('--qualification', type=Path,
                        help='M4 evidence report matching the current source, device and measurement profiles')
    parser.add_argument('--online-lut-state', type=Path,
                        help='reuse compatible adaptive state; default: output-dir/online-lut.json')
    parser.add_argument('--no-online-lut', action='store_true',
                        help='use the frozen source LUT without request feedback')
    parser.add_argument('--online-lut-alpha', type=float, default=None,
                        help='EMA weight in (0,1], default 0.1; development only')
    return parser


def _repository(args):
    reference = getattr(args, 'reference_lut_dir', None)
    if args.lut_dir is not None:
        from .policy.compact import load_lut_repository
        return load_lut_repository(args.lut_dir, reference_lut_dir=reference)
    if reference is not None:
        raise ValueError('--reference-lut-dir requires --lut-dir; cannot combine with --mock')
    if args.mock is None:
        from .policy.selection import load_current_lut_repository
        return load_current_lut_repository()
    from .policy.mock import MockLUTRepository
    return MockLUTRepository(args.mock)


def _tokenizer(model_dir):
    """Check small pinned files and use Rust tokenizers without torch/HF imports."""
    from .backend.provenance import MODEL_FILES, checked_file
    model_dir = Path(model_dir).resolve()
    if model_dir.name != SNAPSHOT:
        raise ValueError('model directory must name the fixed snapshot revision')
    files = {name: checked_file(model_dir / name, MODEL_FILES[name])
             for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json', 'generation_config.json')}
    try:
        version = metadata.version('tokenizers')
    except metadata.PackageNotFoundError as error:
        raise ImportError('tokenizers==0.19.1 is required for CPU input preflight') from error
    if version != TOKENIZERS_VERSION:
        raise ImportError(f'tokenizers: expected {TOKENIZERS_VERSION}, got {version}')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    # The snapshot normally has neither enabled; explicitly disable both so no
    # persisted tokenizer setting can truncate/pad the frozen input IDs.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    def encode(text, *, add_special_tokens=False):
        if add_special_tokens is not False:
            raise ValueError('standard run forbids automatically added special tokens')
        return {'input_ids': tokenizer.encode(text, add_special_tokens=False).ids}
    return encode, {'implementation': 'tokenizers.Tokenizer.from_file',
                    'version': version, 'required_version': TOKENIZERS_VERSION,
                    'snapshot': SNAPSHOT, 'files': files,
                    'add_special_tokens': False, 'truncation': False, 'padding': False}


def _device_profile(path):
    from .device import DeviceProfile
    return DeviceProfile.from_json(path)


def _power_profile(path):
    from .telemetry import PowerProfile
    return PowerProfile.from_json(path)


def _decode_controller(repo, prepared, clock):
    from .policy.decode import DecodeController
    if all(item.spec.nmax == 1 for item in prepared.requests):
        adapter = object()
    else:
        from .policy.knee import KneeAdapter
        adapter = KneeAdapter()
    return DecodeController(repo, knee_adapter=adapter, clock=clock)


def _hardware_lock():
    from .device import HardwareLock
    return HardwareLock()


def _backend(args, clock):
    # This function is invoked only while the common hardware lock is held.
    from .backend.model import InferenceBackend
    return InferenceBackend(kivi_root=args.kivi_root, native_profile=args.native_profile, clock=clock)


def _device(profile, lock, clock):
    from .device import DeviceController
    return DeviceController(profile, lock=lock, clock_ns=clock)


def _collector(profile, clock):
    from .telemetry import INAPowerCollector
    return INAPowerCollector(profile, clock=clock)


def _runner(**kwargs):
    from .execution.runner import SessionRunner
    return SessionRunner(**kwargs)


def _runtime_metadata():
    from .cli import _runtime_metadata as capture
    return capture()


def _identity(value):
    """Discard artifact locations and import-state noise from source identities.

    Hardware profiles do not use this filter: their sysfs paths and collection
    timing settings identify the measurement/control conditions themselves.
    """
    if isinstance(value, dict):
        return {key: _identity(item) for key, item in value.items()
                if key not in ('path', 'loaded_versions', 'source_worktree_dirty')
                and not key.endswith('_path')}
    if isinstance(value, (tuple, list)):
        return [_identity(item) for item in value]
    return value


def _online_context(args, provenance):
    from .backend.provenance import MODEL_FILES, NATIVE_BUILD_PROFILES, SOURCE_HASHES, VERSIONS
    installed = {}
    for package in sorted(set(VERSIONS) | {'tokenizers', 'numpy', 'scipy', 'kneed'}):
        try:
            installed[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed[package] = None
    return {'model': {'snapshot': SNAPSHOT, 'files_sha256': dict(MODEL_FILES)},
            'input_tokenizer': _identity(provenance['input_tokenizer']),
            'native_profile': args.native_profile,
            'native_build': dict(NATIVE_BUILD_PROFILES[args.native_profile]),
            'backend_source_sha256': dict(SOURCE_HASHES),
            'device_profile': provenance['device_profile'],
            'power_profile': provenance.get('power_profile'),
            'collect_power': args.collect_power,
            'runtime': _identity(provenance['runtime']),
            'installed_distribution_versions': installed}


@dataclass(frozen=True)
class RunFactories:
    """Narrow construction seam for CPU tests and internal qualification harnesses.

    Production uses these defaults. Neither factories nor hardware-lock paths
    are CLI options. Injected Backends must still obey the ordinary contracts.
    """
    repository: object = _repository
    tokenizer: object = _tokenizer
    device_profile: object = _device_profile
    power_profile: object = _power_profile
    recorder: object = SessionRecorder
    decode_controller: object = _decode_controller
    hardware_lock: object = _hardware_lock
    backend: object = _backend
    device: object = _device
    collector: object = _collector
    runner: object = _runner
    runtime_metadata: object = _runtime_metadata
    clock: object = time.monotonic_ns


@contextmanager
def _signals(stop):
    previous = {}
    def handler(number, frame):
        stop.request(number)
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, handler)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@contextmanager
def _stdout_channel(emit):
    """Keep Python and native loader chatter on stderr; JSONL owns stdout."""
    stream = sys.stdout
    saved_fd = None
    target, private_target = stream, None
    try:
        stdout_fd, stderr_fd = stream.fileno(), sys.stderr.fileno()
        saved_fd = os.dup(stdout_fd)
        if emit is None:
            target = os.fdopen(os.dup(saved_fd), 'w', encoding='utf-8', buffering=1)
            private_target = target
        stream.flush()
        os.dup2(stderr_fd, stdout_fd)
    except (AttributeError, OSError, ValueError):
        if saved_fd is not None:
            os.close(saved_fd)
            saved_fd = None
        if private_target is not None:
            private_target.close()
            private_target = None
        target = stream
    if emit is None:
        def emit(record):
            target.write(json_line(record))
            target.flush()
    try:
        with redirect_stdout(sys.stderr):
            yield emit
    finally:
        if saved_fd is not None:
            try:
                sys.stdout.flush()
                # Extensions may use buffered libc stdout instead of Python.
                import ctypes
                ctypes.CDLL(None).fflush(None)
            finally:
                try:
                    os.dup2(saved_fd, stdout_fd)
                finally:
                    os.close(saved_fd)
                    if private_target is not None:
                        private_target.close()


def _resolved(args):
    return {key: (None if value is MISSING else
                  os.path.abspath(value) if isinstance(value, Path) else
                  str(value) if type(value) is float and not math.isfinite(value) else to_json_native(value))
            for key, value in vars(args).items() if not key.startswith('_')}


def _failure(error, stage, code, *, recorder=None, provenance=None, session_id=None):
    provenance = dict(provenance or {})
    if 'formal_performance_qualified' in provenance:
        provenance['formal_performance_qualified'] = False
    session = {'schema_version': 'kvolt-session-v1', 'session_id': session_id or uuid.uuid4().hex,
               'status': 'rejected' if code == 2 else ('interrupted' if code >= 128 else 'failed'),
               'counts': dict.fromkeys(('generation_completed', 'generation_failed', 'generation_interrupted',
                                        'not_run', 'timing_valid', 'energy_valid', 'energy_not_collected',
                                        'request_cleanup_failed'), 0),
               'errors': [error_record(error, stage=stage)], 'exit_code': code,
               'provenance': provenance}
    if recorder is not None:
        try:
            recorder.save_session(session)
        except Exception as save_error:
            session['errors'].append(error_record(save_error, stage='session_save'))
            if code < 128:
                session.update(status='failed', exit_code=1)
    return session


def _summary(session, recorder):
    return {'schema_version': 'kvolt-stdout-v1', 'record_type': 'session_result',
            'session_id': session['session_id'], 'status': session['status'],
            'counts': session['counts'], 'errors': session['errors'], 'exit_code': session['exit_code'],
            'command': session.get('provenance', {}).get('command'),
            'artifact_root': str(recorder.output_dir) if recorder else None,
            'session_path': recorder.session_path if recorder else None}


def run(args, *, lock=None, factories=None, emit=None):
    """Run parsed standard arguments; a supplied internal lock stays caller-owned."""
    factories = factories or RunFactories()
    recorder = None
    stop = StopController()
    session_id = uuid.uuid4().hex
    provenance = {'command': {'original': getattr(args, '_original', None), 'resolved': _resolved(args)},
                  'effective_config': asdict(RuntimeConfig()), 'formal_performance_qualified': False}
    stage, failure_code = 'input', 2
    session = None
    unlock_errors = []
    with _stdout_channel(emit) as output:
        try:
            online_state = getattr(args, 'online_lut_state', None)
            online_alpha = getattr(args, 'online_lut_alpha', None)
            online_disabled = getattr(args, 'no_online_lut', False)
            if online_disabled and (online_state is not None or online_alpha is not None):
                raise ValueError('--no-online-lut conflicts with --online-lut-state/--online-lut-alpha')
            if online_alpha is not None and (not math.isfinite(online_alpha) or not 0 < online_alpha <= 1):
                raise ValueError('--online-lut-alpha must be finite and in (0,1]')
            if args.mode == 'performance' and (online_state is not None or online_alpha is not None):
                raise ValueError('adaptive LUT has no formal performance qualification; omit online state/alpha')
            online_enabled = args.mode == 'development' and not online_disabled
            provenance['online_lut'] = {'enabled': online_enabled,
                'policy': 'request_end_exact_length_ema' if online_enabled else 'frozen',
                'disabled_reason': None if online_enabled else
                    ('explicit_no_online_lut' if online_disabled else 'formal_performance_frozen')}
            if getattr(args, 'reference_lut_dir', None) is not None and args.mock is not None:
                raise ValueError('--reference-lut-dir requires --lut-dir; cannot combine with --mock')
            qualification_path = getattr(args, 'qualification', None)
            if args.mode == 'performance' and qualification_path is None:
                raise ExecutionError('M4_QUALIFICATION_REQUIRED', 'qualification',
                                     'M4 qualification_required: performance mode requires a verified evidence report')
            if args.mode == 'performance' and (args.mock is not None or args.native_profile != 'agx2'
                                              or not args.collect_power or args.warmup != 'formal'):
                raise ValueError('performance requires measured AGX2 LUTs, collected power and --warmup formal')
            if args.mode != 'performance' and qualification_path is not None:
                raise ValueError('--qualification is only used with --mode performance')
            if args.collect_power and args.power_profile is None:
                raise ValueError('--collect-power requires an explicit --power-profile')
            if args.power_profile is not None and not args.collect_power:
                raise ValueError('--power-profile requires --collect-power')
            stage = 'output'
            recorder = factories.recorder(args.output_dir)
            stage = 'lut'
            repo = factories.repository(args)
            expected_kind = 'synthetic' if args.mock is not None else 'measured'
            if repo.provenance.kind != expected_kind:
                raise ValueError('LUT provenance does not match the explicitly selected source')
            from .cli import _source_metadata
            provenance['lut'] = _source_metadata(repo.provenance)
            source_device = repo.provenance.source_device
            device_match = source_device.casefold() == args.native_profile.casefold() if expected_kind == 'measured' else None
            provenance['lut_device_conditions'] = {
                'source_device': source_device, 'requested_native_profile': args.native_profile,
                'device_match': device_match, 'development_device_mismatch': device_match is False,
                'formal_condition_match_qualified': False}
            stage = 'tokenizer'
            tokenizer, provenance['input_tokenizer'] = factories.tokenizer(args.model_dir)
            stage = 'input_preflight'
            from .execution.inputs import preflight_inputs
            prepared = preflight_inputs(prompt_file=args.prompt_file, manifest=args.manifest,
                                        tokenizer=tokenizer, repo=repo, cli_nmax=args.max_new_tokens)
            stage = 'profile'
            profile = factories.device_profile(args.device_profile)
            provenance['device_profile'] = profile.as_dict()
            power = factories.power_profile(args.power_profile) if args.collect_power else None
            if power is not None:
                provenance['power_profile'] = power.to_dict()
            stage, failure_code = 'preflight_save', 1
            provenance['preflight_paths'] = recorder.save_preflight(prepared)
            stage = 'dependency'
            if online_enabled:
                provenance['runtime'] = factories.runtime_metadata()
                stage, failure_code = 'online_lut', 2
                from .policy.online import OnlineLUTRepository
                repo = OnlineLUTRepository(repo,
                    state_path=online_state if online_state is not None else recorder.output_dir / 'online-lut.json',
                    context=_online_context(args, provenance),
                    alpha=0.1 if online_alpha is None else online_alpha)
                provenance['online_lut']['initial'] = repo.snapshot()
                stage, failure_code = 'online_lut_initial_save', 1
                provenance['online_lut']['initial_path'] = recorder.atomic_json(
                    'online-lut.initial.json', repo.export_state())
            stage = 'dependency'
            dc = factories.decode_controller(repo, prepared, factories.clock)
            if not online_enabled:
                # Formal grants retain the original dependency-load/capture order.
                provenance['runtime'] = factories.runtime_metadata()
            qualification = None
            if args.mode == 'performance':
                stage, failure_code = 'qualification', 2
                from .qualification.contracts import qualification_binding
                from .qualification.evidence import load_evidence
                binding = qualification_binding(repo=repo, device_profile=profile, power_profile=power,
                                                native_profile=args.native_profile,
                                                runtime=provenance['runtime'])
                qualification = load_evidence(qualification_path, binding)
                qualification.assert_requests(prepared)
                provenance['qualification'] = qualification.as_dict()
                provenance['formal_evidence_accepted'] = True
                provenance['lut_device_conditions']['formal_condition_match_qualified'] = True
            options = SessionOptions(model_dir=args.model_dir, output_dir=args.output_dir,
                                     mode=args.mode, collect_power=args.collect_power,
                                     native_profile=args.native_profile)
            stage, failure_code = 'device_lock', 3
            owned_lock = lock is None
            active_lock = factories.hardware_lock() if owned_lock else lock
            entered = False
            try:
                if owned_lock:
                    active_lock.__enter__()
                    entered = True
                active_lock.assert_held()
                stage, failure_code = 'runtime_setup', 1
                backend = factories.backend(args, factories.clock)
                device = factories.device(profile, active_lock, factories.clock)
                def emit_request(record):
                    # Release ownership before publishing the terminal result.
                    if record.get('record_type') != 'session_result':
                        output(record)
                qualification_args = {'qualification': qualification} if qualification is not None else {}
                runner = factories.runner(backend=backend, repo=repo, device=device, recorder=recorder,
                                          decode_controller=dc, clock=factories.clock, stop=stop,
                                          collector_factory=(lambda index: factories.collector(power, factories.clock))
                                          if args.collect_power else None,
                                          emit=emit_request,
                                          warmup='formal' if args.warmup == 'formal' else args.warmup == 'basic',
                                          **qualification_args)
                with _signals(stop):
                    session = runner.run(prepared, options, provenance=provenance, session_id=session_id)
            finally:
                if entered:
                    try:
                        active_lock.__exit__(None, None, None)
                    except Exception as error:
                        # Unlock is independent cleanup. Preserve any earlier
                        # loading/execution error and completed request records.
                        unlock_errors.append(error_record(error, stage='device_unlock'))
        except Exception as error:
            if isinstance(error, PreparationError):
                failure_code = 3
            elif isinstance(error, ImportError):
                failure_code = 1
            session = _failure(error, stage, stop.exit_code or failure_code,
                               recorder=recorder, provenance=provenance, session_id=session_id)
        except KeyboardInterrupt as error:
            stop.request(int(signal.SIGINT))
            session = _failure(error, stage, stop.exit_code, recorder=recorder,
                               provenance=provenance, session_id=session_id)
        if unlock_errors:
            session.setdefault('provenance', {})['formal_performance_qualified'] = False
            session['errors'].extend(unlock_errors)
            if session['exit_code'] < 128:
                session.update(status='failed', exit_code=1)
            try:
                recorder.save_session(session)
            except Exception as error:
                session['errors'].append(error_record(error, stage='session_save'))
        if session['exit_code']:
            print(f"kvolt run: {session['status']}: exit {session['exit_code']}", file=sys.stderr)
        output(_summary(session, recorder))
    return session['exit_code']


def main(argv=None, *, factories=None, lock=None, emit=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = configure_parser(RunParser(prog='kvolt run', description='Sequential adaptive development execution'))
    try:
        args = parser.parse_args(argv)
    except RunArgumentError as error:
        session = _failure(error, 'arguments', 2, provenance={
            'command': {'original': ['kvolt', 'run', *argv], 'resolved': None}})
        with _stdout_channel(emit) as output:
            parser.print_usage(file=sys.stderr)
            print(f'kvolt run: arguments: {error}', file=sys.stderr)
            output(_summary(session, None))
        return 2
    args._original = ['kvolt', 'run', *argv]
    return run(args, factories=factories, lock=lock, emit=emit)
