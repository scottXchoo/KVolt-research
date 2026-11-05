"""CPU policy diagnostics. Scientific and model libraries are never help imports."""
import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

SCHEMA = 'kvolt.policy_diagnostic.v1'
SCENARIOS = ('baseline', 'flat_quality', 'nonmonotonic', 'exact_ties')


def _parser():
    parser = argparse.ArgumentParser(prog='kvolt', description='KVolt CPU policy diagnostics')
    commands = parser.add_subparsers(dest='command', required=True)
    diagnose = commands.add_parser('diagnose', help='predict policy choices using assumed CPU phase times')
    diagnose.add_argument('--length', type=int, required=True, help='assumed input token length, 1024..8192; no tokenizer')
    diagnose.add_argument('--max-new-tokens', type=int, default=512, help='output cap, 1..512 (default: 512)')
    diagnose.add_argument('--pc-elapsed-ns', type=int, required=True, help='assumed nonnegative PC elapsed ns, not a measurement')
    diagnose.add_argument('--dc-elapsed-ns', type=int, help='assumed nonnegative DC elapsed ns; required when DC is active')
    diagnose.add_argument('--first-token-id', type=int, default=17, help='assumed first token ID, 0..128255 (default: 17); EOS skips DC')
    source = diagnose.add_mutually_exclusive_group(required=True)
    source.add_argument('--lut-bundle', type=Path, help='sealed measured LUT bundle; never falls back to Mock')
    source.add_argument('--mock', choices=SCENARIOS, help='explicit synthetic LUT scenario')
    diagnose.add_argument('--output', type=Path, help='also save the JSON to a new file; existing paths are refused')
    return parser


def _base_report(args):
    return {
        'schema_version': SCHEMA,
        'record_type': 'policy_diagnostic',
        'synthetic_assumptions': True,
        'predictions_only': True,
        'measurements': None,
        'inputs': {'length': args.length, 'max_new_tokens': args.max_new_tokens},
        'assumptions': {
            'first_token_id': args.first_token_id,
            'first_token_measured': False,
            'pc_elapsed_ns': args.pc_elapsed_ns,
            'dc_elapsed_ns': args.dc_elapsed_ns,
            'times_measured': False,
            'phase_clock_origins_ns': {'t0_ns': 0, 't1_ns': 0},
            'dc_elapsed_applied': False,
        },
    }


def _validate_inputs(args):
    from .config import EOS_TOKEN_IDS, require_int
    from .policy.types import length
    length(args.length)
    require_int(args.max_new_tokens, 'Nmax', 1, 512)
    require_int(args.first_token_id, 'assumed first token ID', 0, 128255)
    require_int(args.pc_elapsed_ns, 'assumed PC elapsed ns', 0)
    if args.dc_elapsed_ns is not None:
        require_int(args.dc_elapsed_ns, 'assumed DC elapsed ns', 0)
    active = args.max_new_tokens != 1 and args.first_token_id not in EOS_TOKEN_IDS
    if active and args.dc_elapsed_ns is None:
        raise ValueError('--dc-elapsed-ns is required when Decode is active')
    return active


def _build_repository(args):
    if args.lut_bundle is not None:
        from .policy.lut import MeasuredLUTRepository
        return MeasuredLUTRepository(args.lut_bundle)
    from .policy.mock import MockLUTRepository
    return MockLUTRepository(args.mock)


def _new_knee_adapter():
    from .policy.knee import KneeAdapter
    return KneeAdapter()


def _source_metadata(provenance):
    """Keep source conditions/flags and original hashes, omitting bulk raw rows."""
    raw = json.loads(provenance.metadata_json)
    excluded = {'entries', 'cells', 'macro_cells', 'fp16_references', 'outputs_sha256',
                'source_sha256', 'requests', 'knots', 'basic60_strata', 'hard_strata'}

    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items() if key not in excluded}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    return {
        'kind': provenance.kind,
        'source_device': provenance.source_device,
        'source_commit': provenance.source_commit,
        'manifest_sha256': provenance.manifest_sha256,
        'files_sha256': dict(provenance.files_sha256),
        'notes': list(provenance.notes),
        'metadata': compact(raw),
        'metadata_complete': False if provenance.kind == 'measured' else True,
        'full_metadata_reference': 'sealed source files identified by files_sha256' if provenance.kind == 'measured' else None,
    }


def _runtime_metadata():
    package = Path(__file__).resolve().parent
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(package.rglob('*.py'))}
    root = package.parent.parent
    commit = None
    dirty = None
    # A site-packages directory inside another Git checkout is not this source.
    try:
        top = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                             text=True, capture_output=True, check=True, timeout=5).stdout.strip()
        if Path(top).resolve() == root:
            commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                    text=True, capture_output=True, check=True, timeout=5).stdout.strip()
            status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--',
                                     'src/kvolt_runtime', 'pyproject.toml', 'tools/diagnose_policy.py'],
                                    text=True, capture_output=True, check=True, timeout=5).stdout
            dirty = bool(status)
    except (OSError, subprocess.SubprocessError):
        pass
    versions = {}
    installed = {}
    for name in ('numpy', 'scipy', 'kneed'):
        module = sys.modules.get(name)
        versions[name] = getattr(module, '__version__', None) if module else None
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    return {
        'python': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'source_commit': commit,
        'source_worktree_dirty': dirty,
        'source_sha256': hashes,
        'loaded_versions': versions,
        'installed_distribution_versions': installed,
        'required_policy_versions': {'numpy': '1.26.4', 'scipy': '1.15.3', 'kneed': '0.9.0'},
    }


class _DiagnosticFailure(Exception):
    def __init__(self, stage, cause):
        super().__init__(str(cause))
        self.stage, self.cause = stage, cause


def _diagnose(args, result):
    stage = 'input'
    try:
        active = _validate_inputs(args)
        stage = 'lut'
        repo = _build_repository(args)
        stage = 'preflight'
        repo.preflight(args.length)
        result['preflight'] = {'status': 'supported', 'scope': 'full_quality_prefill_decode_and_targets'}
        expected_kind = 'measured' if args.lut_bundle is not None else 'synthetic'
        if repo.provenance.kind != expected_kind:
            raise ValueError('LUT provenance kind does not match the explicitly selected source')
        result['source'] = _source_metadata(repo.provenance)
        # Real dependency setup happens outside the assumed online phases.
        adapter = None
        if active:
            stage = 'dependency'
            adapter = _new_knee_adapter()
        stage = 'prefill'
        from .policy.prefill import PrefillController
        result['prefill'] = asdict(PrefillController(repo, clock=lambda: args.pc_elapsed_ns).choose(args.length, 0))
        stage = 'decode'
        if active:
            from .policy.decode import DecodeController
            decision = DecodeController(repo, adapter, clock=lambda: args.dc_elapsed_ns).choose(
                args.length, args.max_new_tokens, args.first_token_id, 0)
            result['assumptions']['dc_elapsed_applied'] = True
        else:
            from .policy.decode import DecodeController
            # The real controller owns skip reasons and precedence. Its unused
            # adapter sentinel avoids loading scientific dependencies on skip.
            def unused_clock():
                raise RuntimeError('Decode skip unexpectedly read its clock')
            decision = DecodeController(repo, knee_adapter=object(), clock=unused_clock).choose(
                args.length, args.max_new_tokens, args.first_token_id, 0)
        result['decode'] = asdict(decision)
        result['status'] = 'ok'
    except (ValueError, RuntimeError, OSError, ImportError, OverflowError) as error:
        raise _DiagnosticFailure(stage, error) from error
    return result


def _error_report(result, stage, error):
    result['status'] = 'rejected' if stage in ('input', 'lut', 'preflight', 'output') else 'error'
    result['error'] = {'stage': stage, 'type': type(error).__name__, 'reason': str(error)}
    return result


def main(argv=None):
    """Print exactly one JSON record after valid argument syntax; return 0 or 2."""
    args = _parser().parse_args(argv)
    result = _base_report(args)
    stream = None
    exit_code = 0
    try:
        # Exclusive creation also refuses dangling symlinks and concurrent writers.
        if args.output is not None:
            stream = args.output.open('x', encoding='utf-8')
    except OSError as error:
        result = _error_report(result, 'output', error)
        exit_code = 2
    if not exit_code:
        try:
            result = _diagnose(args, result)
        except _DiagnosticFailure as error:
            result = _error_report(result, error.stage, error.cause)
            exit_code = 2
    try:
        result['runtime'] = _runtime_metadata()
    except (OSError, RuntimeError, ValueError) as error:
        result = _error_report(result, 'provenance', error)
        exit_code = 2
    try:
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
    except (ValueError, TypeError) as error:
        result = _error_report(_base_report(args), 'serialization', error)
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
        exit_code = 2
    if stream is not None:
        try:
            with stream:
                stream.write(encoded + '\n')
        except OSError as error:
            result = _error_report(result, 'output', error)
            encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
            exit_code = 2
    if exit_code:
        print(f"kvolt diagnose: {result['error']['stage']}: {result['error']['reason']}", file=sys.stderr)
    print(encoded)
    return exit_code
