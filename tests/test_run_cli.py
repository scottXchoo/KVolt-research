"""Standard run stays CPU-only until a held common hardware lock exists."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvolt_runtime.config import SNAPSHOT
from kvolt_runtime.execution.contracts import PreparationError
from kvolt_runtime.policy.mock import MockLUTRepository


def arguments(tmp_path, *extra):
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('x' * 1024)
    return ['--prompt-file', str(prompt), '--model-dir', str(tmp_path / SNAPSHOT),
            '--mock', 'baseline', '--output-dir', str(tmp_path / 'out'),
            '--device-profile', str(tmp_path / 'device.json'), '--warmup', 'none', *extra]


class Lock:
    def __init__(self, events, busy=False):
        self.events, self.busy, self.held = events, busy, False
    def __enter__(self):
        self.events.append('lock')
        if self.busy:
            raise PreparationError('HARDWARE_BUSY', 'device_lock', 'busy')
        self.held = True
        return self
    def assert_held(self):
        assert self.held
    def __exit__(self, *_):
        self.events.append('unlock')
        self.held = False


def factories(events, *, busy=False):
    from kvolt_runtime.run_cli import RunFactories
    lock = Lock(events, busy)
    class Runner:
        def __init__(self, **kw):
            self.kw = kw
            assert lock.held
        def run(self, prepared, options, **kw):
            events.append('runner')
            print('loader noise')
            session = {'schema_version': 'kvolt-session-v1', 'session_id': 'test',
                       'status': 'completed', 'counts': {}, 'errors': [], 'exit_code': 0,
                       'provenance': kw['provenance']}
            self.kw['recorder'].save_session(session)
            self.kw['emit']({'record_type': 'request_result', 'request_id': prepared.requests[0].spec.request_id})
            self.kw['emit']({'record_type': 'session_result', **session,
                             'session_path': 'session.json'})
            return session
    def tokenizer(path):
        events.append('tokenizer')
        return (lambda text, **kw: {'input_ids': [7] * len(text)}), {'test': True}
    def repository(args):
        events.append('repo')
        return MockLUTRepository(args.mock)
    def backend(args, clock):
        assert lock.held
        events.append('backend')
        return object()
    def device(profile, held, clock):
        held.assert_held()
        events.append('device')
        return object()
    def decode(repo, prepared, clock):
        events.append('decode_setup')
        return object()
    return replace(RunFactories(), tokenizer=tokenizer, repository=repository,
                   device_profile=lambda path: SimpleNamespace(as_dict=lambda: {'test': True}),
                   decode_controller=decode, hardware_lock=lambda: lock,
                   backend=backend, device=device, runner=Runner,
                   runtime_metadata=lambda: {'test': True}), lock


def test_new_help_and_import_have_no_gpu_or_science_imports():
    code = """import sys
from kvolt_runtime.cli import main
try: main(['run', '--help'])
except SystemExit as e: assert e.code == 0
assert not ({'torch','transformers','triton','flash_attn','numpy','scipy','kneed'} & set(sys.modules))
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert '--prompt-file' in result.stdout and '--device-profile' in result.stdout


def test_preflight_and_dependency_before_lock_backend_after_lock(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    events = []
    seam, lock = factories(events)
    assert main(arguments(tmp_path), factories=seam) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r['record_type'] for r in records] == ['request_result', 'session_result']
    assert events == ['repo', 'tokenizer', 'decode_setup', 'lock', 'backend', 'device', 'runner', 'unlock']
    assert lock.held is False
    assert (tmp_path / 'out/preflight.json').is_file()
    saved = json.loads((tmp_path / 'out/session.json').read_text())
    assert saved['provenance']['command']['resolved']['warmup'] == 'none'


@pytest.mark.parametrize('suffix', [['--mode', 'performance'], ['--max-new-tokens', '0'],
                                  ['--group', '8'], ['--collect-power']])
def test_preflight_rejection_always_jsonl_and_no_gpu(tmp_path, capsys, suffix):
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events)
    assert main(arguments(tmp_path, *suffix), factories=seam) == 2
    record = json.loads(capsys.readouterr().out)
    assert record['record_type'] == 'session_result' and record['exit_code'] == 2
    assert record['command']['original'][-len(suffix):] == suffix
    assert 'backend' not in events and 'lock' not in events


def test_busy_lock_returns_three_with_saved_preflight(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events, busy=True)
    assert main(arguments(tmp_path), factories=seam) == 3
    record = json.loads(capsys.readouterr().out)
    assert record['exit_code'] == 3 and record['session_path'] == 'session.json'
    assert 'backend' not in events
    assert (tmp_path / 'out/preflight.json').is_file()


@pytest.mark.parametrize('kind', ['directory', 'symlink'])
def test_existing_output_refuses_all_writes(tmp_path, capsys, kind):
    from kvolt_runtime.run_cli import main
    args = arguments(tmp_path)
    if kind == 'directory':
        (tmp_path / 'out').mkdir()
        (tmp_path / 'out/untouched').write_text('original')
    else:
        (tmp_path / 'out').symlink_to(tmp_path / 'missing')
    events = []
    seam, _ = factories(events)
    assert main(args, factories=seam) == 2
    record = json.loads(capsys.readouterr().out)
    assert record['artifact_root'] is None and record['session_path'] is None
    assert events == []
    if kind == 'directory':
        assert list((tmp_path / 'out').iterdir()) == [tmp_path / 'out/untouched']


def test_preheld_lock_is_asserted_and_never_released(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    events = []
    seam, lock = factories(events)
    lock.held = True
    assert main(arguments(tmp_path), factories=seam, lock=lock) == 0
    assert lock.held and 'lock' not in events and 'unlock' not in events
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['exit_code'] == 0


def test_missing_knee_dependency_is_technical_before_lock(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events)
    def missing(*args):
        raise ImportError('kneed unavailable')
    seam = replace(seam, decode_controller=missing)
    assert main(arguments(tmp_path), factories=seam) == 1
    assert json.loads(capsys.readouterr().out)['errors'][0]['stage'] == 'dependency'
    assert 'lock' not in events


def test_manifest_late_invalid_request_rejects_whole_session_before_lock(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    args = arguments(tmp_path)
    prompt_index = args.index('--prompt-file')
    del args[prompt_index:prompt_index + 2]
    (tmp_path / 'short.txt').write_text('a' * 1023)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 'kvolt-request-manifest-v1', 'requests': [
        {'request_id': 'good', 'prompt_file': 'prompt.txt'},
        {'request_id': 'bad', 'prompt_file': 'short.txt'}]}))
    events = []
    seam, _ = factories(events)
    assert main([*args, '--manifest', str(manifest)], factories=seam) == 2
    assert 'backend' not in events and 'lock' not in events
    record = json.loads(capsys.readouterr().out)
    assert record['errors'][0]['stage'] == 'input_preflight'


def test_invalid_cli_nmax_still_rejected_when_manifest_overrides(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    args = arguments(tmp_path)
    del args[:2]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 'kvolt-request-manifest-v1', 'requests': [
        {'request_id': 'one', 'prompt_file': 'prompt.txt', 'nmax': 1}]}))
    events = []
    seam, _ = factories(events)
    assert main([*args, '--manifest', str(manifest), '--max-new-tokens', '0'], factories=seam) == 2
    assert 'lock' not in events
    assert json.loads(capsys.readouterr().out)['exit_code'] == 2


def test_actual_decode_setup_nmax_one_never_loads_scientific_dependencies():
    code = """import sys
from types import SimpleNamespace
from kvolt_runtime.run_cli import _decode_controller
from kvolt_runtime.policy.mock import MockLUTRepository
prepared = SimpleNamespace(requests=[SimpleNamespace(spec=SimpleNamespace(nmax=1))])
controller = _decode_controller(MockLUTRepository(), prepared, lambda: 1)
assert controller.choose(1024,1,17,1).status == 'skipped'
assert not ({'numpy','scipy','kneed','torch','transformers'} & set(sys.modules))
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, env=environment)
    assert result.returncode == 0, result.stderr


def test_pinned_cpu_tokenizer_hashes_small_files_and_disables_special_tokens(tmp_path, monkeypatch):
    from kvolt_runtime import run_cli
    from kvolt_runtime.backend import provenance
    checks, calls = [], []
    model = tmp_path / SNAPSHOT
    def checked(path, expected):
        checks.append((path.name, expected))
        return {'path': str(path), 'sha256': expected}
    monkeypatch.setattr(provenance, 'checked_file', checked)
    monkeypatch.setattr(run_cli.metadata, 'version', lambda name: '0.19.1')
    class RustTokenizer:
        @classmethod
        def from_file(cls, path):
            calls.append(('from_file', path))
            return cls()
        def no_truncation(self): calls.append('no_truncation')
        def no_padding(self): calls.append('no_padding')
        def encode(self, text, *, add_special_tokens):
            calls.append((text, add_special_tokens))
            return SimpleNamespace(ids=[17, 23])
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(Tokenizer=RustTokenizer))
    tokenizer, evidence = run_cli._tokenizer(model)
    assert tokenizer('a\r\nb', add_special_tokens=False) == {'input_ids': [17, 23]}
    assert [name for name, _ in checks] == ['tokenizer.json', 'tokenizer_config.json', 'config.json', 'generation_config.json']
    assert all(provenance.MODEL_FILES[name] == digest for name, digest in checks)
    assert calls == [('from_file', str(model / 'tokenizer.json')), 'no_truncation', 'no_padding', ('a\r\nb', False)]
    assert evidence['version'] == '0.19.1' and evidence['add_special_tokens'] is False


def test_signal_handlers_only_set_flag_and_are_restored(tmp_path, capsys):
    import signal
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events)
    original_runner = seam.runner
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    class InspectRunner(original_runner):
        def run(self, *args, **kwargs):
            stop = self.kw['stop']
            handler = signal.getsignal(signal.SIGTERM)
            assert handler != previous[signal.SIGTERM]
            handler(int(signal.SIGTERM), None)
            assert stop.signal_number == signal.SIGTERM and stop.exit_code == 143
            handler(int(signal.SIGINT), None)
            assert stop.signal_number == signal.SIGTERM
            session = super().run(*args, **kwargs)
            session.update(status='interrupted', exit_code=stop.exit_code)
            self.kw['recorder'].save_session(session)
            return session
    assert main(arguments(tmp_path), factories=replace(seam, runner=InspectRunner)) == 143
    assert all(signal.getsignal(number) == handler for number, handler in previous.items())
    capsys.readouterr()


def test_terminal_summary_published_after_unlock(tmp_path):
    from kvolt_runtime.run_cli import main
    events, records = [], []
    seam, lock = factories(events)
    def emit(record):
        records.append(record)
        assert lock.held == (record['record_type'] == 'request_result')
    assert main(arguments(tmp_path), factories=seam, emit=emit) == 0
    assert records[-1]['record_type'] == 'session_result'


@pytest.mark.parametrize('custom_emit', [False, True])
def test_native_fd_stdout_noise_is_redirected(custom_emit):
    code = f"""import ctypes, json, os, sys
from kvolt_runtime.run_cli import _stdout_channel
records = []
with _stdout_channel(records.append if {custom_emit!r} else None) as emit:
    print('python loader noise')
    os.write(1,b'native loader noise\\n')
    ctypes.CDLL(None).printf(b'buffered native noise\\n')
    emit({{'record_type':'session_result','exit_code':0}})
if {custom_emit!r}: print(json.dumps(records[0]))
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['record_type'] == 'session_result'
    assert all(message in result.stderr for message in ('python loader noise', 'native loader noise', 'buffered native noise'))


@pytest.mark.parametrize('runner_fails', [False, True])
def test_unlock_failure_preserves_primary_error_and_results(tmp_path, capsys, runner_fails):
    from kvolt_runtime.run_cli import main
    events = []
    seam, lock = factories(events)
    class BrokenUnlock(Lock):
        def __exit__(self, *args):
            super().__exit__(*args)
            raise OSError('unlock failed')
    lock.__class__ = BrokenUnlock
    if runner_fails:
        def runner(**kwargs):
            class Broken:
                def run(self, *args, **kwargs):
                    raise RuntimeError('primary execution failure')
            return Broken()
        seam = replace(seam, runner=runner)
    assert main(arguments(tmp_path), factories=seam) == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = records[-1]
    assert summary['errors'][-1]['stage'] == 'device_unlock'
    if runner_fails:
        assert summary['errors'][0]['message'] == 'primary execution failure'
        assert len(summary['errors']) == 2
    else:
        assert records[0]['record_type'] == 'request_result'
    saved = json.loads((tmp_path / 'out/session.json').read_text())
    assert saved['exit_code'] == 1 and saved['errors'] == summary['errors']


def test_keyboard_interrupt_before_signal_handler_install_returns_130(tmp_path, capsys):
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events)
    def interrupted(args):
        raise KeyboardInterrupt()
    assert main(arguments(tmp_path), factories=replace(seam, repository=interrupted)) == 130
    summary = json.loads(capsys.readouterr().out)
    assert summary['status'] == 'interrupted' and summary['exit_code'] == 130
    assert 'lock' not in events
