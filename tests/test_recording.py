"""Exclusive session storage and atomic, lossless result evidence."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.execution.inputs import preflight_inputs
from kvolt_runtime.recording import SessionRecorder, json_line, to_json_native


def inputs(tmp_path, *, manifest=False):
    prompt = tmp_path / 'p.txt'
    prompt.write_bytes(b' original\r\ntext ')
    kwargs = {'prompt_file': prompt, 'manifest': None}
    if manifest:
        path = tmp_path / 'm.json'
        path.write_bytes(b'{\r\n "schema_version":"kvolt-request-manifest-v1", "requests":[{"request_id":"../../escape", "prompt_file":"p.txt"}]}\r\n')
        kwargs = {'prompt_file': None, 'manifest': path}
    class Repository:
        def preflight(self, length):
            assert length == 1024
    return preflight_inputs(**kwargs, tokenizer=lambda *a, **k: {'input_ids': [4] * 1024}, repo=Repository())


@pytest.mark.parametrize('kind', ['directory', 'file', 'symlink', 'dangling'])
def test_existing_output_never_overwritten(tmp_path, kind):
    path = tmp_path / 'out'
    if kind == 'directory':
        path.mkdir()
        (path / 'keep').write_text('original')
    elif kind == 'file':
        path.write_text('original')
    else:
        target = tmp_path / 'target'
        if kind == 'symlink':
            target.mkdir()
        path.symlink_to(target)
    with pytest.raises(FileExistsError):
        SessionRecorder(path)
    if kind == 'directory':
        assert (path / 'keep').read_text() == 'original'
    elif kind == 'file':
        assert path.read_text() == 'original'
    else:
        assert path.is_symlink()


@pytest.mark.parametrize('manifest', [False, True])
def test_preflight_preserves_original_bytes_after_sources_change(tmp_path, manifest):
    prepared = inputs(tmp_path, manifest=manifest)
    (tmp_path / 'p.txt').write_text('changed after preflight')
    if manifest:
        (tmp_path / 'm.json').unlink()
    recorder = SessionRecorder(tmp_path / 'out')
    paths = recorder.save_preflight(prepared)
    prompt = prepared.requests[0].prompt
    assert (recorder.output_dir / f'inputs/{prompt.sha256}.txt').read_bytes() == b' original\r\ntext '
    assert paths['manifest_original_path'] == ('manifest.original.json' if manifest else None)
    if manifest:
        raw = (recorder.output_dir / 'manifest.original.json').read_bytes()
        assert raw == prepared.manifest_raw
        assert hashlib.sha256(raw).hexdigest() == prepared.manifest_sha256
    assert json.loads((recorder.output_dir / 'manifest.resolved.json').read_text()) == prepared.resolved_manifest
    evidence = json.loads((recorder.output_dir / 'preflight.json').read_text())
    assert evidence['requests'][0]['token_ids'] == [4] * 1024
    assert evidence['requests'][0]['length'] == 1024


def test_result_statuses_and_partial_prefix_preserved_exactly(tmp_path):
    prepared = inputs(tmp_path, manifest=True)
    recorder = SessionRecorder(tmp_path / 'out')
    result = {'generation': {'status': 'interrupted'}, 'timing': {'valid': False, 'ttft_s': 0.2},
              'energy': {'status': 'not_collected'}, 'cleanup': {'status': 'failed', 'errors': ['restore']}}
    generation = {'status': 'interrupted', 'token_ids': [9, 10], 'ready_ns': [100, 200]}
    paths = recorder.save_request(0, prepared.requests[0], result, generation, '부분\r\n')
    assert paths == {'request_path': 'requests/000000/request.json',
                     'generation_path': 'requests/000000/generation.json',
                     'result_path': 'requests/000000/result.json',
                     'output_path': 'requests/000000/output.txt'}
    assert json.loads((recorder.output_dir / paths['result_path']).read_text()) == result
    assert json.loads((recorder.output_dir / paths['generation_path']).read_text()) == generation
    assert (recorder.output_dir / paths['output_path']).read_bytes() == '부분\r\n'.encode()
    assert not (tmp_path / 'escape').exists()


def test_not_run_has_no_invented_output_or_timing(tmp_path):
    prepared = inputs(tmp_path)
    recorder = SessionRecorder(tmp_path / 'out')
    result = {'generation': {'status': 'not_run'}, 'timing': None, 'energy': None, 'cleanup': None}
    paths = recorder.save_request(3, prepared.requests[0], result)
    assert paths['generation_path'] is None and paths['output_path'] is None
    assert not (recorder.output_dir / 'requests/000003/output.txt').exists()
    assert json.loads((recorder.output_dir / paths['result_path']).read_text()) == result
    assert recorder.session_path is None
    assert recorder.save_session({'status': 'failed', 'counts': {'not_run': 1}}) == 'session.json'
    assert recorder.session_path == 'session.json'


def test_atomic_replace_failure_preserves_previous_json_and_cleans_temp(tmp_path, monkeypatch):
    recorder = SessionRecorder(tmp_path / 'out')
    recorder.atomic_json('session.json', {'status': 'running'})
    def fail(*args, **kwargs):
        raise OSError('storage unavailable')
    monkeypatch.setattr('kvolt_runtime.recording.os.replace', fail)
    with pytest.raises(OSError, match='storage unavailable'):
        recorder.save_session({'status': 'completed'})
    assert json.loads((recorder.output_dir / 'session.json').read_text()) == {'status': 'running'}
    assert sorted(p.name for p in recorder.output_dir.iterdir()) == ['session.json']


def test_fsync_failure_does_not_publish(tmp_path, monkeypatch):
    recorder = SessionRecorder(tmp_path / 'out')
    monkeypatch.setattr('kvolt_runtime.recording.os.fsync', lambda *a: (_ for _ in ()).throw(OSError('disk error')))
    with pytest.raises(OSError, match='disk error'):
        recorder.atomic_json('result.json', {'status': 'completed'})
    assert recorder.path('result.json') is None
    assert list(recorder.output_dir.iterdir()) == []


@pytest.mark.parametrize('path', ['../escape', '/absolute', 'nested/../../escape', '.', ''])
def test_no_path_escape(tmp_path, path):
    recorder = SessionRecorder(tmp_path / 'out')
    with pytest.raises(ContractError):
        recorder.write_text(path, 'text')


def test_no_symlink_escape_inside_session(tmp_path):
    recorder = SessionRecorder(tmp_path / 'out')
    target = tmp_path / 'outside'
    target.mkdir()
    (recorder.output_dir / 'nested').symlink_to(target)
    with pytest.raises(OSError):
        recorder.write_text('nested/result.json', 'text')
    (recorder.output_dir / 'file').symlink_to(target / 'new')
    with pytest.raises(ContractError):
        recorder.write_text('file', 'text')
    assert list(target.iterdir()) == []


def test_session_directory_replacement_rejected(tmp_path):
    recorder = SessionRecorder(tmp_path / 'out')
    recorder.output_dir.rename(tmp_path / 'moved')
    recorder.output_dir.mkdir()
    with pytest.raises(ContractError, match='ownership'):
        recorder.write_text('result.json', 'text')
    assert list(recorder.output_dir.iterdir()) == []


def test_native_dataclass_json_and_nonfinite_rejection(tmp_path):
    @dataclass(frozen=True)
    class Evidence:
        times_ns: tuple[int, ...]
        path: Path
    assert to_json_native(Evidence((100, 200), Path('file'))) == {'times_ns': [100, 200], 'path': 'file'}
    assert json.loads(json_line({'text': '한국어\n', 'valid': None})) == {'text': '한국어\n', 'valid': None}
    assert json_line({'text': '한국어\n'}).count('\n') == 1
    recorder = SessionRecorder(tmp_path / 'out')
    with pytest.raises((ContractError, ValueError)):
        recorder.atomic_json('bad.json', {'time': float('nan')})
    assert recorder.path('bad.json') is None


@pytest.mark.parametrize('index', [True, -1, 1.5, '0'])
def test_request_index_strict_integer(tmp_path, index):
    prepared = inputs(tmp_path)
    recorder = SessionRecorder(tmp_path / 'out')
    with pytest.raises(ContractError):
        recorder.save_request(index, prepared.requests[0], {})


def test_directory_fsync_failure_remains_storage_failure_after_replace(tmp_path, monkeypatch):
    recorder = SessionRecorder(tmp_path / 'out')
    original_fsync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError('directory durability unknown')
        return original_fsync(fd)

    monkeypatch.setattr('kvolt_runtime.recording.os.fsync', fail_directory)
    with pytest.raises(OSError, match='directory durability unknown'):
        recorder.save_session({'status': 'completed'})
    assert json.loads((recorder.output_dir / 'session.json').read_text()) == {'status': 'completed'}
    assert sorted(p.name for p in recorder.output_dir.iterdir()) == ['session.json']


def test_missing_parent_and_absent_artifacts_have_no_fake_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        SessionRecorder(tmp_path / 'missing' / 'out')
    recorder = SessionRecorder(tmp_path / 'out')
    assert recorder.path('missing/result.json') is None
    (recorder.output_dir / 'directory.json').mkdir()
    (recorder.output_dir / 'dangling.json').symlink_to(tmp_path / 'absent')
    assert recorder.path('directory.json') is None
    assert recorder.path('dangling.json') is None
    prepared = inputs(tmp_path)
    paths = recorder.save_request(1000000, prepared.requests[0], {}, output_text='')
    assert paths['output_path'] == 'requests/1000000/output.txt'
    assert (recorder.output_dir / paths['output_path']).read_bytes() == b''


@pytest.mark.parametrize('replace_root', [False, True])
def test_directory_replacement_during_write_cannot_report_success(tmp_path, monkeypatch, replace_root):
    recorder = SessionRecorder(tmp_path / 'out')
    (recorder.output_dir / 'nested').mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    original_replace = os.replace

    def replace_after_directory_swap(*args, **kwargs):
        directory = recorder.output_dir if replace_root else recorder.output_dir / 'nested'
        directory.rename(tmp_path / 'moved')
        directory.symlink_to(outside, target_is_directory=True)
        return original_replace(*args, **kwargs)

    monkeypatch.setattr('kvolt_runtime.recording.os.replace', replace_after_directory_swap)
    with pytest.raises((ContractError, OSError)):
        recorder.write_text('nested/result.json', 'must not report a visible artifact')
    assert list(outside.iterdir()) == []
