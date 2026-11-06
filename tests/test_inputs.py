"""CPU input preflight preserves byte identity and rejects ambiguous manifests."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.execution.inputs import preflight_inputs


class Tokenizer:
    def __init__(self, length=1024):
        self.length, self.calls = length, []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {'input_ids': list(range(self.length))}


class Repository:
    def __init__(self):
        self.lengths = []

    def preflight(self, length):
        self.lengths.append(length)


def manifest_file(tmp_path, requests, **extra):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'schema_version': 'kvolt-request-manifest-v1',
                                'requests': requests, **extra}), encoding='utf-8')
    return path


def test_prompt_bytes_ids_and_metadata_frozen(tmp_path):
    path = tmp_path / 'prompt.txt'
    raw = b'  first\r\nsecond\r\n '
    path.write_bytes(raw)
    tokenizer, repo = Tokenizer(), Repository()
    prepared = preflight_inputs(prompt_file=path, manifest=None, tokenizer=tokenizer, repo=repo)
    request = prepared.requests[0]
    assert tokenizer.calls == [(raw.decode(), {'add_special_tokens': False})]
    assert request.prompt.text.encode() == raw
    assert request.spec.prompt_sha256 == hashlib.sha256(raw).hexdigest()
    assert request.spec.token_ids == tuple(range(1024))
    assert request.spec.nmax == 512 and repo.lengths == [1024]
    path.write_text('changed')
    assert request.prompt.text.encode() == raw
    with pytest.raises(FrozenInstanceError):
        prepared.requests = ()
    assert prepared.manifest_raw is None and prepared.manifest_path is None


def test_manifest_relative_paths_original_resolved_and_nmax(tmp_path):
    (tmp_path / 'p.txt').write_text('original')
    path = manifest_file(tmp_path, [
        {'request_id': '../unsafe/id', 'prompt_file': 'p.txt', 'nmax': 1,
         'metadata': {'nested': [{'repeat_id': 4}]}},
        {'request_id': 'r2', 'prompt_file': 'p.txt'},
    ])
    raw = path.read_bytes()
    repo = Repository()
    prepared = preflight_inputs(prompt_file=None, manifest=path, tokenizer=Tokenizer(),
                                repo=repo, cli_nmax=7)
    assert [r.spec.nmax for r in prepared.requests] == [1, 7]
    assert repo.lengths == [1024, 1024]
    assert prepared.manifest_raw == raw
    assert prepared.manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert prepared.manifest_path == path.resolve()
    resolved = prepared.resolved_manifest
    assert resolved['requests'][0]['prompt_file'] == str((tmp_path / 'p.txt').resolve())
    resolved['requests'][0]['metadata']['nested'].clear()
    assert prepared.requests[0].metadata['nested'] == [{'repeat_id': 4}]
    metadata = prepared.requests[0].metadata
    metadata['nested'].clear()
    assert prepared.requests[0].metadata['nested'] == [{'repeat_id': 4}]


@pytest.mark.parametrize('bad', [None, True, False, '1', 1.0, 0, 513])
@pytest.mark.parametrize('source', ['cli', 'item'])
def test_every_supplied_nmax_validated_even_if_overridden(tmp_path, bad, source):
    (tmp_path / 'p').write_text('text')
    path = manifest_file(tmp_path, [{'request_id': 'r', 'prompt_file': 'p',
                                     'nmax': bad if source == 'item' else 1}])
    kwargs = {'cli_nmax': bad} if source == 'cli' else {}
    with pytest.raises(ContractError):
        preflight_inputs(prompt_file=None, manifest=path, tokenizer=Tokenizer(), repo=Repository(), **kwargs)


@pytest.mark.parametrize('text', [
    '{}', '[]', '{',
    '{"schema_version":"kvolt-request-manifest-v1","requests":[]}',
    '{"schema_version":"wrong","requests":[{}]}',
    '{"schema_version":"kvolt-request-manifest-v1","requests":[{}],"extra":1}',
    '{"schema_version":"kvolt-request-manifest-v1","schema_version":"kvolt-request-manifest-v1","requests":[]}',
    '{"schema_version":"kvolt-request-manifest-v1","requests":[{"request_id":"r","prompt_file":"p","prompt_file":"p"}]}',
    '{"schema_version":"kvolt-request-manifest-v1","requests":[{"request_id":"r","prompt_file":"p","metadata":{"x":{"a":1,"a":2}}}]}',
    '{"schema_version":"kvolt-request-manifest-v1","requests":[{"request_id":"r","prompt_file":"p","metadata":{"x":NaN}}]}',
    '{"schema_version":"kvolt-request-manifest-v1","requests":[{"request_id":"r","prompt_file":"p","metadata":{"x":1e999}}]}',
])
def test_manifest_rejects_malformed_or_ambiguous_json(tmp_path, text):
    path = tmp_path / 'm.json'
    path.write_text(text)
    tokenizer = Tokenizer()
    with pytest.raises(ContractError):
        preflight_inputs(prompt_file=None, manifest=path, tokenizer=tokenizer, repo=Repository())
    assert tokenizer.calls == []


@pytest.mark.parametrize('updates', [
    {'request_id': ''}, {'request_id': ' \t'}, {'request_id': 1},
    {'prompt_file': None}, {'prompt_file': ''}, {'metadata': None}, {'metadata': []}, {'unknown': 1},
])
def test_bad_request_fields(tmp_path, updates):
    item = {'request_id': 'r', 'prompt_file': 'p', **updates}
    path = manifest_file(tmp_path, [item])
    with pytest.raises(ContractError):
        preflight_inputs(prompt_file=None, manifest=path, tokenizer=Tokenizer(), repo=Repository())


def test_duplicate_request_ids_rejected_before_tokenization(tmp_path):
    item = {'request_id': 'same', 'prompt_file': 'p'}
    path = manifest_file(tmp_path, [item, item])
    tokenizer = Tokenizer()
    with pytest.raises(ContractError):
        preflight_inputs(prompt_file=None, manifest=path, tokenizer=tokenizer, repo=Repository())
    assert tokenizer.calls == []


@pytest.mark.parametrize('length', [0, 1023, 8193])
def test_actual_length_outside_lut_domain(tmp_path, length):
    path = tmp_path / 'p'
    path.write_text('not approximated from characters')
    with pytest.raises(ContractError):
        preflight_inputs(prompt_file=path, manifest=None, tokenizer=Tokenizer(length), repo=Repository())


@pytest.mark.parametrize('length', [1024, 8192])
def test_actual_length_boundaries(tmp_path, length):
    path = tmp_path / 'p'
    path.write_text('short text')
    repo = Repository()
    prepared = preflight_inputs(prompt_file=path, manifest=None, tokenizer=Tokenizer(length), repo=repo)
    assert prepared.requests[0].spec.length == length and repo.lengths == [length]


def test_all_requests_lut_preflight_and_failure_propagates(tmp_path):
    (tmp_path / 'p').write_text('text')
    path = manifest_file(tmp_path, [{'request_id': str(i), 'prompt_file': 'p'} for i in range(3)])
    class BrokenRepository(Repository):
        def preflight(self, length):
            super().preflight(length)
            if len(self.lengths) == 3:
                raise ContractError('incomplete LUT')
    repo = BrokenRepository()
    with pytest.raises(ContractError, match='incomplete LUT'):
        preflight_inputs(prompt_file=None, manifest=path, tokenizer=Tokenizer(), repo=repo)
    assert repo.lengths == [1024] * 3


@pytest.mark.parametrize('which', ['prompt', 'manifest'])
def test_utf8_and_unreadable_input_rejected(tmp_path, which):
    path = tmp_path / 'bad'
    path.write_bytes(b'\xff')
    kwargs = {'prompt_file': path, 'manifest': None} if which == 'prompt' else {'prompt_file': None, 'manifest': path}
    with pytest.raises(ContractError):
        preflight_inputs(**kwargs, tokenizer=Tokenizer(), repo=Repository())
    path.unlink()
    with pytest.raises(ContractError):
        preflight_inputs(**kwargs, tokenizer=Tokenizer(), repo=Repository())


def test_exactly_one_input(tmp_path):
    for prompt, manifest in [(None, None), (tmp_path / 'p', tmp_path / 'm')]:
        with pytest.raises(ContractError):
            preflight_inputs(prompt_file=prompt, manifest=manifest, tokenizer=Tokenizer(), repo=Repository())


def test_modules_have_no_gpu_or_scientific_imports():
    code = "import sys; import kvolt_runtime.execution.inputs; import kvolt_runtime.recording; assert not any(m in sys.modules for m in ('torch','transformers','numpy','scipy','kneed'))"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')})
    assert result.returncode == 0, result.stderr


def test_tokenizer_mutation_cannot_change_frozen_prompt_or_token_list(tmp_path):
    prompt = tmp_path / 'p'
    original = b'  <|begin_of_text|>\r\noriginal '
    prompt.write_bytes(original)
    ids = [7] * 1024

    def tokenizer(text, **kwargs):
        assert text.encode('utf-8') == original
        assert kwargs == {'add_special_tokens': False}
        prompt.write_bytes(b'changed during tokenization')
        return {'input_ids': ids}

    prepared = preflight_inputs(prompt_file=prompt, manifest=None,
                                tokenizer=tokenizer, repo=Repository())
    ids[0] = 9
    request = prepared.requests[0]
    assert request.prompt.text.encode('utf-8') == original
    assert request.spec.prompt_sha256 == hashlib.sha256(original).hexdigest()
    assert request.spec.token_ids == (7,) * 1024
    assert request.spec.token_sha256 == hashlib.sha256(
        json.dumps([7] * 1024, separators=(',', ':')).encode()).hexdigest()


def test_manifest_mutation_during_preflight_preserves_original_and_metadata(tmp_path):
    (tmp_path / 'p').write_text('original')
    manifest = manifest_file(tmp_path, [
        {'request_id': 'r', 'prompt_file': 'p', 'metadata': {'nested': [1]}}])
    raw = manifest.read_bytes()

    class MutatingRepository(Repository):
        def preflight(self, length):
            super().preflight(length)
            manifest.write_text('{}')
            (tmp_path / 'p').unlink()

    prepared = preflight_inputs(prompt_file=None, manifest=manifest,
                                tokenizer=Tokenizer(), repo=MutatingRepository())
    assert prepared.manifest_raw == raw
    assert prepared.manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert prepared.requests[0].metadata == {'nested': [1]}
    assert prepared.requests[0].prompt.text == 'original'
