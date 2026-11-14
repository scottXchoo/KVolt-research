"""CPU-only validation of the pinned representative RULER legacy v1 bundle."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from experiments.quality.RULER import dataset

BUNDLE = Path(dataset.__file__).resolve().parent
TASKS = ('niah_single_1', 'niah_multikey_2', 'vt', 'fwe')


@pytest.fixture
def copied(tmp_path):
    target = tmp_path / 'dataset'
    shutil.copytree(BUNDLE, target, ignore=shutil.ignore_patterns('__pycache__'))
    return target


def test_public_contract_and_all_rows():
    assert dataset.TASK_NAMES == TASKS
    assert dataset.PINNED_REVISION == 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
    records = dataset.load_dataset()
    assert len(records) == 40
    assert [r['id'] for r in records] == [f'ruler/{task}/{i:04d}' for task in TASKS for i in range(10)]
    for record in records:
        assert set(record) == {'id', 'task', 'prompt', 'answers', 'max_new_tokens', 'source_index', 'source_length'}
        assert record['max_new_tokens'] == {'vt': 30, 'fwe': 50}.get(record['task'], 128)
        assert 0 < record['source_length'] <= 4096
        assert dataset.score(', '.join(record['answers']), record['answers']) == 1
        # Retrieval answers legitimately occur in the context; they must not be
        # appended as a separate gold completion to the task's answer prefix.
        assert not record['prompt'].endswith(', '.join(record['answers']))
    assert dataset.load_dataset() == records


def test_raw_upstream_fields_are_preserved():
    for task in TASKS:
        raw = [json.loads(line) for line in (BUNDLE / f'data/{task}.jsonl').read_text().splitlines()]
        selected = dataset.load_dataset(task=task)
        assert len(selected) == 10
        for prepared, original in zip(selected, raw):
            assert prepared['prompt'] == original['input'] + original['answer_prefix']
            assert prepared['answers'] == original['outputs']
            assert prepared['source_index'] == original['index']
            assert prepared['source_length'] == original['length']


def test_validation_report():
    report = dataset.validate_bundle()
    assert report['validated_prompts'] == 40
    assert report['task_rows'] == {task: 10 for task in TASKS}
    assert report['source_sha256'] == dataset.PINNED_SOURCE_SHA256
    assert report['protocol_sha256'] == dataset.PINNED_PROTOCOL_SHA256
    assert report['prompt_sha256'] == hashlib.sha256(''.join(r['prompt'] + '\n' for r in dataset.load_dataset()).encode()).hexdigest()
    assert report['source_length_tokenizer'] == 'cl100k_base'
    json.dumps(report)


@pytest.mark.parametrize('prediction,answers,expected', [
    ('ALPHA, beta, extra', ['alpha', 'BETA'], 1.),
    ('ALPHA', ['alpha', 'beta'], .5),
    ('largerword', ['word'], 1.),  # Official substring metric, no word boundary.
    ('nothing', ['alpha'], 0.),
    ('', ['alpha'], 0.),
    ('  ALPHA\x00BETA\tGAMMA  ', ['alpha\nbeta\ngamma'], 1.),
    ('ALPHA\x00BETA', ['alphabeta'], 0.),
])
def test_official_recall_and_prediction_cleanup(prediction, answers, expected):
    assert dataset.score(prediction, answers) == expected


@pytest.mark.parametrize('answers', [[], [''], [' '], ['ok', 1], 'alpha', None])
def test_invalid_references(answers):
    with pytest.raises((TypeError, ValueError)):
        dataset.score('alpha', answers)


def test_invalid_prediction_type():
    with pytest.raises(TypeError):
        dataset.score(None, ['alpha'])


@pytest.mark.parametrize('task', ['', 'qa_1', ['vt'], 2])
def test_invalid_task(task):
    with pytest.raises((TypeError, ValueError)):
        dataset.load_dataset(task=task)


@pytest.mark.parametrize('filename', ['source.json', 'protocol.json', 'LICENSE', *[f'data/{task}.jsonl' for task in TASKS]])
def test_missing_file(copied, filename):
    (copied / filename).unlink()
    with pytest.raises(ValueError, match='cannot read'):
        dataset.load_dataset(copied)


@pytest.mark.parametrize('filename', ['source.json', 'protocol.json', 'LICENSE', *[f'data/{task}.jsonl' for task in TASKS]])
def test_corrupt_bytes_rejected(copied, filename):
    path = copied / filename
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match='SHA-256'):
        dataset.load_dataset(copied)


def test_all_tasks_validated_before_filter(copied):
    (copied / 'data/fwe.jsonl').write_text('broken\n')
    with pytest.raises(ValueError):
        dataset.load_dataset(copied, task='vt')


def test_data_and_source_rehash_cannot_bypass_code_pin(copied):
    path = copied / 'data/vt.jsonl'
    raw = path.read_bytes().replace(b'green', b'black', 1)
    path.write_bytes(raw)
    source_path = copied / 'source.json'
    source = json.loads(source_path.read_text())
    source['files']['data/vt.jsonl']['sha256'] = hashlib.sha256(raw).hexdigest()
    source['files']['data/vt.jsonl']['bytes'] = len(raw)
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='source.json pin SHA-256'):
        dataset.load_dataset(copied)


def test_default_bundle_independent_of_cwd(tmp_path, monkeypatch):
    expected = dataset.load_dataset()
    monkeypatch.chdir(tmp_path)
    assert dataset.load_dataset() == expected


@pytest.mark.parametrize('entrypoint', ['direct', 'module'])
def test_validation_cli_from_other_cwd(tmp_path, entrypoint):
    command = [sys.executable, str(BUNDLE / 'dataset.py')] if entrypoint == 'direct' else [sys.executable, '-m', 'experiments.quality.RULER.dataset']
    env = dict(os.environ, PYTHONPATH=str(BUNDLE.parents[3]))
    result = subprocess.run(command + ['validate'], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['validated_prompts'] == 40


def test_cli_validation_failure(copied, tmp_path):
    (copied / 'protocol.json').unlink()
    result = subprocess.run([sys.executable, str(BUNDLE / 'dataset.py'), 'validate', '--bundle-dir', str(copied)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'RULER validation failed' in result.stderr


def test_import_is_stdlib_only(tmp_path):
    code = "import sys; from experiments.quality.RULER import dataset; dataset.validate_bundle(); assert not ({'torch','transformers','tiktoken','numpy','nltk','requests'} & set(sys.modules))"
    result = subprocess.run([sys.executable, '-S', '-c', code], cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(BUNDLE.parents[3])), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduction_requires_pinned_source(copied):
    from experiments.quality.RULER.dataset.reproduce import _verify_source
    (copied / 'source.json').unlink()
    with pytest.raises(ValueError, match='cannot read'):
        _verify_source(copied, copied)


def test_reproduction_rejects_changed_upstream(tmp_path):
    from experiments.quality.RULER.dataset.reproduce import _verify_source
    source = json.loads((BUNDLE / 'source.json').read_text())
    first = next(iter(source['upstream_files']))
    changed = tmp_path / first
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text('changed source')
    with pytest.raises(ValueError, match='upstream source SHA-256'):
        _verify_source(tmp_path, BUNDLE)


def test_reproduction_rejects_changed_generated_data():
    from experiments.quality.RULER.dataset.reproduce import _check_generated
    source = json.loads((BUNDLE / 'source.json').read_text())
    for task in TASKS:
        raw = (BUNDLE / f'data/{task}.jsonl').read_bytes()
        _check_generated(raw, task, source)
        with pytest.raises(ValueError, match='generated data SHA-256'):
            _check_generated(raw + b'\n', task, source)
