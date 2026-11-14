"""Offline, pinned representative RULER legacy v1 inputs and official recall.

Only standard-library text preparation and validation live here. Generation
requires the separate, transient environment documented in README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

PINNED_REVISION = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
PINNED_SOURCE_SHA256 = '1a53c16329b193ee2a4622c167373bd61d485be23b0afba964828eb8c3871777'
PINNED_PROTOCOL_SHA256 = '8add183f2212be00b38fd99e402a136ca582e58fdaf2006611606f6f8308154d'
TASK_NAMES = ('niah_single_1', 'niah_multikey_2', 'vt', 'fwe')
_BUNDLE_DIR = Path(__file__).resolve().parent
_TOKENS = {'niah_single_1': 128, 'niah_multikey_2': 128, 'vt': 30, 'fwe': 50}
_CONTROL_RE = re.compile(r'[\x00-\x1f]')


def score(prediction: str, answers: list[str]) -> float:
    """Return unrounded per-row official string_match_all recall in [0, 1].

    Apply upstream evaluate.py's postprocess_pred: strip, replace each ASCII
    control character with LF, strip. Compare case-insensitive substrings, with
    no token boundaries or precision penalty for extra guesses. Pass generated
    continuation only. Upstream's task-level percentage rounds only after the
    mean across examples; this per-example API does not round partial credit.
    """
    if not isinstance(prediction, str):
        raise TypeError('prediction must be a string')
    if not isinstance(answers, list):
        raise TypeError('answers must be a list of reference strings')
    if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError('answers must contain nonempty reference strings')
    cleaned = _CONTROL_RE.sub('\n', prediction.strip()).strip().lower()
    return sum(answer.lower() in cleaned for answer in answers) / len(answers)


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '\\' not in name and all(p not in ('', '.', '..') for p in name.split('/'))


def _read_bytes(bundle: Path, name: str) -> bytes:
    if not isinstance(name, str) or not _safe_name(name):
        raise ValueError('unsafe source metadata path')
    path = bundle / name
    if not path.resolve().is_relative_to(bundle.resolve()):
        raise ValueError(f'unsafe source metadata path: {name}')
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f'cannot read bundle file {name}: {exc}') from exc


def _json(raw: bytes, name: str) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid JSON in {name}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{name} must contain an object')
    return value


def _read_pinned_json(bundle: Path, name: str, digest: str) -> dict:
    raw = _read_bytes(bundle, name)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f'{name} pin SHA-256 mismatch')
    return _json(raw, name)


def _read_task(raw: bytes, task: str, expected_rows: int) -> list[dict]:
    try:
        lines = raw.decode('utf-8').splitlines()
    except UnicodeError as exc:
        raise ValueError(f'{task} must be UTF-8') from exc
    if len(lines) != expected_rows:
        raise ValueError(f'{task} row count mismatch')
    rows = []
    required = {'index', 'input', 'outputs', 'length', 'length_w_model_temp', 'answer_prefix'}
    fields = required | ({'token_position_answer'} if task.startswith('niah_') else set())
    for ordinal, line in enumerate(lines):
        row = _json(line.encode('utf-8'), f'{task} row {ordinal}')
        if set(row) != fields:
            raise ValueError(f'invalid {task} fields at row {ordinal}')
        for field in ('index', 'length', 'length_w_model_temp', *(['token_position_answer'] if task.startswith('niah_') else [])):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f'invalid {task} {field} at row {ordinal}')
        if not 0 < row['length'] <= 4096 or row['length_w_model_temp'] != row['length']:
            raise ValueError(f'invalid {task} source length at row {ordinal}')
        if any(not isinstance(row[f], str) or not row[f].strip() for f in ('input', 'answer_prefix')):
            raise ValueError(f'invalid {task} prompt text at row {ordinal}')
        score('', row['outputs'])  # Validate references without changing bytes.
        expected_answers = 5 if task == 'vt' else 3 if task == 'fwe' else 1
        if len(row['outputs']) != expected_answers or len(set(row['outputs'])) != expected_answers:
            raise ValueError(f'invalid {task} reference count at row {ordinal}')
        rows.append(row)
    return rows


def _prepare(bundle_dir: str | Path | None) -> tuple[list[dict], dict]:
    bundle = _BUNDLE_DIR if bundle_dir is None else Path(bundle_dir).resolve()
    source = _read_pinned_json(bundle, 'source.json', PINNED_SOURCE_SHA256)
    protocol = _read_pinned_json(bundle, 'protocol.json', PINNED_PROTOCOL_SHA256)
    if source.get('revision') != PINNED_REVISION or protocol.get('task_names') != list(TASK_NAMES):
        raise ValueError('source revision or protocol task pin mismatch')
    expected = {'LICENSE', 'generation-requirements.txt', *[f'data/{task}.jsonl' for task in TASK_NAMES]}
    if set(source.get('files', {})) != expected:
        raise ValueError('source file inventory mismatch')
    original = {}
    # Validate ALL bytes, rows and fields before allowing a task filter.
    for name, metadata in source['files'].items():
        raw = _read_bytes(bundle, name)
        if len(raw) != metadata['bytes']:
            raise ValueError(f'{name} byte count mismatch')
        if hashlib.sha256(raw).hexdigest() != metadata['sha256']:
            raise ValueError(f'{name} SHA-256 mismatch')
        if name.startswith('data/'):
            task = Path(name).stem
            if metadata['task'] != task or metadata['rows'] != 10:
                raise ValueError(f'{name} task or row pin mismatch')
            original[task] = _read_task(raw, task, metadata['rows'])
    records = []
    prompt_digest = hashlib.sha256()
    for task in TASK_NAMES:
        for ordinal, row in enumerate(original[task]):
            prompt = row['input'] + row['answer_prefix']
            records.append({
                'id': f'ruler/{task}/{ordinal:04d}', 'task': task,
                'prompt': prompt, 'answers': list(row['outputs']),
                'max_new_tokens': _TOKENS[task],
                # NIAH upstream index is a character offset, not row ordinal.
                'source_index': row['index'], 'source_length': row['length'],
            })
            prompt_digest.update((prompt + '\n').encode('utf-8'))
    report = {
        'schema': 'kvolt.ruler-validation.v1', 'revision': PINNED_REVISION,
        'scope': 'representative legacy v1 subset; not formal 13-task evaluation',
        'validated_prompts': len(records),
        'task_rows': {task: len(original[task]) for task in TASK_NAMES},
        'source_sha256': PINNED_SOURCE_SHA256,
        'protocol_sha256': PINNED_PROTOCOL_SHA256,
        'prompt_sha256': prompt_digest.hexdigest(),
        'prompt_sha256_encoding': 'UTF-8 concatenation of each prompt followed by LF, in task/file order',
        'source_length_tokenizer': 'cl100k_base',
        'source_length_includes_output_reservation': True,
        'actual_model_input_lengths_measured': False,
    }
    return records, report


def load_dataset(bundle_dir: str | Path | None = None, task: str | None = None) -> list[dict]:
    """Validate every pinned file, then return all rows or a selected task.

    Default bundle location is independent of cwd. source_length includes the
    upstream cl100k output reservation; runners must measure their own actual
    model-tokenizer input lengths and must never treat this value as Llama L.
    """
    if task is not None and not isinstance(task, str):
        raise TypeError('task must be a string or None')
    if task is not None and task not in TASK_NAMES:
        raise ValueError(f'unknown RULER task: {task}')
    records, _ = _prepare(bundle_dir)
    return records if task is None else [row for row in records if row['task'] == task]


def validate_bundle(bundle_dir: str | Path | None = None) -> dict:
    """Check pinned metadata, every original file and every prepared prompt."""
    _, report = _prepare(bundle_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate'])
    parser.add_argument('--bundle-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_bundle(args.bundle_dir)
    except (ValueError, TypeError) as exc:
        print(f'RULER validation failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
