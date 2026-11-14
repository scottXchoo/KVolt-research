"""Regenerate the representative bundle with an unmodified pinned RULER checkout.

Run this script using the isolated generation environment described in README.
No model weights, external corpora, or inference backend are required.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .dataset import PINNED_SOURCE_SHA256, TASK_NAMES, _read_pinned_json
else:
    from dataset import PINNED_SOURCE_SHA256, TASK_NAMES, _read_pinned_json


def _verify_source(upstream_dir: Path, bundle_dir: Path) -> dict:
    source = _read_pinned_json(bundle_dir, 'source.json', PINNED_SOURCE_SHA256)
    for name, expected in source['upstream_files'].items():
        raw = (upstream_dir / name).read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError(f'upstream source SHA-256 mismatch: {name}')
    return source


def _check_generated(raw: bytes, task: str, source: dict) -> None:
    expected = source['files'][f'data/{task}.jsonl']
    if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
        raise ValueError(f'generated data SHA-256 mismatch: {task}')
    if len(raw.splitlines()) != expected['rows']:
        raise ValueError(f'generated row count mismatch: {task}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = _verify_source(args.upstream_dir, Path(__file__).resolve().parent)
    env = dict(os.environ)
    # Upstream prepare.py launches its children with the literal command python.
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')

    for task in TASK_NAMES:
        destination = args.output_dir / task / 'validation.jsonl'
        if destination.exists():
            raise ValueError(f'refusing to reuse existing output: {destination}')
        command = [
            sys.executable, str(args.upstream_dir / 'scripts/data/prepare.py'),
            '--save_dir', str(args.output_dir.resolve()),
            '--benchmark', 'synthetic', '--task', task,
            '--tokenizer_path', 'cl100k_base', '--tokenizer_type', 'openai',
            '--max_seq_length', '4096', '--num_samples', '10',
            '--random_seed', '42', '--model_template_type', 'base',
        ]
        subprocess.run(command, env=env, check=True)
        # Upstream prepare.py catches child errors without a nonzero exit code.
        _check_generated(destination.read_bytes(), task, source)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
