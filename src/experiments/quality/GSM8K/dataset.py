"""Offline GSM8K main/test inputs for a fixed 5-shot CoT greedy baseline.

This module only prepares and scores text. It does not import an inference
backend, select a model, apply a chat template, or run generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


PINNED_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
PINNED_SOURCE_SHA256 = "9f31829434a4713b4a27bca61159999668004bddc7bde844b0766c824d06f7cc"
PINNED_PROTOCOL_SHA256 = "6dd960f1a7eb2076db75bb98cc7d654a25a65837e3d515e5a5d39342d537ca40"
_BUNDLE_DIR = Path(__file__).resolve().parent
# Keep the upstream extraction grammar and first-match behavior verbatim.
_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_NORMALIZED_ANSWER_RE = re.compile(r"-?[0-9.]+")


def extract_answer(completion: str) -> str | None:
    """Return upstream's first #### numeric match with commas removed.

    No arithmetic normalization is performed: ``18.0`` differs from ``18``.
    Missing markers return None rather than upstream's string sentinel.
    """
    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    match = _ANSWER_RE.search(completion)
    return match.group(1).strip().replace(",", "") if match else None


def score(prediction: str, answer: str) -> float:
    """Score generated continuation against a normalized answer or raw solution.

    ``prediction`` must contain the original ``#### `` marker. Pass only the
    newly generated continuation, never the full prompt with its demonstrations.
    The reference may be load_dataset's normalized final number or an unmodified
    upstream solution. Invalid references raise ValueError; missing prediction
    markers score 0.0. The first marker wins, as in the upstream scorer.
    """
    if not isinstance(answer, str):
        raise TypeError("reference answer must be a string")
    reference = extract_answer(answer)
    if reference is None:
        if not _NORMALIZED_ANSWER_RE.fullmatch(answer):
            raise ValueError("reference answer must be a final numeric string or #### solution")
        reference = answer
    return float(extract_answer(prediction) == reference)


def _read_bytes(bundle: Path, name: str) -> bytes:
    try:
        return (bundle / name).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read bundle file {name}: {exc}") from exc


def _read_pinned_json(bundle: Path, name: str, digest: str) -> dict:
    raw = _read_bytes(bundle, name)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{name} pin SHA-256 mismatch")
    try:
        result = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {name}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must contain an object")
    return result


def _read_split(raw: bytes, split: str, expected_rows: int) -> list[dict[str, str]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError(f"{split} must be UTF-8") from exc
    if len(lines) != expected_rows:
        raise ValueError(f"{split} row count mismatch: {len(lines)} != {expected_rows}")
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {split} JSONL row {index}") from exc
        if not isinstance(row, dict) or set(row) != {"question", "answer"}:
            raise ValueError(f"invalid {split} fields at row {index}")
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError(f"invalid {split} text at row {index}")
        if extract_answer(row["answer"]) is None:
            raise ValueError(f"invalid {split} reference answer at row {index}")
        rows.append(row)
    return rows


def _prepare(bundle_dir: str | Path | None) -> tuple[list[dict[str, str]], dict]:
    bundle = _BUNDLE_DIR if bundle_dir is None else Path(bundle_dir).resolve()
    # The source/protocol pins are code constants: editing the manifest together
    # with a corrupted data file cannot silently authorize a different dataset.
    source = _read_pinned_json(bundle, "source.json", PINNED_SOURCE_SHA256)
    protocol = _read_pinned_json(bundle, "protocol.json", PINNED_PROTOCOL_SHA256)
    splits = {}
    for name, metadata in source["files"].items():
        raw = _read_bytes(bundle, name)
        if len(raw) != metadata["bytes"]:
            raise ValueError(f"{name} byte count mismatch")
        if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch")
        if "rows" in metadata:
            split = Path(name).stem
            if metadata["upstream_path"] != f"grade_school_math/data/{split}.jsonl":
                raise ValueError(f"{name} split pin mismatch")
            splits[split] = _read_split(raw, split, metadata["rows"])
    train, test = splits["train"], splits["test"]
    if {row["question"] for row in train} & {row["question"] for row in test}:
        raise ValueError("train/test question overlap")
    prompt_format = protocol["prompt"]
    separator = prompt_format["separator"]
    demonstrations = [
        prompt_format["example_template"].format(
            question=train[index]["question"], reference_solution=train[index]["answer"]
        )
        for index in protocol["few_shot"]["indices"]
    ]
    prefix = separator.join([prompt_format["instruction"], *demonstrations])
    records = []
    prompt_digest = hashlib.sha256()
    for index, row in enumerate(test):
        target = prompt_format["target_template"].format(question=row["question"])
        prompt = prefix + separator + target
        if not prompt.endswith("\nAnswer:") or row["answer"] in prompt:
            raise ValueError(f"invalid prompt/gold separation at test row {index}")
        answer = extract_answer(row["answer"])
        records.append({
            "id": protocol["evaluation"]["id_template"].format(index=index),
            "question": row["question"],
            "prompt": prompt,
            "answer": answer,
            "reference_solution": row["answer"],
        })
        prompt_digest.update((prompt + "\n").encode("utf-8"))
    report = {
        "schema": "kvolt.gsm8k-validation.v1",
        "revision": source["revision"],
        "train_rows": len(train),
        "test_rows": len(test),
        "validated_prompts": len(records),
        "source_sha256": PINNED_SOURCE_SHA256,
        "protocol_sha256": PINNED_PROTOCOL_SHA256,
        "prompt_sha256": prompt_digest.hexdigest(),
        "prompt_sha256_encoding": "UTF-8 concatenation of each prompt followed by LF, in test order",
    }
    return records, report


def load_dataset(bundle_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Validate all original bytes and return all 1,319 main/test records.

    The default bundle resolves against this module, independent of cwd.
    References remain separate from each target's complete 5-shot prompt.
    """
    records, _ = _prepare(bundle_dir)
    return records


def validate_bundle(bundle_dir: str | Path | None = None) -> dict:
    """Check pinned metadata/data, every row, split isolation, and all prompts."""
    _, report = _prepare(bundle_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("--bundle-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = validate_bundle(args.bundle_dir)
    except (ValueError, TypeError) as exc:
        print(f"GSM8K validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
