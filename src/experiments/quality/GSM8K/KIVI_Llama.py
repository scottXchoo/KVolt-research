"""Run the pinned GSM8K 5-shot completion baseline with KIVI Llama Instruct."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

# Support direct script execution from any cwd, as well as PYTHONPATH=src -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.quality.GSM8K.dataset import (
    PINNED_PROTOCOL_SHA256, PINNED_REVISION, PINNED_SOURCE_SHA256,
    extract_answer, load_dataset, score,
)
from model import kivi_llama


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
KV_SETTINGS = {"k_bits": 2, "v_bits": 2, "group_size": 32, "residual_length": 128}
MAX_NEW_TOKENS = 512


def run(output_dir: str | Path, *, limit: int | None = None, model_name: str = MODEL_NAME) -> dict:
    """Generate sequentially; preserve partial JSONL if any request fails.

    The output directory must be new. No model/GPU setup occurs before argument,
    complete pinned-data validation, and output-directory collision checks.
    """
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty model ID or local path")
    records = load_dataset()
    total = len(records)
    if limit is not None and limit > total:
        raise ValueError(f"limit must not exceed {total}")
    selected = records if limit is None else records[:limit]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    correct = 0
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        model, tokenizer = kivi_llama.load_model(model_name, **KV_SETTINGS)
        # The shared helper expects a single token tensor, even if a checkpoint
        # carries different generation defaults. Keep the pinned one-output flow.
        model.generation_config.num_return_sequences = 1
        model.generation_config.return_dict_in_generate = False
        # Model Inference
        for record in selected:
            prediction = kivi_llama.generate(
                model, tokenizer, record["prompt"], max_new_tokens=MAX_NEW_TOKENS,
            )
            result = {
                "id": record["id"], "question": record["question"],
                "prediction": prediction, "predicted_answer": extract_answer(prediction),
                "answer": record["answer"], "score": score(prediction, record["answer"]),
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            correct += int(result["score"])

    summary = {
        "schema": "kvolt.gsm8k-kivi-results.v1",
        "evaluated": len(selected), "correct": correct, "accuracy": correct / len(selected),
        "total_test_rows": total, "subset": len(selected) < total,
        "first_id": selected[0]["id"], "last_id": selected[-1]["id"],
        "settings": {
            "model_name": model_name, **KV_SETTINGS, "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
            "use_cache": True, "chat_template": None,
        },
        "dataset": {
            "revision": PINNED_REVISION, "source_sha256": PINNED_SOURCE_SHA256,
            "protocol_sha256": PINNED_PROTOCOL_SHA256,
            "selected_prompt_sha256": hashlib.sha256(
                "".join(record["prompt"] + "\n" for record in selected).encode("utf-8")
            ).hexdigest(),
            "prompt_sha256_encoding": "UTF-8 prompts followed by LF, in selected test order",
        },
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    return summary


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=_positive_int, help="first N test rows; default: all 1319")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Instruct model ID or local path; default: %(default)s")
    parser.add_argument("--output-dir", type=Path, help="new directory; must not already exist")
    args = parser.parse_args(argv)
    output = args.output_dir or Path("artifacts/gsm8k") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    try:
        summary = run(output, limit=args.limit, model_name=args.model_name)
    except Exception as exc:
        print(f"GSM8K run failed: {exc}; partial results, if any: {output}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(output), **summary}, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
