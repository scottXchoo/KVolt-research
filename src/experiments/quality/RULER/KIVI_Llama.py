"""Run a pinned four-task RULER subset with KIVI Llama Instruct."""

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

from experiments.quality.RULER.dataset import (
    TASK_NAMES, PINNED_PROTOCOL_SHA256, PINNED_REVISION, PINNED_SOURCE_SHA256,
    load_dataset, score,
)
from model import kivi_llama


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
KV_SETTINGS = {"k_bits": 2, "v_bits": 2, "group_size": 32, "residual_length": 128}


def run(
    output_dir: str | Path, *, limit: int | None = None,
    model_name: str = MODEL_NAME, task: str | None = None,
) -> dict:
    """Generate sequentially and retain flushed partial JSONL on failure.

    Validate arguments, the complete pinned bundle, and output collision before
    loading the model. ``limit`` selects the first N rows after task filtering.
    """
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty model ID or local path")
    if task is not None and (not isinstance(task, str) or task not in TASK_NAMES):
        raise ValueError(f"task must be one of {TASK_NAMES}")
    records = load_dataset()  # Validate every task, including unselected files.
    filtered = records if task is None else [row for row in records if row["task"] == task]
    if not filtered:
        raise ValueError("selected task has no dataset rows")
    if limit is not None and limit > len(filtered):
        raise ValueError(f"limit must not exceed {len(filtered)} selected task rows")
    selected = filtered if limit is None else filtered[:limit]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    scores_by_task: dict[str, list[float]] = {}
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        model, tokenizer = kivi_llama.load_model(model_name, **KV_SETTINGS)
        # The shared helper decodes a single token tensor, independent of a
        # checkpoint's default generation configuration.
        model.generation_config.num_return_sequences = 1
        model.generation_config.return_dict_in_generate = False
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if type(context_limit) is not int or context_limit <= 0:
            raise ValueError("model max_position_embeddings must be a positive integer")
        # Model Inference
        for record in selected:
            # Match the shared helper's tokenizer invocation exactly. Special
            # token defaults are owned by the tokenizer; do not prepend BOS or
            # apply a chat template. Never truncate an over-budget prompt.
            input_ids = tokenizer(record["prompt"], return_tensors="pt")["input_ids"].tolist()[0]
            input_tokens = len(input_ids)
            if input_tokens + record["max_new_tokens"] > context_limit:
                raise ValueError(
                    f"{record['id']}: context budget exceeded: input {input_tokens} + "
                    f"output {record['max_new_tokens']} > max_position_embeddings {context_limit}"
                )
            prediction = kivi_llama.generate(
                model, tokenizer, record["prompt"], max_new_tokens=record["max_new_tokens"],
            )
            result = {
                "id": record["id"], "task": record["task"],
                "prediction": prediction, "answers": record["answers"],
                "score": score(prediction, record["answers"]),
                "source_index": record["source_index"], "source_length": record["source_length"],
                "source_length_tokenizer": "cl100k_base (input + reserved output)",
                "input_tokenizer": model_name,
                "input_tokens": input_tokens,
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "max_new_tokens": record["max_new_tokens"],
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            scores_by_task.setdefault(record["task"], []).append(result["score"])

    mean_score = sum(sum(values) for values in scores_by_task.values()) / len(selected)
    summary = {
        "schema": "kvolt.ruler-kivi-results.v1",
        "evaluated": len(selected), "mean_score": mean_score,
        "score_percent": round(mean_score * 100, 2),
        "aggregation": "micro mean of per-row official substring recall",
        "per_task": {
            name: {"evaluated": len(values), "mean_score": sum(values) / len(values),
                   "score_percent": round(sum(values) / len(values) * 100, 2)}
            for name, values in scores_by_task.items()
        },
        "total_dataset_rows": len(records), "total_selected_task_rows": len(filtered),
        "subset": len(selected) < len(records),
        "first_id": selected[0]["id"], "last_id": selected[-1]["id"],
        "input_scope": {
            "benchmark": "RULER legacy v1 representative four-task subset",
            "full_benchmark": False, "available_tasks": list(TASK_NAMES),
            "selected_task": task,
            "evaluated_tasks": list(scores_by_task),
            "source_max_seq_length": 4096, "source_tokenizer": "cl100k_base",
            "source_length_semantics": "upstream input token count + reserved output tokens; not Llama input length",
        },
        "settings": {
            "model_name": model_name, **KV_SETTINGS,
            "max_new_tokens_by_task": {row["task"]: row["max_new_tokens"] for row in filtered},
            "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
            "use_cache": True, "chat_template": None,
            "context_limit": context_limit,
            "input_tokenizer": "generation model tokenizer; tokenizer(prompt, return_tensors='pt') defaults",
            "input_ids_sha256_encoding": "UTF-8 compact JSON integer list, separators=(',', ':')",
        },
        "dataset": {
            "revision": PINNED_REVISION, "source_sha256": PINNED_SOURCE_SHA256,
            "protocol_sha256": PINNED_PROTOCOL_SHA256,
            "selected_prompt_sha256": hashlib.sha256(
                "".join(record["prompt"] + "\n" for record in selected).encode("utf-8")
            ).hexdigest(),
            "prompt_sha256_encoding": "UTF-8 prompts followed by LF, in selected source order",
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
    parser.add_argument("--task", choices=TASK_NAMES, help="one task; default: all four representative tasks")
    parser.add_argument("--limit", type=_positive_int, help="first N rows after task selection; default: all selected rows")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Instruct model ID or local path; default: %(default)s")
    parser.add_argument("--output-dir", type=Path, help="new directory; must not already exist")
    args = parser.parse_args(argv)
    output = args.output_dir or Path("artifacts/ruler") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        summary = run(output, limit=args.limit, model_name=args.model_name, task=args.task)
    except Exception as exc:
        print(f"RULER run failed: {exc}; partial results, if any: {output}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
