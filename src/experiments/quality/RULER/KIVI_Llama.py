"""Run a pinned four-task RULER subset with KIVI Llama Instruct."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
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
from kvolt_runtime.telemetry import EnergyMeter, PowerProfile, discover_power_profile


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
KV_SETTINGS = {"k_bits": 2, "v_bits": 2, "group_size": 32, "residual_length": 128}
ENERGY_BOUNDARY = (
    "Synchronized call to model.kivi_llama.generate: includes its tokenization, "
    "host-to-device input transfer, model.generate, and continuation text decoding; "
    "excludes model load, preflight tokenization/input hash/context checks, warmup, "
    "scoring, result serialization, and file I/O. CUDA synchronization uses model.device "
    "immediately before the start and end timestamps. Collector startup, right-bracket "
    "waiting, cleanup, and artifact saving are outside the interval."
)


def _diagnostic_error(original: BaseException, stage: str, error: BaseException) -> None:
    """Report secondary persistence errors without replacing a workload failure."""
    message = f"RULER {stage} failed: {type(error).__name__}: {error}"
    if hasattr(original, "add_note"):
        original.add_note(message)
    print(message, file=sys.stderr)


def _measured_generate(model, tokenizer, record, profile, output, ordinal):
    # Import CUDA only in the opt-in path, after argument/profile validation and
    # model loading. The shared helper remains the sole owner of generation.
    import torch

    relative_dir = f"energy/{ordinal:06d}"
    meter = EnergyMeter(profile, synchronize=lambda: torch.cuda.synchronize(model.device))
    try:
        with meter:
            prediction = kivi_llama.generate(
                model, tokenizer, record["prompt"], max_new_tokens=record["max_new_tokens"],
            )
    except BaseException as original:
        saved = False
        try:
            meter.save(output / relative_dir)
            saved = True
        except BaseException as error:
            _diagnostic_error(original, "failed-measurement artifact save", error)
        try:
            failure = {
                "id": record["id"], "task": record["task"], "ordinal": ordinal,
                "error": {"type": type(original).__name__, "message": str(original)},
                "energy": meter.result, "energy_artifacts_dir": relative_dir,
                "energy_artifacts_saved": saved,
                "boundary": ENERGY_BOUNDARY, "power_profile": profile.to_dict(),
            }
            with (output / "energy-failures.jsonl").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, ensure_ascii=False, allow_nan=False) + "\n")
        except BaseException as error:
            _diagnostic_error(original, "failed-measurement record save", error)
        raise
    meter.save(output / relative_dir)
    return prediction, meter.result, relative_dir


def run(
    output_dir: str | Path, *, limit: int | None = None,
    model_name: str = MODEL_NAME, task: str | None = None,
    measure_energy: bool = False, power_profile: str | Path | None = None,
    warmup_per_task: int = 0,
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
    if type(measure_energy) is not bool:
        raise ValueError("measure_energy must be bool")
    if power_profile is not None and not measure_energy:
        raise ValueError("power_profile requires measure_energy")
    if type(warmup_per_task) is not int or warmup_per_task < 0:
        raise ValueError("warmup_per_task must be a nonnegative integer")
    records = load_dataset()  # Validate every task, including unselected files.
    filtered = records if task is None else [row for row in records if row["task"] == task]
    if not filtered:
        raise ValueError("selected task has no dataset rows")
    if limit is not None and limit > len(filtered):
        raise ValueError(f"limit must not exceed {len(filtered)} selected task rows")
    selected = filtered if limit is None else filtered[:limit]
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    # Read-only preflight happens once, before creating output or loading CUDA.
    # Discovery never supplies guessed numeric sensor values on failure.
    profile = None
    if measure_energy:
        profile = (PowerProfile.from_json(power_profile) if power_profile is not None
                   else discover_power_profile())
    output.mkdir(parents=True, exist_ok=False)

    scores_by_task: dict[str, list[float]] = {}
    energy_results = []
    warmed_tasks = []
    completed_warmups = 0
    warmup_stream = ((output / "warmup.jsonl").open("x", encoding="utf-8")
                     if warmup_per_task else nullcontext(None))
    with warmup_stream as warmups, (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        model, tokenizer = kivi_llama.load_model(model_name, **KV_SETTINGS)
        # The shared helper decodes a single token tensor, independent of a
        # checkpoint's default generation configuration.
        model.generation_config.num_return_sequences = 1
        model.generation_config.return_dict_in_generate = False
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if type(context_limit) is not int or context_limit <= 0:
            raise ValueError("model max_position_embeddings must be a positive integer")
        # Model Inference
        for ordinal, record in enumerate(selected):
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
            input_ids_sha256 = hashlib.sha256(
                json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if warmup_per_task and record["task"] not in warmed_tasks:
                for repetition in range(1, warmup_per_task + 1):
                    warmup_prediction = kivi_llama.generate(
                        model, tokenizer, record["prompt"], max_new_tokens=record["max_new_tokens"],
                    )
                    warmup = {
                        "id": record["id"], "task": record["task"], "repetition": repetition,
                        "prediction": warmup_prediction, "max_new_tokens": record["max_new_tokens"],
                    }
                    warmups.write(json.dumps(warmup, ensure_ascii=False, allow_nan=False) + "\n")
                    warmups.flush()
                    completed_warmups += 1
                warmed_tasks.append(record["task"])
            energy = artifact_dir = None
            if measure_energy:
                prediction, energy, artifact_dir = _measured_generate(
                    model, tokenizer, record, profile, output, ordinal,
                )
            else:
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
                "input_ids_sha256": input_ids_sha256,
                "max_new_tokens": record["max_new_tokens"],
                "energy": energy, "energy_artifacts_dir": artifact_dir,
            }
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            scores_by_task.setdefault(record["task"], []).append(result["score"])
            if measure_energy:
                energy_results.append(energy)

    mean_score = sum(sum(values) for values in scores_by_task.values()) / len(selected)
    valid_energy = [result["energy_j"] for result in energy_results if result["valid"]]
    valid_rows_energy_j = math.fsum(valid_energy) if measure_energy else None
    if valid_rows_energy_j is not None and not math.isfinite(valid_rows_energy_j):
        raise ValueError("aggregate energy must be finite")
    invalid_count = len(energy_results) - len(valid_energy)
    summary = {
        "schema": "kvolt.ruler-kivi-results.v1",
        "evaluated": len(selected), "mean_score": mean_score,
        "score_percent": round(mean_score * 100, 2),
        "aggregation": "micro mean of per-row official substring recall",
        "energy": {
            "enabled": measure_energy, "valid_count": len(valid_energy),
            "invalid_count": invalid_count,
            "total_energy_j": valid_rows_energy_j if measure_energy and not invalid_count else None,
            "valid_rows_energy_j": valid_rows_energy_j,
            "boundary": ENERGY_BOUNDARY if measure_energy else None,
            "power_profile": profile.to_dict() if profile is not None else None,
            "sensor_response_validated": False,
        },
        "warmup": {
            "per_task": warmup_per_task, "completed": completed_warmups, "tasks": warmed_tasks,
            "selection": "first selected record of each selected task; original output cap",
        },
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
        stream.write(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    return summary


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("warmup-per-task must be a nonnegative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("warmup-per-task must be a nonnegative integer")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, help="one task; default: all four representative tasks")
    parser.add_argument("--limit", type=_positive_int, help="first N rows after task selection; default: all selected rows")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Instruct model ID or local path; default: %(default)s")
    parser.add_argument("--output-dir", type=Path, help="new directory; must not already exist")
    parser.add_argument("--measure-energy", action="store_true", help="measure each full generate call with INA3221")
    parser.add_argument("--power-profile", type=Path, help="explicit PowerProfile JSON; requires --measure-energy")
    parser.add_argument("--warmup-per-task", type=_nonnegative_int, default=0,
                        help="unmeasured full-cap warmups on each selected task's first record (default: 0)")
    args = parser.parse_args(argv)
    if args.power_profile is not None and not args.measure_energy:
        parser.error("--power-profile requires --measure-energy")
    output = args.output_dir or Path("artifacts/ruler") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        summary = run(output, limit=args.limit, model_name=args.model_name, task=args.task,
                      measure_energy=args.measure_energy, power_profile=args.power_profile,
                      warmup_per_task=args.warmup_per_task)
    except Exception as exc:
        print(f"RULER run failed: {exc}; partial results, if any: {output}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(output), **summary}, ensure_ascii=False, indent=2))
    return 2 if summary["energy"]["invalid_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
