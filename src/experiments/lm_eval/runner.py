"""Evaluate the existing KIVI Llama model with pinned lm-eval tasks.

This module and --help are CPU-safe without the optional evaluation packages.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from . import LM_EVAL_VERSION

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name+" must be a positive integer")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return value.__module__+"."+value.__qualname__
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("unsupported result JSON value: "+type(value).__name__)


def _write(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2,
                  default=_json_default)
        stream.write("\n")


def _package_versions():
    result = {}
    for name in ("lm_eval", "datasets", "huggingface-hub", "torch", "transformers",
                 "flash-attn", "triton", "accelerate", "nltk", "wonderwords", "scipy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _evaluation_sources():
    """Bind the installed evaluator and referenced task families to exact bytes."""
    distribution = importlib.metadata.distribution("lm_eval")
    result = {}
    for entry in distribution.files or ():
        name = str(entry)
        if (name in ("lm_eval/evaluator.py", "lm_eval/utils.py") or
                name.startswith(("lm_eval/api/", "lm_eval/tasks/gsm8k/", "lm_eval/tasks/ruler/"))):
            if Path(name).suffix in (".py", ".yaml"):
                result[name] = hashlib.sha256(distribution.locate_file(entry).read_bytes()).hexdigest()
    return result


def run_evaluation(output_dir, *, tasks, model_name=MODEL_NAME, limit=None,
                   group_size=32, residual_length=128, max_new_tokens=512,
                   max_input_tokens=8192, measure_energy=False, power_profile=None,
                   seed=1234, ruler_lengths=(4096,), num_fewshot=None,
                   model=None, tokenizer=None, task_manager=None):
    """Run generation tasks with no response cache, batching or truncation.

    Preloaded model/tokenizer and task_manager allow network-free integration
    tests. That path is labelled separately and leaves hardware locking to the
    caller. Production loading owns the existing nonblocking hardware lock.
    Task data preparation and scoring are outside request energy intervals.
    """
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise ValueError("tasks must be a nonempty list of task names")
    task_names = []
    for task in tasks:
        name = task if isinstance(task, str) else getattr(getattr(task, "config", None), "task", None)
        if (not isinstance(name, str) or not name.strip() or
                (not isinstance(task, str) and model is None)):
            raise ValueError("tasks must contain names; task objects require a preloaded test model")
        task_names.append(name)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be nonempty")
    for name, value in (("group_size", group_size), ("residual_length", residual_length),
                        ("max_new_tokens", max_new_tokens), ("max_input_tokens", max_input_tokens)):
        _positive(value, name)
    if group_size not in (8, 16, 32, 64, 128) or residual_length % group_size:
        raise ValueError("unsupported KIVI group/residual combination")
    if limit is not None:
        _positive(limit, "limit")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if num_fewshot is not None and (type(num_fewshot) is not int or num_fewshot < 0):
        raise ValueError("num_fewshot must be a nonnegative integer or None")
    if type(measure_energy) is not bool:
        raise ValueError("measure_energy must be bool")
    if power_profile is not None and not measure_energy:
        raise ValueError("power_profile requires measure_energy")
    if (model is None) != (tokenizer is None):
        raise ValueError("supply model and tokenizer together")
    if not isinstance(ruler_lengths, (list, tuple)) or not ruler_lengths:
        raise ValueError("ruler_lengths must be a nonempty sequence")
    uses_ruler = any(task == "ruler" or task.startswith(("ruler_", "niah_")) for task in task_names)
    for length in ruler_lengths:
        _positive(length, "ruler_length")
        if uses_ruler and length not in (4096, 8192, 16384, 32768, 65536, 131072):
            raise ValueError("ruler_length is not a metric length in the pinned official task definitions")
        if uses_ruler and length > max_input_tokens:
            raise ValueError("ruler_length exceeds max_input_tokens")
    if "ruler" in task_names and 4096 not in ruler_lengths:
        raise ValueError("the pinned ruler group aggregates 4096 only; select individual tasks for other lengths")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("output directory already exists: "+str(output))
    versions = _package_versions()
    if versions["lm_eval"] != LM_EVAL_VERSION:
        raise RuntimeError("requires lm_eval=="+LM_EVAL_VERSION+
                           "; install the project's [lm-eval] extra in an isolated environment")
    profile = None
    if measure_energy:
        from kvolt_runtime.telemetry import PowerProfile, discover_power_profile
        profile = PowerProfile.from_json(power_profile) if power_profile else discover_power_profile()
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), Path(__file__).with_name("backend.py"),
                    source_root/"model/kivi_llama.py"]
    manifest = {
        "schema": "kvolt.lm-eval.run.v1", "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": task_names, "model_name": model_name, "limit": limit,
        "partial_evaluation": limit is not None, "versions": versions,
        "settings": dict(group_size=group_size, residual_length=residual_length,
                         k_bits=2, v_bits=2, max_new_tokens=max_new_tokens,
                         max_input_tokens=max_input_tokens, seed=seed,
                         ruler_lengths=list(ruler_lengths), num_fewshot=num_fewshot,
                         batch_size=1, response_cache=False, request_cache=False,
                         chat_template=False, input_truncation=False, warmup_requests=0,
                         ruler_group_aggregate_length=4096 if "ruler" in task_names else None),
        "preloaded_model": model is not None,
        "hardware_lock": "caller_owned_preloaded_model" if model is not None else
                         "/run/lock/kvolt-p3-hardware.lock",
        "energy_enabled": measure_energy,
        "power_profile": profile.to_dict() if profile else None,
        "source_sha256": {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "evaluator_and_reference_task_sha256": _evaluation_sources(),
        "protocol": "lm-eval task configuration; distinct from the historical bundled GSM8K/RULER protocols",
        "formal_qualification": False,
    }
    _write(output/"run.json", manifest)
    lm = None
    try:
        if model is None:
            from kvolt_runtime.device.lock import HardwareLock
            lock = HardwareLock()
        else:
            lock = nullcontext()
        # Acquire before importing evaluator/Torch in the production path.
        with lock:
            from lm_eval import evaluator
            from .backend import KIVILM
            lm = KIVILM(model_name=model_name, group_size=group_size,
                        residual_length=residual_length, max_new_tokens=max_new_tokens,
                        max_input_tokens=max_input_tokens, measure_energy=measure_energy,
                        power_profile=profile, output_dir=output, model=model, tokenizer=tokenizer)
            results = evaluator.simple_evaluate(
                model=lm, model_args={"tokenizer": model_name, "pretrained": model_name},
                tasks=list(tasks), task_manager=task_manager, num_fewshot=num_fewshot,
                limit=limit, batch_size=1, use_cache=None, cache_requests=False,
                apply_chat_template=False, fewshot_as_multiturn=False,
                log_samples=True, bootstrap_iters=0, random_seed=seed,
                numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed,
                metadata={"tokenizer": model_name, "pretrained": model_name,
                          "max_seq_lengths": list(ruler_lengths)},
            )
            if results is None:
                raise RuntimeError("lm-eval did not return single-process evaluation results")
            samples = results.pop("samples", {})
            with (output/"samples.jsonl").open("x", encoding="utf-8") as stream:
                for task, rows in samples.items():
                    for row in rows:
                        stream.write(json.dumps({"task": task, **row}, ensure_ascii=False,
                                                allow_nan=False, default=_json_default)+"\n")
            completed = [row for row in lm.records if row.get("status") == "completed"]
            energies = [row.get("energy") for row in completed]
            invalid = sum(not energy or not energy.get("valid") for energy in energies) if measure_energy else 0
            manifest.update(status="completed", requests=len(completed),
                            energy_invalid_count=invalid,
                            energy_valid_count=len(completed)-invalid if measure_energy else 0)
            manifest["model_class"] = type(lm.model).__module__+"."+type(lm.model).__qualname__
            manifest["resolved_task_configs_sha256"] = hashlib.sha256(json.dumps(
                results.get("configs", {}), sort_keys=True, ensure_ascii=False,
                default=_json_default).encode("utf-8")).hexdigest()
            _write(output/"results.json", results)
            manifest["artifacts_sha256"] = {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.rglob("*") if path.is_file() and path.name != "run.json"}
            _write(output/"run.json", manifest)
            return {"output_dir": str(output), "results": results.get("results", {}),
                    "requests": len(completed), "energy_invalid_count": invalid}
    except BaseException as original:
        manifest.update(status="failed", error={"type": type(original).__name__, "message": str(original)},
                        requests=len(lm.records) if lm is not None else 0)
        try:
            _write(output/"run.json", manifest)
        except BaseException as secondary:
            print("lm-eval failure manifest save failed: "+str(secondary), file=sys.stderr)
        raise


def _int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv=None, *, default_tasks=("gsm8k",)):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", "--task", nargs="+", default=list(default_tasks),
                        help="official lm-eval task names; e.g. gsm8k or niah_multiquery")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=_int)
    parser.add_argument("--group-size", type=_int, default=32)
    parser.add_argument("--residual-length", type=_int, default=128)
    parser.add_argument("--max-new-tokens", type=_int, default=512, help="hard cap; smaller task caps are retained")
    parser.add_argument("--max-input-tokens", type=_int, default=8192)
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--power-profile", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--ruler-lengths", nargs="+", type=_int, default=[4096])
    args = parser.parse_args(argv)
    output = args.output_dir or Path("artifacts/lm-eval")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        result = run_evaluation(output, tasks=args.tasks, model_name=args.model_name,
                                limit=args.limit, group_size=args.group_size,
                                residual_length=args.residual_length, max_new_tokens=args.max_new_tokens,
                                max_input_tokens=args.max_input_tokens, measure_energy=args.measure_energy,
                                power_profile=args.power_profile, seed=args.seed,
                                ruler_lengths=args.ruler_lengths, num_fewshot=args.num_fewshot)
    except Exception as error:
        print("lm-eval KIVI run failed: "+str(error)+"; partial results: "+str(output), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, default=_json_default))
    return 2 if result["energy_invalid_count"] else 0
