"""Existing-path FP16/K2V2 pilot adapter. Imports perform no CUDA or sysfs work.

This deliberately reuses the audited resident precision model and patched KIVI
kernels. It is not the KVolt PC/DC runtime: KIVI packs the initial cache inside
the prefill forward, before t1, even when the first generated token is EOS.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

EOS_IDS = (128001, 128008, 128009)
RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
ENERGY_SCOPE = "INA3221 three independent module rails: VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0"
SCHEDULE_SEED = 20261103
CONDITIONS = ({"name": "fp16", "method": "fp16", "group": None,
               "k_bits": None, "v_bits": None, "residual_length": None},) + tuple(
    {"name": "k2v2", "method": "kivi", "group": g,
     "k_bits": 2, "v_bits": 2, "residual_length": 128} for g in (8, 16, 32, 64, 128))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_schedule(items, seed=SCHEDULE_SEED):
    """Shuffle item blocks, then independently shuffle settings in each block."""
    generator = random.Random(seed)
    blocks = list(items)
    generator.shuffle(blocks)
    result = []
    for item in blocks:
        settings = list(CONDITIONS)
        # Separate per-item streams keep each block's randomization independent.
        random.Random(digest([seed, item["id"]])).shuffle(settings)
        result.extend((item, setting) for setting in settings)
    return result


def wait_temperature(reader, *, minimum_mc=35000, timeout_s=900, poll_s=1,
                     clock_ns=time.monotonic_ns, sleep=time.sleep):
    """Wait without heating or clock changes; record admission outside t0.

    Formal admission requires both sensors in the inclusive 35..55 C range.
    Pre-warmup callers may pass minimum_mc=None, allowing natural model work to
    warm an idle device while retaining the upper bound and fresh sensor checks.
    """
    start = clock_ns()
    result = {"passed": False, "minimum_mc": minimum_mc, "maximum_mc": 55000,
              "timeout_s": timeout_s, "poll_s": poll_s, "start_ns": start,
              "observations": [], "forced_heating": False}
    while True:
        now = clock_ns()
        observation = {"observed_ns": now}
        try:
            sample = reader()
            now = clock_ns()
            observation["observed_ns"] = now
            age_s = (now - sample["t_ns"]) / 1e9
            values = {label: sample.get("temperatures_mc", {}).get(label)
                      for label in ("cpu-thermal", "gpu-thermal")}
            observation.update(sample_ns=sample["t_ns"], age_s=age_s, temperatures_mc=values)
            observation["admitted"] = bool(0 <= age_s <= 2 and all(
                type(value) in (int, float) and (minimum_mc is None or value >= minimum_mc)
                and value <= 55000 for value in values.values()))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            observation.update(admitted=False, error=f"{type(exc).__name__}: {exc}")
        result["observations"].append(observation)
        if observation["admitted"]:
            result["passed"] = True
            break
        elapsed = (clock_ns() - start) / 1e9
        if elapsed >= timeout_s:
            result["reason"] = "temperature_admission_timeout"
            break
        sleep(min(poll_s, timeout_s - elapsed))
    result["end_ns"] = clock_ns()
    result["wait_s"] = (result["end_ns"] - start) / 1e9
    return result


def wait_power_coverage(logger, end_ns, *, timeout_s=2, poll_s=0.02,
                        clock_ns=time.monotonic_ns, sleep=time.sleep):
    """Wait until each rail has a post-tN midpoint, then caller stops logger."""
    start = clock_ns()
    result = {"passed": False, "start_ns": start, "target_tN_ns": end_ns,
              "timeout_s": timeout_s, "poll_s": poll_s, "polls": 0}
    while True:
        result["polls"] += 1
        try:
            status = logger.status()
            result["last_status"] = status
            result["passed"] = all(
                status.get("rails", {}).get(rail, {}).get("valid") is True and
                type(status["rails"][rail].get("t_ns")) is int and
                status["rails"][rail]["t_ns"] >= end_ns for rail in RAILS)
        except Exception as exc:
            result.update(reason="power_status_error", error=f"{type(exc).__name__}: {exc}")
            break
        if result["passed"]:
            break
        elapsed = (clock_ns() - start) / 1e9
        if elapsed >= timeout_s:
            result["reason"] = "post_tN_power_coverage_timeout"
            break
        sleep(min(poll_s, timeout_s - elapsed))
    result["end_ns"] = clock_ns()
    result["wait_s"] = (result["end_ns"] - start) / 1e9
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class GenerationError(RuntimeError):
    def __init__(self, message, partial_result):
        super().__init__(message)
        self.partial_result = partial_result


def _verify_layers(rows, condition):
    if len(rows) != 32 or [row.get("layer") for row in rows] != list(range(32)):
        raise RuntimeError("Expected readback for all 32 attention layers")
    for row in rows:
        for field in ("k_bits", "v_bits", "group", "residual_length"):
            if row.get(field) != condition[field]:
                raise RuntimeError(f"layer {row.get('layer')}: {field} readback mismatch")


def generate_request(handle, item, condition, *, max_new_tokens=128,
                     verify_device=None, apply_condition=None, warmup=False):
    """Fresh-cache greedy generation; per-request GPU preparation is after t0.

    Model/JIT preparation belongs to the caller. The injected backend follows
    scripts.precision.generation.TorchGenerationBackend. Synchronization before
    every host monotonic token timestamp makes t1/tN GPU completion boundaries.
    Warmup uses the same natural EOS policy and cap as a formal request.
    """
    if handle.active_request or handle.released:
        raise RuntimeError("Model handle busy or released")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("Pilot output cap must be in [1,128], including first token")
    if condition not in CONDITIONS:
        raise ValueError("Only FP16 and fixed K2V2 G8/16/32/64/128 R128 are supported")
    ids = item.get("input_ids")
    if (not isinstance(ids, list) or not ids or len(ids) > 8192
            or any(type(x) is not int or not 0 <= x < 128256 for x in ids)
            or item.get("length") != len(ids)):
        raise ValueError("Prepared input IDs/actual length are invalid")
    if apply_condition is None:
        from scripts.precision.model import apply_precision_and_verify
        apply_condition = apply_precision_and_verify
    backend = handle.backend
    row = {"item_id": item["id"], "condition": dict(condition), "input_length": len(ids),
           "input_hash": digest(ids), "output_ids": [], "token_timestamps_ns": [],
           "t0_ns": None, "t1_ns": None, "tN_ns": None, "n": 0,
           "decode_forward_count": 0, "generation_valid": False,
           "termination_reason": None, "fresh_cache": True,
           "backend": getattr(backend, "name", "injected"),
           "do_sample": False, "num_beams": 1, "max_new_tokens": max_new_tokens,
           "eos_token_ids": list(EOS_IDS), "eos_suppressed": False,
           "purpose": "warmup" if warmup else "pilot_request",
           "quantization_path": "KIVI initial cache quantization/packing inside prefill forward before t1"
               if condition["method"] == "kivi" else "native HF FP16 cache",
           "quantization_separate_timing": False,
           "quantization_timing_note": "included once in TTFT; recurring packing included in decode; no separate additive estimate",
           "boundary_clock": "host time.monotonic_ns, explicit CUDA synchronize after each selected token",
           "finite_check_scope": "unmasked last-position logits each forward"}
    cache = token = output = state = None
    monitor_factory = getattr(backend, "compile_monitor", None)
    monitor = monitor_factory() if monitor_factory else None
    try:
        with backend.inference_mode(), monitor if monitor else nullcontext():
            backend.synchronize()
            row["t0_ns"] = time.monotonic_ns()
            start = time.monotonic_ns()
            row["layer_settings"] = apply_condition(handle, condition)
            _verify_layers(row["layer_settings"], condition)
            row["precision_apply_ns"] = time.monotonic_ns() - start
            handle.active_request = True
            if verify_device:
                row["device_verification"] = verify_device()
                if row["device_verification"].get("passed") is not True:
                    raise RuntimeError(f"Device controls do not match pilot: {row['device_verification']}")
            token = backend.prepare_input(ids)
            state = backend.begin_finite_check()
            for index in range(max_new_tokens):
                output = backend.forward(handle.model, token, cache)
                cache = output.past_key_values
                token, token_id, state = backend.select(output, state, ())
                backend.synchronize()
                stamp = time.monotonic_ns()
                output = None
                if type(token_id) is not int or not 0 <= token_id < 128256:
                    raise RuntimeError("Invalid output token/EOS policy violation")
                row["output_ids"].append(token_id)
                row["token_timestamps_ns"].append(stamp)
                row["n"], row["tN_ns"] = index + 1, stamp
                if index == 0:
                    row["t1_ns"] = stamp
                else:
                    row["decode_forward_count"] += 1
                if token_id in EOS_IDS:
                    row["termination_reason"] = "eos"
                    break
            if row["termination_reason"] is None:
                row["termination_reason"] = "max_new_tokens"
            # Finite/cache diagnostics and prediction conversion are outside tN.
            row["finite"] = backend.finish_finite_check(state)
            row["final_cache_length"] = backend.cache_length(cache, handle.method)
            row["expected_cache_length"] = len(ids) + row["n"] - 1
            row["generation_valid"] = bool(row["finite"] and
                row["final_cache_length"] == row["expected_cache_length"] and
                row["decode_forward_count"] == row["n"] - 1)
        row["jit"] = monitor.report() if monitor else {"observation_supported": False,
                                                       "reason": "backend_has_no_triton_monitor"}
        row["warm_timing_valid"] = bool(row["jit"].get("observation_supported") and
                                         not row["jit"].get("cache_misses"))
        return row
    except Exception as exc:
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if monitor:
            row["jit"] = monitor.report()
        raise GenerationError(str(exc), row) from exc
    finally:
        cache = token = output = state = None
        handle.active_request = False


def latencies(row):
    t0, t1, tn, n = (row.get(k) for k in ("t0_ns", "t1_ns", "tN_ns", "n"))
    valid = (type(n) is int and n >= 1 and all(type(v) is int for v in (t0, t1, tn))
             and t0 < t1 <= tn and ((n == 1 and t1 == tn) or (n > 1 and t1 < tn)))
    return {"status": "valid" if valid else "invalid",
            "ttft_s": (t1 - t0) / 1e9 if valid else None,
            "request_s": (tn - t0) / 1e9 if valid else None,
            "decode_s": (tn - t1) / 1e9 if valid and n > 1 else None,
            "tpot_s": (tn - t1) / (n - 1) / 1e9 if valid and n > 1 else None}


def measure_energy(samples, row, *, integrator=None, series_builder=None):
    """Preserve the existing LUT's independent three-rail sum and each rail.

    Do not add VDD_IN, collector total_w or any overlapping aggregate. This is
    module-rail sensor energy, not wall power or a claim of whole-board energy.
    """
    if integrator is None or series_builder is None:
        from scripts.p2.measurement import integrate_rails, rail_series
        integrator, series_builder = integrator or integrate_rails, series_builder or rail_series
    boundaries = {key: row.get(key) for key in ("t0_ns", "t1_ns", "tN_ns")}
    try:
        result = integrator({rail: series_builder(samples, rail) for rail in RAILS}, boundaries,
                            {"max_gap_ms": 100})
    except Exception as exc:
        result = {"status": "invalid", "reasons": [{"type": type(exc).__name__, "message": str(exc)}],
                  "rails": {}, "request_j": None, "prefill_j": None, "decode_j": None}
    result.update(scope=ENERGY_SCOPE, primary_rails=list(RAILS), rail_sum_used=True,
                  overlapping_aggregate_included=False, sensor_resolution_validated=False)
    n = row.get("n", 0)
    result["decode_j_per_token"] = (result["decode_j"] / (n - 1)
        if n > 1 and result.get("status") == "valid" and result.get("decode_j") is not None else None)
    if n <= 1:
        result["decode_j"] = None
        for rail in result.get("rails", {}).values():
            rail["decode_j"] = None
    coverage = {}
    for rail in RAILS:
        try:
            series = series_builder(samples, rail)
        except Exception as exc:
            result["status"] = "invalid"
            result.setdefault("reasons", []).append({"rail": rail, "type": type(exc).__name__, "message": str(exc)})
            coverage[rail] = {"brackets_request": False, "error": str(exc)}
            for metric in ("request_j", "prefill_j", "decode_j", "decode_j_per_token"):
                result[metric] = None
            continue
        points = series.get("points", [])
        stamps = [p["t_ns"] for p in points if type(p.get("t_ns")) is int]
        t0, tn = row.get("t0_ns"), row.get("tN_ns")
        coverage[rail] = {"samples": len(points), "first_midpoint_ns": min(stamps) if stamps else None,
                          "last_midpoint_ns": max(stamps) if stamps else None,
                          "brackets_request": bool(stamps and t0 is not None and tn is not None
                                                    and min(stamps) <= t0 and max(stamps) >= tn),
                          "read_errors": series.get("errors", [])}
    result["coverage"] = coverage
    return result


class AttemptJournal:
    """Fsync append-only starts/results. Interrupted attempts remain visible."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if self.path.exists():
            for number, line in enumerate(self.path.read_text().splitlines(), 1):
                try:
                    self.rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt journal line {number}; preserve and repair explicitly: {self.path}") from exc

    def append(self, row):
        with self.path.open("a") as stream:
            stream.write(canonical(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)

    def completed(self, key):
        return any(r.get("type") == "attempt_result" and r.get("request_key") == key
                   and r.get("technical_valid") is True for r in self.rows)

    def start(self, key, metadata):
        attempt = 1 + max((r.get("attempt", 0) for r in self.rows if r.get("request_key") == key), default=0)
        self.append({**metadata, "type": "attempt_start", "request_key": key,
                     "attempt": attempt, "start_ns": time.monotonic_ns()})
        return attempt

    def finish(self, row):
        self.append({**row, "type": "attempt_result"})


def configure_existing_sources(root):
    """Extend the scripts package search path without modifying AGX2 sources."""
    root = Path(root).resolve()
    required = ("scripts/precision/model.py", "scripts/p2/measurement.py",
                "third_party/KIVI/models/llama_kivi.py")
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    import scripts
    location = str(root / "scripts")
    if location not in scripts.__path__:
        scripts.__path__.append(location)
    # Spawned power workers inherit sys.path, not mutations to scripts.__path__.
    # The existing scripts tree is a namespace package on the audited device.
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return {relative: file_hash(root / relative) for relative in required}


def validate_inputs(document, *, formal):
    if document.get("schema") != "kvolt.eval_pilot.inputs.v1":
        raise ValueError("Unsupported inputs schema")
    items = document.get("items", [])
    if not items or len({item["id"] for item in items}) != len(items):
        raise ValueError("Inputs require unique nonempty item IDs")
    for item in items:
        if item["length"] != len(item["input_ids"]) or not item.get("answers"):
            raise ValueError("Actual token length/answers missing")
    if formal:
        if len(items) != 30 or sorted(set(i["length"] for i in items)) != [1024, 4096, 8192]:
            raise ValueError("Formal pilot requires 30 items across L1024/4096/8192")
        for length in (1024, 4096, 8192):
            for task in ("niah_multiquery", "niah_multivalue"):
                selected = [i for i in items if i["length"] == length and i["task"] == task]
                if sorted(i["difficulty"] for i in selected) != ["basic", "basic", "basic", "hard", "hard"]:
                    raise ValueError("Formal pilot requires each length/task basic3 + hard2")
    return items


def _memory(backend):
    cuda = backend.torch.cuda
    return {"allocated_bytes": cuda.memory_allocated(), "reserved_bytes": cuda.memory_reserved(),
            "max_allocated_bytes": cuda.max_memory_allocated(),
            "max_reserved_bytes": cuda.max_memory_reserved()}


def _warmup(handle, item, condition, output, verify_device):
    """Natural-EOS preparation; repeat once only when compilation needs it."""
    passes = []
    for index in range(2):
        start = time.monotonic_ns()
        try:
            row = generate_request(handle, item, condition, verify_device=verify_device, warmup=True)
        except GenerationError as exc:
            write_json(output / f"pass-{index + 1}.json", {**exc.partial_result,
                       "formal": False, "wall_start_ns": start, "wall_end_ns": time.monotonic_ns(),
                       "warmup_pass": index + 1})
            raise
        row.update(wall_start_ns=start, wall_end_ns=time.monotonic_ns(), formal=False,
                   warmup_pass=index + 1)
        passes.append(row)
        write_json(output / f"pass-{index + 1}.json", row)
        if not row["generation_valid"]:
            raise RuntimeError("Warmup failed finite/cache validation")
        if not row["jit"].get("observation_supported"):
            raise RuntimeError("Warmup cannot verify JIT observation; no repeat authorized without a cache miss")
        if row["warm_timing_valid"]:
            break
    if not passes[-1]["warm_timing_valid"]:
        raise RuntimeError("Warmup has JIT cache misses or unsupported JIT observation")
    return {"passed": True, "paths": [str(output / f"pass-{i + 1}.json") for i in range(len(passes))],
            "natural_eos": True, "actual_output_tokens": [row["n"] for row in passes],
            "wall_s": (passes[-1]["wall_end_ns"] - passes[0]["wall_start_ns"]) / 1e9}


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    document = json.loads(Path(args.inputs).read_text())
    items = validate_inputs(document, formal=not args.smoke)
    sources = configure_existing_sources(args.model_root)
    from scripts.precision.model import load_method, release_model
    from scripts.p2.measurement import RequestPowerLogger
    from .device import verify_environment, _last_complete_sample
    for module in ("scripts.precision.model", "scripts.p2.model", "scripts.p2.measurement"):
        if not Path(sys.modules[module].__file__).resolve().is_relative_to(Path(args.model_root).resolve()):
            raise RuntimeError(f"Existing backend imported from a different source root: {module}")
    evidence_path = Path(args.device_evidence).resolve()
    evidence = json.loads(evidence_path.read_text())
    verify_device = lambda: verify_environment(evidence)
    read_temperature = lambda: _last_complete_sample(evidence["monitor_path"])
    if verify_device().get("passed") is not True:
        raise RuntimeError("Device verification failed before model loading")
    planned = paired_schedule(items if not args.smoke else items[:1])
    if args.smoke and args.limit is not None:
        planned = planned[:args.limit]
    schedule = [{"position": position, "item_id": item["id"], "condition": condition}
                for position, (item, condition) in enumerate(planned)]
    manifest = {"schema": "kvolt.eval_pilot.run.v1", "formal": not args.smoke,
                "inputs_sha256": file_hash(args.inputs), "snapshot": str(Path(args.snapshot).resolve()),
                "model_root": str(Path(args.model_root).resolve()), "conditions": list(CONDITIONS),
                "source_sha256": sources, "runner_sha256": file_hash(__file__),
                "max_new_tokens": 128, "energy_scope": ENERGY_SCOPE,
                "device_evidence": str(evidence_path), "device_session": evidence,
                "input_metadata": document.get("metadata", {}), "repeat": 1,
                "schedule_seed": SCHEDULE_SEED, "schedule": schedule, "schedule_sha256": digest(schedule),
                "schedule_policy": "random item blocks, independently randomized six settings within each item",
                "temperature_admission": {"formal_minimum_mc": 35000, "maximum_mc": 55000,
                    "prewarmup_minimum_mc": None, "timeout_s": 900, "poll_s": 1,
                    "maximum_monitor_age_s": 2, "forced_heating": False},
                "post_tN_power_guard": {"timeout_s": 2, "poll_s": 0.02,
                    "criterion": "each required rail valid midpoint >= tN", "inside_request_timing": False}}
    # Session evidence changes across a legitimate resume; protocol/source/input
    # identity may not change. Each invocation's evidence remains separately saved.
    protocol = {k: v for k, v in manifest.items() if k not in ("device_evidence", "device_session")}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["protocol_hash"] != digest(protocol):
            raise RuntimeError("Resume protocol/source/input mismatch; use a different output directory")
    else:
        write_json(manifest_path, {**manifest, "protocol_hash": digest(protocol)})
    invocation = str(time.time_ns())
    write_json(output / "sessions" / f"{invocation}.json", manifest)
    journal = AttemptJournal(output / "requests.jsonl")
    handle = None
    lifecycle = {"invocation": invocation, "start_ns": time.monotonic_ns(), "events": []}
    warmed = {}
    try:
        started = time.monotonic_ns()
        lifecycle["events"].append({"kind": "model_load_start", "start_ns": started})
        handle = load_method(args.snapshot, "fp16")
        lifecycle["events"].append({"kind": "model_load", "start_ns": started,
                                    "end_ns": time.monotonic_ns(), "provenance": handle.provenance})
        for position, (item, condition) in enumerate(planned):
            request_key = f"{item['id']}:{condition['name']}:g{condition['group']}"
            if journal.completed(request_key):
                continue
            shape = (item["id"], condition["name"], condition["group"])
            if shape not in warmed:
                folder = output / "warmup" / invocation / digest(request_key)[:16]
                lifecycle["events"].append({"kind": "warmup_start", "item_id": item["id"],
                                            "condition": condition, "start_ns": time.monotonic_ns()})
                admission = wait_temperature(read_temperature, minimum_mc=None)
                write_json(folder / "temperature-admission.json", admission)
                lifecycle["events"].append({"kind": "prewarmup_temperature_admission",
                                            "item_id": item["id"], "condition": condition, **admission})
                if not admission["passed"]:
                    raise RuntimeError("Pre-warmup temperature admission failed; no extra heating performed")
                proof = _warmup(handle, item, condition, folder, verify_device)
                proof["temperature_admission_path"] = str(folder / "temperature-admission.json")
                warmed[shape] = proof
                lifecycle["events"].append({"kind": "warmup", "shape": list(shape), **proof})
                write_json(output / "lifecycle" / f"{invocation}.json", lifecycle)
            attempt = journal.start(request_key, {"formal": not args.smoke, "invocation": invocation,
                                                  "item_id": item["id"], "condition": condition})
            folder = output / "requests" / digest(request_key)[:16] / f"attempt-{attempt}"
            folder.mkdir(parents=True, exist_ok=False)
            row = {"request_key": request_key, "attempt": attempt, "formal": not args.smoke,
                   "invocation": invocation, "item_id": item["id"], "condition": condition,
                   "input_length": item["length"], "input_hash": digest(item["input_ids"]),
                   "answers": item["answers"], "task": item.get("task"),
                   "difficulty": item.get("difficulty"), "family_id": item.get("family_id"),
                   "cohort": item.get("cohort"), "warmup": warmed[shape], "schedule_position": position,
                   "wall_start_ns": time.monotonic_ns(), "errors": []}
            power = None
            try:
                handle.backend.torch.cuda.reset_peak_memory_stats()
                row["memory_before"] = _memory(handle.backend)
                power = RequestPowerLogger(folder / "power.jsonl", {"interval_ms": 20,
                    "max_gap_ms": 100, "ready_max_age_ms": 100, "telemetry": True,
                    "status_timeout_s": 2})
                power.start()
                row["power_ready"] = power.status(require_valid=True)
                row["temperature_admission"] = wait_temperature(read_temperature)
                if not row["temperature_admission"]["passed"]:
                    raise RuntimeError("Formal temperature admission failed; no extra heating performed")
                generated = generate_request(handle, item, condition, verify_device=verify_device)
                row.update(generated)
            except Exception as exc:
                if isinstance(exc, GenerationError):
                    row.update(exc.partial_result)
                row["errors"].append({"phase": "request", "type": type(exc).__name__, "message": str(exc),
                                      "traceback": traceback.format_exc()})
            finally:
                if power is not None:
                    if row.get("tN_ns") is not None:
                        row["post_tN_power_guard"] = wait_power_coverage(power, row["tN_ns"])
                        if not row["post_tN_power_guard"]["passed"]:
                            row["errors"].append({"phase": "power_coverage", **row["post_tN_power_guard"]})
                    try:
                        row["power_summary"] = power.stop()
                    except Exception as exc:
                        row["errors"].append({"phase": "power_stop", "type": type(exc).__name__, "message": str(exc)})
                row["power_trace"] = str(folder / "power.jsonl")
                try:
                    row["power_trace_sha256"] = file_hash(row["power_trace"]) if Path(row["power_trace"]).exists() else None
                except OSError as exc:
                    row["power_trace_sha256"] = None
                    row["errors"].append({"phase": "power_hash", "type": type(exc).__name__, "message": str(exc)})
                row["latencies"] = latencies(row)
                row["energy"] = measure_energy(power.samples if power is not None else [], row)
                try:
                    row["memory_after"] = _memory(handle.backend)
                except Exception as exc:
                    row["errors"].append({"phase": "memory_after", "type": type(exc).__name__, "message": str(exc)})
                row["technical_valid"] = bool(row.get("generation_valid") and row.get("warm_timing_valid")
                    and row["latencies"]["status"] == "valid" and row["energy"]["status"] == "valid"
                    and not row["errors"])
                try:
                    postprocess_start = time.monotonic_ns()
                    row["prediction"] = handle.tokenizer.decode(row.get("output_ids", []),
                        skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    from .scoring import score_prediction
                    row["score"] = score_prediction(row["prediction"], item["answers"])
                except Exception as exc:
                    row["postprocess_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    row["technical_valid"] = False
                row["postprocess_wall_s"] = (time.monotonic_ns() - postprocess_start) / 1e9
                row["wall_end_ns"] = time.monotonic_ns()
                row["wall_s"] = (row["wall_end_ns"] - row["wall_start_ns"]) / 1e9
                write_json(folder / "result.json", row)
                journal.finish(row)
            print(canonical({"request_key": request_key, "attempt": attempt,
                             "technical_valid": row["technical_valid"], "n": row.get("n"),
                             "formal": not args.smoke}), flush=True)
            if not row["technical_valid"]:
                # Stop on technical failure; resume adds a new attempt and never
                # discards this failure or repeats already successful requests.
                raise RuntimeError(f"Technical failure retained for {request_key}; inspect result before resume")
        return 0
    finally:
        if sys.exc_info()[1] is not None:
            exc = sys.exc_info()[1]
            lifecycle["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if handle is not None:
            try:
                lifecycle["events"].append({"kind": "model_release", **release_model(handle)})
            except Exception as exc:
                lifecycle["release_error"] = {"type": type(exc).__name__, "message": str(exc)}
        lifecycle["end_ns"] = time.monotonic_ns()
        write_json(output / "lifecycle" / f"{invocation}.json", lifecycle)
        if "release_error" in lifecycle and "error" not in lifecycle:
            raise RuntimeError(f"Model release failed: {lifecycle['release_error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True, help="Output directory; append-only checkpoint journal")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model-root", default="/mnt/kvolt-data/kvolt")
    parser.add_argument("--device-evidence", required=True)
    parser.add_argument("--smoke", action="store_true", help="Separate functional smoke; never formal")
    parser.add_argument("--limit", type=int, help="Smoke request limit only; formal always schedules 180")
    args = parser.parse_args(argv)
    if args.limit is not None and (not args.smoke or args.limit < 1):
        parser.error("--limit requires --smoke and a positive integer")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
