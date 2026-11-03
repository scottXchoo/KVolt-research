"""Summarize the fixed 180-request pilot without treating repeats as samples."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(rows):
    attempts = [r for r in rows if r.get("type") == "attempt_result" and r.get("formal") is True]
    valid = {}
    for row in attempts:
        if row.get("technical_valid"):
            if row["request_key"] in valid:
                raise ValueError("A successful formal request was measured more than once")
            valid[row["request_key"]] = row
    grouped = defaultdict(list)
    for row in valid.values():
        c = row["condition"]
        grouped[(row["input_length"], c["name"], c["group"])].append(row)
    cells = []
    for (length, method, group), records in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0)):
        cells.append({"length": length, "method": method, "group": group,
            "requests": len(records), "families": len({r["family_id"] for r in records}),
            "mean_score": _mean([r.get("score") for r in records]),
            "mean_request_j": _mean([r["energy"].get("request_j") for r in records]),
            "mean_decode_j_per_token": _mean([r["energy"].get("decode_j_per_token") for r in records]),
            "mean_ttft_s": _mean([r["latencies"].get("ttft_s") for r in records]),
            "mean_tpot_s": _mean([r["latencies"].get("tpot_s") for r in records]),
            "mean_request_s": _mean([r["latencies"].get("request_s") for r in records]),
            "min_request_s": min(r["latencies"]["request_s"] for r in records),
            "max_request_s": max(r["latencies"]["request_s"] for r in records),
            "mean_output_tokens": _mean([r["n"] for r in records]),
            "eos_requests": sum(r.get("termination_reason") == "eos" for r in records),
            "cap_requests": sum(r.get("termination_reason") == "max_new_tokens" for r in records),
            "n1_requests": sum(r["n"] == 1 for r in records),
            "wall_s": sum(r.get("wall_s", 0) for r in records)})
    expected = {(length, name, group) for length in (1024, 4096, 8192)
                for name, group in [("fp16", None), *(("k2v2", g) for g in (8, 16, 32, 64, 128))]}
    families = {r["family_id"] for r in valid.values()}
    complete = (set(grouped) == expected and len(valid) == 180 and len(families) == 10
                and all(len(records) == 10 and {r["family_id"] for r in records} == families
                        for records in grouped.values()))
    finished = {(r.get("request_key"), r.get("attempt")) for r in attempts}
    unfinished = sum((r.get("request_key"), r.get("attempt")) not in finished for r in rows
                     if r.get("type") == "attempt_start" and r.get("formal") is True)
    return {"schema": "kvolt.eval_pilot.summary.v1", "complete": complete,
            "expected_formal_requests": 180, "valid_formal_requests": len(valid),
            "technical_failed_attempts": sum(not r.get("technical_valid") for r in attempts),
            "unfinished_attempts": unfinished,
            "independent_semantic_families": len(families),
            "repeat": 1, "cells": cells,
            "limitations": ["Input families are paired across lengths and methods.",
                "Observed request ranges include input differences, not repeated-measurement uncertainty.",
                "No confidence interval, fixed-frequency or new KVolt Runtime claim is made."]}


def analyze(output):
    output = Path(output)
    path = output / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = summarize(rows)
    lifecycle = [json.loads(p.read_text()) for p in (output / "lifecycle").glob("*.json")]
    result["preparation"] = {"warmup_s": sum(e.get("wall_s", 0) for l in lifecycle for e in l.get("events", []) if e.get("kind") == "warmup"),
        "model_load_s": sum((e["end_ns"] - e["start_ns"]) / 1e9 for l in lifecycle for e in l.get("events", []) if e.get("kind") == "model_load"),
        "worker_lifecycle_s": sum((l["end_ns"] - l["start_ns"]) / 1e9 for l in lifecycle if "end_ns" in l)}
    result["device_restorations"] = [json.loads(p.read_text()) for p in output.glob("device-session-*/restoration*.json")]
    folder = output / "analysis"
    folder.mkdir(exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["cells"]:
        with (folder / "cells.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["cells"][0]))
            writer.writeheader()
            writer.writerows(result["cells"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    result = analyze(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ("complete", "valid_formal_requests", "technical_failed_attempts")}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
