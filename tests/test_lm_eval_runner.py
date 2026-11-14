"""Installed lm-eval integration using only local task data and tiny CPU weights."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("lm_eval.api.task")

from datasets import Dataset, DatasetDict
from lm_eval import simple_evaluate
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

from experiments.lm_eval.backend import KIVILM

# Reuse the shared helper's real HF generation-loop fixture, not a fake generate.
from test_kivi_stopped_generation import toy


@pytest.fixture
def local_task():
    class LocalTask(ConfigurableTask):
        def download(self, dataset_kwargs=None, **kwargs):
            self.dataset = DatasetDict(test=Dataset.from_list([
                {"question": "<END>Q0", "answer": "answer"},
                {"question": "Q1", "answer": "answer"},
            ]))

    return LocalTask(config={
        "task": "kvolt_local_stop", "dataset_path": "", "output_type": "generate_until",
        "test_split": "test", "doc_to_text": "question", "doc_to_target": "answer",
        "num_fewshot": 0, "generation_kwargs": {
            "until": ["<END>"], "max_gen_toks": 30, "do_sample": False,
            "temperature": 0.0,
        },
        "metric_list": [{"metric": "exact_match", "aggregation": "mean",
                         "higher_is_better": True}],
    })


def test_actual_simple_evaluate_stops_real_generation_and_scores(toy, local_task, tmp_path):
    lm = KIVILM(model=toy.model, tokenizer=toy.tokenizer, output_dir=tmp_path,
                max_new_tokens=40, max_input_tokens=32)
    result = simple_evaluate(
        model=lm, tasks=[local_task], batch_size=1, use_cache=None, cache_requests=False,
        log_samples=True, apply_chat_template=False, bootstrap_iters=0,
        random_seed=1234, numpy_random_seed=1234, torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    assert result["results"]["kvolt_local_stop"]["exact_match,none"] == 1.0
    assert [record["doc_id"] for record in lm.records] == [0, 1]
    assert [record["task_name"] for record in lm.records] == ["kvolt_local_stop"] * 2
    for record in lm.records:
        assert record["text"] == "answer" and record["raw_text"] == "answer<END>"
        assert record["output_token_ids"] == [ord(c) + 3 for c in "answer<END>"]
        assert record["output_tokens"] == len("answer<END>") < 30
        assert record["stop_reason"] == "stop_sequence"
        assert record["energy"] is None and record["energy_artifacts_dir"] is None
    assert toy.generated_steps == 2 * len("answer<END>")
    assert "lm_eval.models.huggingface" not in sys.modules


def test_official_gsm8k_ruler_task_discovery_without_download():
    manager = TaskManager()
    assert "gsm8k" in manager.all_tasks
    assert "ruler" in manager.all_tasks
    for name in ("niah_single_1", "niah_multikey_2", "ruler_vt", "ruler_fwe"):
        assert name in manager.all_tasks
    gsm_config = load_yaml(manager.task_index["gsm8k"].yaml_path, resolve_func=False)
    ruler_config = load_yaml(manager.task_index["niah_single_1"].yaml_path, resolve_func=False)
    assert gsm_config["output_type"] == ruler_config["output_type"] == "generate_until"
    assert ruler_config["generation_kwargs"]["max_gen_toks"] == 128
    vt_config = load_yaml(manager.task_index["ruler_vt"].yaml_path, resolve_func=False)
    assert vt_config["generation_kwargs"]["max_gen_toks"] == 30
    group_config = load_yaml(manager.task_index["ruler"].yaml_path, resolve_func=False)
    assert [metric["metric"] for metric in group_config["aggregate_metric_list"]] == ["4096"]


def test_runner_writes_complete_results_samples_and_manifest(toy, local_task, tmp_path):
    from experiments.lm_eval.runner import run_evaluation

    output = tmp_path / "complete"
    result = run_evaluation(output, tasks=[local_task], model=toy.model,
                            tokenizer=toy.tokenizer, max_new_tokens=30, max_input_tokens=32)
    assert result["results"]["kvolt_local_stop"]["exact_match,none"] == 1.0
    saved = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "run.json").read_text())
    assert saved["results"]["kvolt_local_stop"]["exact_match,none"] == 1.0
    assert "samples" not in saved
    assert len((output / "samples.jsonl").read_text().splitlines()) == 2
    assert len((output / "records.jsonl").read_text().splitlines()) == 2
    assert manifest["schema"].startswith("kvolt.lm-eval.")
    assert "0.4.12" in json.dumps(manifest)


def test_runner_failure_does_not_write_complete_results(toy, local_task, tmp_path):
    from experiments.lm_eval.runner import run_evaluation

    output = tmp_path / "failure"
    error = RuntimeError("tiny model failure")
    toy.fail = error
    with pytest.raises(RuntimeError) as caught:
        run_evaluation(output, tasks=[local_task], model=toy.model,
                       tokenizer=toy.tokenizer, max_new_tokens=30, max_input_tokens=32)
    assert caught.value is error
    assert not (output / "results.json").exists()
    assert json.loads((output / "records.jsonl").read_text().splitlines()[0])["status"] == "failed"


@pytest.mark.parametrize("kwargs", [
    {"tasks": []}, {"tasks": ["gsm8k"], "limit": True},
    {"tasks": ["gsm8k"], "max_new_tokens": 0},
    {"tasks": ["gsm8k"], "seed": True},
    {"tasks": ["gsm8k"], "power_profile": "power.json"},
    {"tasks": ["niah_single_1"], "ruler_lengths": (1024,)},
    {"tasks": ["ruler"], "ruler_lengths": (8192,)},
])
def test_runner_rejects_invalid_configuration_without_model(tmp_path, kwargs):
    from experiments.lm_eval.runner import run_evaluation

    with pytest.raises((TypeError, ValueError)):
        run_evaluation(tmp_path / "invalid", **kwargs)


def test_cli_help_and_import_remain_cpu_safe():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    code = """
import sys
from experiments.lm_eval import runner
from experiments.quality.GSM8K import KIVI_Llama
from experiments.quality.RULER import KIVI_Llama
assert 'torch' not in sys.modules
assert 'lm_eval' not in sys.modules
assert 'transformers' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], env=env,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    completed = subprocess.run([sys.executable, "-m", "experiments.lm_eval", "--help"],
                               env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "--tasks" in completed.stdout


def test_lm_eval_version_mismatch_fails_before_output_or_hardware(tmp_path, monkeypatch):
    from experiments.lm_eval import runner

    monkeypatch.setattr(runner, "_package_versions", lambda: {"lm_eval": "0.4.11"})
    output = tmp_path / "wrong-version"
    with pytest.raises(RuntimeError, match="lm_eval==0.4.12"):
        runner.run_evaluation(output, tasks=["gsm8k"])
    assert not output.exists()


@pytest.mark.parametrize("module_name,expected_tasks", [
    ("experiments.quality.GSM8K.KIVI_Llama", ("gsm8k",)),
    ("experiments.quality.RULER.KIVI_Llama", ("ruler",)),
])
def test_legacy_cli_routes_only_explicit_lm_eval_flag(monkeypatch, module_name, expected_tasks):
    import importlib
    from experiments.lm_eval import runner

    calls = []

    def main(argv, *, default_tasks):
        calls.append((argv, default_tasks))
        return 7

    monkeypatch.setattr(runner, "main", main)
    legacy = importlib.import_module(module_name)
    assert legacy.main(["--lm-eval", "--limit", "1"]) == 7
    assert calls == [(["--limit", "1"], expected_tasks)]


def test_input_overflow_is_recorded_without_truncation(toy, tmp_path):
    from lm_eval.api.instance import Instance

    lm = KIVILM(model=toy.model, tokenizer=toy.tokenizer,
                output_dir=tmp_path, max_new_tokens=10, max_input_tokens=2)
    instance = Instance("generate_until", {}, ("three", {}), 0,
                        metadata=("local", 0, 1))
    with pytest.raises(ValueError, match="truncation"):
        lm.generate_until([instance])
    assert toy.calls == []
    assert lm.records[0]["status"] == "failed"


def test_invalid_energy_keeps_quality_score_and_null_energy(toy, local_task, tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from experiments.lm_eval.runner import run_evaluation
    from kvolt_runtime import telemetry
    from model import kivi_llama

    profile = SimpleNamespace(to_dict=lambda: {"fake_test_profile": True})
    monkeypatch.setattr(telemetry, "discover_power_profile", lambda: profile)
    raw = '{"fake":"invalid power trace"}\n'
    sha = hashlib.sha256(raw.encode()).hexdigest()
    original = kivi_llama.generate_with_details

    def measured(model, tokenizer, prompt, **kwargs):
        assert kwargs["measure_energy"] is True and kwargs["power_profile"] is profile
        result = original(model, tokenizer, prompt, **{
            **kwargs, "measure_energy": False, "power_profile": None,
        })
        result["energy"] = {"valid": False, "energy_j": None, "trace_sha256": sha}
        result["energy_receipt"] = {"raw_trace_jsonl": raw, "samples": [], "trace_sha256": sha}
        return result

    monkeypatch.setattr(kivi_llama, "generate_with_details", measured)
    output = tmp_path / "invalid-energy"
    result = run_evaluation(output, tasks=[local_task], model=toy.model, tokenizer=toy.tokenizer,
                            max_new_tokens=30, max_input_tokens=32, measure_energy=True)
    assert result["results"]["kvolt_local_stop"]["exact_match,none"] == 1.0
    assert result["energy_invalid_count"] == 2
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    for ordinal, row in enumerate(rows):
        assert row["energy"]["energy_j"] is None and row["energy"]["valid"] is False
        assert (output / f"energy/{ordinal:06d}/power-trace.jsonl").read_bytes() == raw.encode()
        assert row["energy_receipt"]["trace_sha256"] == sha
