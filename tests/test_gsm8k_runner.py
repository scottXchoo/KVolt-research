"""CPU fake-backend checks for the GSM8K generation and scoring skeleton."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.quality.GSM8K import KIVI_Llama as runner


@pytest.fixture
def fake_backend(monkeypatch):
    calls = []
    model = SimpleNamespace(generation_config=SimpleNamespace(
        num_return_sequences=3, return_dict_in_generate=True,
    ))
    tokenizer = object()

    def load(model_name, **settings):
        calls.append(("load", model_name, settings))
        return model, tokenizer

    predictions = iter(["Reasoning.\n#### 18\n", "#### 18.0", "No marker"])

    def generate(received_model, received_tokenizer, prompt, **settings):
        assert received_model is model
        assert received_tokenizer is tokenizer
        assert model.generation_config.num_return_sequences == 1
        assert model.generation_config.return_dict_in_generate is False
        calls.append(("generate", prompt, settings))
        return next(predictions)

    monkeypatch.setattr(runner.kivi_llama, "load_model", load)
    monkeypatch.setattr(runner.kivi_llama, "generate", generate)
    return calls


def test_run_preserves_continuation_scores_and_fixed_settings(tmp_path, fake_backend):
    output = tmp_path / "run"
    summary = runner.run(output, limit=3)
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    records = runner.load_dataset()
    assert len(rows) == 3
    assert rows[0] == {
        "id": records[0]["id"], "question": records[0]["question"],
        "prediction": "Reasoning.\n#### 18\n", "predicted_answer": "18",
        "answer": "18", "score": 1.0,
    }
    assert rows[1]["score"] == rows[2]["score"] == 0.0
    assert rows[2]["predicted_answer"] is None
    assert fake_backend[0] == ("load", "meta-llama/Llama-3.1-8B", {
        "k_bits": 2, "v_bits": 2, "group_size": 32, "residual_length": 128,
    })
    for call, record in zip(fake_backend[1:], records):
        assert call == ("generate", record["prompt"], {"max_new_tokens": 512})
        assert record["reference_solution"] not in call[1]
    assert summary["evaluated"] == 3
    assert summary["correct"] == 1
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["subset"] is True
    assert summary["total_test_rows"] == 1319
    assert summary["first_id"] == records[0]["id"]
    assert summary["last_id"] == records[2]["id"]
    assert summary["dataset"]["revision"] == runner.PINNED_REVISION
    assert summary["dataset"]["source_sha256"] == runner.PINNED_SOURCE_SHA256
    assert summary["dataset"]["protocol_sha256"] == runner.PINNED_PROTOCOL_SHA256
    assert summary["settings"]["max_new_tokens"] == 512
    assert summary == json.loads((output / "summary.json").read_text())


def test_default_evaluates_all_records(tmp_path, monkeypatch, fake_backend):
    monkeypatch.setattr(runner.kivi_llama, "generate", lambda *args, **kwargs: "#### 0")
    summary = runner.run(tmp_path / "full", model_name="/local/base-llama")
    assert summary["evaluated"] == 1319
    assert summary["subset"] is False
    assert fake_backend[0][1] == "/local/base-llama"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2", 1320])
def test_invalid_limit_rejected_before_gpu_or_output(tmp_path, fake_backend, limit):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="limit"):
        runner.run(output, limit=limit)
    assert fake_backend == []
    assert not output.exists()


def test_data_failure_precedes_gpu_and_output(tmp_path, monkeypatch, fake_backend):
    def invalid_data():
        raise ValueError("data pin mismatch")
    monkeypatch.setattr(runner, "load_dataset", invalid_data)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="pin"):
        runner.run(output, limit=1)
    assert fake_backend == []
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path, fake_backend):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "predictions.jsonl"
    sentinel.write_text("keep me")
    with pytest.raises(FileExistsError):
        runner.run(output, limit=1)
    assert sentinel.read_text() == "keep me"
    assert fake_backend == []


def test_generation_failure_preserves_flushed_partial_without_summary(tmp_path, monkeypatch, fake_backend):
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("GPU failure")
        return "#### 18"

    monkeypatch.setattr(runner.kivi_llama, "generate", generate)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="GPU failure"):
        runner.run(output, limit=3)
    rows = (output / "predictions.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["score"] == 1.0
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("arguments", [
    [str(Path(__file__).resolve().parents[1] / "src/experiments/quality/GSM8K/KIVI_Llama.py")],
    ["-m", "experiments.quality.GSM8K.KIVI_Llama"],
])
def test_help_and_bad_limit_work_without_gpu_from_other_cwd(tmp_path, arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    help_result = subprocess.run(
        [sys.executable, *arguments, "--help"], cwd=tmp_path,
        env=environment, capture_output=True, text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--limit" in help_result.stdout
    bad_limit = subprocess.run(
        [sys.executable, *arguments, "--limit", "0"], cwd=tmp_path,
        env=environment, capture_output=True, text=True,
    )
    assert bad_limit.returncode == 2
    assert "positive integer" in bad_limit.stderr


def test_import_keeps_gpu_dependencies_unloaded():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from experiments.quality.GSM8K import KIVI_Llama; "
        "assert not ({'torch', 'transformers', 'triton', 'flash_attn'} & set(sys.modules))",
    ], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
