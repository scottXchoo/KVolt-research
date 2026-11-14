"""CPU contract checks for the RULER KIVI runner; no GPU accuracy claims."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.quality.RULER import KIVI_Llama as runner


@pytest.fixture
def fake_backend(monkeypatch):
    calls = []
    token_calls = []
    model = SimpleNamespace(
        generation_config=SimpleNamespace(num_return_sequences=3, return_dict_in_generate=True),
        config=SimpleNamespace(max_position_embeddings=100_000),
    )

    def tokenizer(prompt, **kwargs):
        token_calls.append((prompt, kwargs))
        return {"input_ids": SimpleNamespace(tolist=lambda: [[128000, 17, 23]])}

    def load(model_name, **settings):
        calls.append(("load", model_name, settings))
        return model, tokenizer

    def generate(received_model, received_tokenizer, prompt, **settings):
        assert received_model is model
        assert received_tokenizer is tokenizer
        assert model.generation_config.num_return_sequences == 1
        assert model.generation_config.return_dict_in_generate is False
        calls.append(("generate", prompt, settings))
        return "new continuation only"

    monkeypatch.setattr(runner.kivi_llama, "load_model", load)
    monkeypatch.setattr(runner.kivi_llama, "generate", generate)
    return SimpleNamespace(calls=calls, token_calls=token_calls, model=model)


def read_rows(output):
    return [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]


def test_run_preserves_continuation_and_token_identity(tmp_path, fake_backend):
    output = tmp_path / "smoke"
    records = runner.load_dataset()
    summary = runner.run(output, limit=2)
    rows = read_rows(output)
    assert len(rows) == 2
    assert fake_backend.calls[0] == ("load", "meta-llama/Llama-3.1-8B-Instruct", {
        "k_bits": 2, "v_bits": 2, "group_size": 32, "residual_length": 128,
    })
    for row, record, call, token_call in zip(rows, records, fake_backend.calls[1:], fake_backend.token_calls):
        assert row["id"] == record["id"]
        assert row["task"] == record["task"]
        assert row["prediction"] == "new continuation only"
        assert row["answers"] == record["answers"]
        assert row["score"] == runner.score(row["prediction"], record["answers"])
        assert row["source_length"] == record["source_length"]
        assert row["source_length_tokenizer"] == "cl100k_base (input + reserved output)"
        assert row["input_tokenizer"] == runner.MODEL_NAME
        assert row["input_tokens"] == 3
        assert row["input_ids_sha256"] == hashlib.sha256(b"[128000,17,23]").hexdigest()
        assert row["max_new_tokens"] == record["max_new_tokens"]
        assert call == ("generate", record["prompt"], {"max_new_tokens": record["max_new_tokens"]})
        assert token_call == (record["prompt"], {"return_tensors": "pt"})
    assert summary["evaluated"] == 2
    assert summary["total_dataset_rows"] == summary["total_selected_task_rows"] == 40
    assert summary["subset"] is True
    assert summary["first_id"] == records[0]["id"]
    assert summary["last_id"] == records[1]["id"]
    assert summary["dataset"]["revision"] == runner.PINNED_REVISION
    assert summary["dataset"]["source_sha256"] == runner.PINNED_SOURCE_SHA256
    assert summary["dataset"]["protocol_sha256"] == runner.PINNED_PROTOCOL_SHA256
    assert summary["dataset"]["selected_prompt_sha256"] == hashlib.sha256(
        "".join(record["prompt"] + "\n" for record in records[:2]).encode()
    ).hexdigest()
    assert summary["settings"]["chat_template"] is None
    assert summary["input_scope"]["full_benchmark"] is False
    assert summary == json.loads((output / "summary.json").read_text())


def test_task_filter_and_limit_preserve_source_order(tmp_path, fake_backend):
    task = "vt"
    records = [record for record in runner.load_dataset() if record["task"] == task]
    summary = runner.run(tmp_path / "task", task=task, limit=3, model_name="/local/instruct")
    assert summary["evaluated"] == 3
    assert summary["total_selected_task_rows"] == 10
    assert summary["total_dataset_rows"] == 40
    assert summary["input_scope"]["selected_task"] == task
    assert [row["id"] for row in read_rows(tmp_path / "task")] == [record["id"] for record in records[:3]]
    assert fake_backend.calls[0][1] == "/local/instruct"
    assert all(call[2] == {"max_new_tokens": 30} for call in fake_backend.calls[1:])


def test_default_all_rows_and_partial_recall_aggregates(tmp_path, monkeypatch, fake_backend):
    values = iter([1 / 3] * 10 + [0.5] * 10 + [1.0] * 10 + [0.0] * 10)
    monkeypatch.setattr(runner, "score", lambda *_: next(values))
    summary = runner.run(tmp_path / "full")
    assert summary["evaluated"] == 40
    assert summary["subset"] is False
    assert summary["mean_score"] == pytest.approx(11 / 24)
    assert summary["score_percent"] == 45.83
    assert summary["per_task"]["niah_single_1"] == {
        "evaluated": 10, "mean_score": pytest.approx(1 / 3), "score_percent": 33.33,
    }
    assert "correct" not in summary
    assert {call[2]["max_new_tokens"] for call in fake_backend.calls[1:]} == {128, 30, 50}


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": -1}, {"limit": True}, {"limit": 1.5}, {"limit": "2"},
    {"limit": 41}, {"task": "vt", "limit": 11}, {"task": "unknown"}, {"task": []},
    {"model_name": ""}, {"model_name": " "}, {"model_name": None},
])
def test_invalid_arguments_precede_gpu_and_output(tmp_path, fake_backend, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        runner.run(output, **kwargs)
    assert fake_backend.calls == []
    assert not output.exists()


def test_complete_data_validation_precedes_task_filter_gpu_and_output(tmp_path, monkeypatch, fake_backend):
    def invalid_data(*args, **kwargs):
        assert kwargs.get("task") is None
        raise ValueError("unselected task pin mismatch")

    monkeypatch.setattr(runner, "load_dataset", invalid_data)
    with pytest.raises(ValueError, match="pin"):
        runner.run(tmp_path / "invalid", task="vt", limit=1)
    assert fake_backend.calls == []
    assert not (tmp_path / "invalid").exists()


def test_existing_output_is_never_overwritten(tmp_path, fake_backend):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "predictions.jsonl"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        runner.run(output, limit=1)
    assert sentinel.read_text() == "preserve"
    assert fake_backend.calls == []


def test_generation_failure_keeps_flushed_partial_without_summary(tmp_path, monkeypatch, fake_backend):
    predictions = iter(["answer", RuntimeError("GPU failure")])

    def generate(*args, **kwargs):
        value = next(predictions)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(runner.kivi_llama, "generate", generate)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="GPU failure"):
        runner.run(output, limit=3)
    assert len(read_rows(output)) == 1
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("context", [130, None, 0, True])
def test_context_overflow_or_invalid_limit_never_truncates(tmp_path, fake_backend, context):
    fake_backend.model.config.max_position_embeddings = context
    output = tmp_path / "too_long"
    with pytest.raises(ValueError, match="context|max_position_embeddings"):
        runner.run(output, limit=1)
    assert len(fake_backend.calls) == 1  # model load only; no generation
    assert read_rows(output) == []
    assert not (output / "summary.json").exists()


def test_exact_context_boundary_is_accepted(tmp_path, fake_backend):
    fake_backend.model.config.max_position_embeddings = 131
    assert runner.run(tmp_path / "boundary", limit=1)["evaluated"] == 1


@pytest.mark.parametrize("arguments", [
    [str(Path(__file__).resolve().parents[1] / "src/experiments/quality/RULER/KIVI_Llama.py")],
    ["-m", "experiments.quality.RULER.KIVI_Llama"],
])
def test_cli_from_other_cwd_is_cpu_safe(tmp_path, arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    for flags, expected, text in [(["--help"], 0, "--task"), (["--limit", "0"], 2, "positive integer"),
                                  (["--task", "unknown"], 2, "invalid choice")]:
        result = subprocess.run([sys.executable, *arguments, *flags], cwd=tmp_path,
                                env=environment, capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
        assert text in result.stdout + result.stderr


def test_import_keeps_gpu_dependencies_unloaded():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from experiments.quality.RULER import KIVI_Llama; "
        "assert not ({'torch', 'transformers', 'triton', 'flash_attn'} & set(sys.modules))",
    ], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
