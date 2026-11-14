"""CPU-only boundary and failure contracts; fixture energy is not real power."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.quality.RULER import KIVI_Llama as runner


def rows(path, name="predictions.jsonl"):
    return [json.loads(line) for line in (path / name).read_text().splitlines()]


@pytest.fixture
def energy_backend(monkeypatch):
    state = SimpleNamespace(events=[], active=False, meters=[], fail=None,
                            save_fail=False, invalid_at=None, generated=0)
    records = [dict(id=f"fixture-{i}", task=task, prompt=f"prompt-{i}",
                    answers=["answer"], max_new_tokens=cap,
                    source_index=i, source_length=4096)
               for i, (task, cap) in enumerate([
                   ("niah_single_1", 128), ("niah_single_1", 128), ("vt", 30)])]
    state.model = SimpleNamespace(
        device="cuda:3", config=SimpleNamespace(max_position_embeddings=10000),
        generation_config=SimpleNamespace())
    state.profile = SimpleNamespace(to_dict=lambda: {"name": "fixture-not-hardware"})

    def discover():
        assert not state.active
        state.events.append("discover")
        return state.profile

    def tokenize(prompt, **kwargs):
        assert not state.active
        state.events.append("pretokenize")
        return {"input_ids": SimpleNamespace(tolist=lambda: [[1, 2, 3]])}

    def load(*args, **kwargs):
        state.events.append("load")
        return state.model, tokenize

    def generate(*args, **kwargs):
        state.generated += 1
        state.events.append(("generate", state.active, args[2], kwargs))
        if state.fail == "generate" and state.generated == 2:
            raise RuntimeError("original workload failure")
        return "answer"

    def score(*args):
        assert not state.active
        state.events.append("score")
        return 1.0

    def synchronize(device):
        assert device == state.model.device
        state.events.append(("sync", device))
        if state.fail == "sync_start" and not state.active:
            raise RuntimeError("original start sync failure")
        if state.fail == "sync_stop" and state.active:
            raise RuntimeError("original stop sync failure")

    class Meter:
        def __init__(self, profile, *, synchronize):
            assert profile is state.profile
            assert not state.active
            self.synchronize = synchronize
            self.result = None
            self.index = len(state.meters)
            state.meters.append(self)

        def start(self):
            state.events.append("start")
            try:
                if state.fail == "start":
                    raise RuntimeError("original collector startup failure")
                self.synchronize()
            except BaseException:
                self.result = {"valid": False, "energy_j": None, "reason": "start_failed"}
                raise
            state.active = True
            return self

        def stop(self, *, completed=True):
            try:
                self.synchronize()
            except BaseException:
                self.result = {"valid": False, "energy_j": None, "reason": "sync_failed"}
                if completed:
                    raise
            finally:
                state.active = False
                state.events.append("stop")
            valid = completed and self.index != state.invalid_at
            self.result = {"valid": valid, "energy_j": 2.5 if valid else None,
                           "reason": None if valid else "fixture_invalid",
                           "sensor_response_validated": False}
            return self.result

        def __enter__(self):
            return self.start()

        def __exit__(self, exc_type, exc, traceback):
            self.stop(completed=exc_type is None)
            return False

        def save(self, path):
            assert not state.active
            assert self.result is not None
            state.events.append("save")
            if state.save_fail:
                raise OSError("fixture save failure")
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "energy.json").write_text(json.dumps(self.result))
            (path / "power-samples.csv").write_text("fixture CSV\n")
            (path / "power-trace.jsonl").write_text("fixture trace\n")

    monkeypatch.setattr(runner, "load_dataset", lambda: records)
    monkeypatch.setattr(runner, "discover_power_profile", discover, raising=False)
    monkeypatch.setattr(runner, "EnergyMeter", Meter, raising=False)
    monkeypatch.setattr(runner.kivi_llama, "load_model", load)
    monkeypatch.setattr(runner.kivi_llama, "generate", generate)
    monkeypatch.setattr(runner, "score", score)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)))
    return state


def test_energy_boundary_profile_reuse_and_artifacts(tmp_path, energy_backend):
    state = energy_backend
    summary = runner.run(tmp_path / "run", measure_energy=True)
    assert state.events[:2] == ["discover", "load"]
    assert state.events.count("discover") == 1
    assert state.events.count(("sync", "cuda:3")) == 6
    assert len(state.meters) == 3
    assert all(event[1] for event in state.events if isinstance(event, tuple) and event[0] == "generate")
    for index, row in enumerate(rows(tmp_path / "run")):
        assert row["energy"] == state.meters[index].result
        relative = row["energy_artifacts_dir"]
        assert relative == f"energy/{index:06d}"
        assert {p.name for p in (tmp_path / "run" / relative).iterdir()} == {
            "energy.json", "power-samples.csv", "power-trace.jsonl"}
    assert summary["energy"]["total_energy_j"] == 7.5
    assert summary["energy"]["valid_count"] == 3
    assert summary["energy"]["invalid_count"] == 0
    assert "tokenization" in summary["energy"]["boundary"]
    assert "text decoding" in summary["energy"]["boundary"]
    assert summary["energy"]["power_profile"] == state.profile.to_dict()
    assert summary["warmup"]["completed"] == 0


def test_disabled_has_no_energy_interaction(tmp_path, energy_backend):
    summary = runner.run(tmp_path / "run")
    assert summary["energy"]["enabled"] is False
    assert energy_backend.meters == []
    assert "discover" not in energy_backend.events
    assert not any(isinstance(event, tuple) and event[0] == "sync" for event in energy_backend.events)
    assert all(row.get("energy") is None for row in rows(tmp_path / "run"))
    assert not (tmp_path / "run" / "energy").exists()


def test_invalid_energy_keeps_scores_and_total_null(tmp_path, energy_backend):
    energy_backend.invalid_at = 1
    summary = runner.run(tmp_path / "run", measure_energy=True)
    assert summary["evaluated"] == 3
    assert summary["mean_score"] == 1
    assert summary["energy"]["valid_count"] == 2
    assert summary["energy"]["invalid_count"] == 1
    assert summary["energy"]["total_energy_j"] is None
    assert summary["energy"]["valid_rows_energy_j"] == 5
    assert rows(tmp_path / "run")[1]["energy"]["energy_j"] is None
    energy_backend.invalid_at = len(energy_backend.meters) + 1
    assert runner.main(["--measure-energy", "--output-dir", str(tmp_path / "cli")]) == 2


@pytest.mark.parametrize("failure", ["start", "sync_start", "sync_stop", "generate"])
def test_failure_keeps_trace_original_exception_and_partial(tmp_path, energy_backend, failure):
    energy_backend.fail = failure
    with pytest.raises(RuntimeError, match="original"):
        runner.run(tmp_path / "run", measure_energy=True)
    output = tmp_path / "run"
    index = 1 if failure == "generate" else 0
    assert len(rows(output)) == index
    assert (output / f"energy/{index:06d}/power-trace.jsonl").exists()
    assert rows(output, "energy-failures.jsonl")[0]["id"] == f"fixture-{index}"
    assert not (output / "summary.json").exists()


def test_workload_error_survives_trace_save_failure(tmp_path, energy_backend, monkeypatch, capsys):
    # First measured row succeeds; second workload and its diagnostic save fail.
    energy_backend.fail = "generate"
    original_generate = runner.kivi_llama.generate

    def generate(*args, **kwargs):
        if energy_backend.generated == 1:
            energy_backend.save_fail = True
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(runner.kivi_llama, "generate", generate)
    with pytest.raises(RuntimeError, match="original workload failure"):
        runner.run(tmp_path / "run", measure_energy=True)
    assert "fixture save failure" in capsys.readouterr().err
    assert len(rows(tmp_path / "run")) == 1
    assert not (tmp_path / "run/summary.json").exists()


def test_normal_save_failure_propagates(tmp_path, energy_backend):
    energy_backend.save_fail = True
    with pytest.raises(OSError, match="fixture save failure"):
        runner.run(tmp_path / "run", measure_energy=True)
    assert not (tmp_path / "run/summary.json").exists()


def test_warmup_each_selected_task_uses_full_cap_outside_meter_and_score(tmp_path, energy_backend):
    summary = runner.run(tmp_path / "run", measure_energy=True, warmup_per_task=2)
    warmups = rows(tmp_path / "run", "warmup.jsonl")
    assert [(row["id"], row["repetition"], row["max_new_tokens"]) for row in warmups] == [
        ("fixture-0", 1, 128), ("fixture-0", 2, 128),
        ("fixture-2", 1, 30), ("fixture-2", 2, 30)]
    assert all(row["prediction"] == "answer" and "score" not in row and "energy" not in row for row in warmups)
    generations = [event for event in energy_backend.events if isinstance(event, tuple) and event[0] == "generate"]
    assert [event[1] for event in generations] == [False, False, True, True, False, False, True]
    assert energy_backend.events.count("score") == 3
    assert summary["warmup"] == {"per_task": 2, "completed": 4, "tasks": ["niah_single_1", "vt"],
                                 "selection": "first selected record of each selected task; original output cap"}


def test_warmup_filter_and_failure_preserve_partial(tmp_path, energy_backend):
    summary = runner.run(tmp_path / "filtered", task="vt", limit=1, warmup_per_task=2)
    assert summary["warmup"]["completed"] == 2
    assert [row["id"] for row in rows(tmp_path / "filtered", "warmup.jsonl")] == ["fixture-2"] * 2
    energy_backend.generated = 0
    energy_backend.fail = "generate"
    with pytest.raises(RuntimeError, match="workload"):
        runner.run(tmp_path / "failed", warmup_per_task=2, measure_energy=True)
    assert len(rows(tmp_path / "failed", "warmup.jsonl")) == 1
    assert rows(tmp_path / "failed") == []
    assert not (tmp_path / "failed/energy").exists()
    assert not (tmp_path / "failed/summary.json").exists()


@pytest.mark.parametrize("kwargs", [
    {"measure_energy": 1}, {"measure_energy": "yes"},
    {"power_profile": "unused.json"}, {"warmup_per_task": -1},
    {"warmup_per_task": True}, {"warmup_per_task": 1.5},
    {"measure_energy": True, "power_profile": "missing.json"},
])
def test_bad_energy_options_precede_gpu_and_output(tmp_path, energy_backend, kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(tmp_path / "bad", **kwargs)
    assert energy_backend.events == []
    assert not (tmp_path / "bad").exists()


def test_profile_discovery_failure_precedes_gpu_and_output(tmp_path, monkeypatch, energy_backend):
    def fail():
        raise RuntimeError("missing rails; no fallback")

    monkeypatch.setattr(runner, "discover_power_profile", fail)
    with pytest.raises(RuntimeError, match="missing rails"):
        runner.run(tmp_path / "bad", measure_energy=True)
    assert energy_backend.events == []
    assert not (tmp_path / "bad").exists()


def test_explicit_profile_preflight_and_no_discovery(tmp_path, monkeypatch, energy_backend):
    path = tmp_path / "profile.json"
    path.write_text("fixture")

    def load_profile(received):
        assert received == path
        assert energy_backend.events == []
        return energy_backend.profile

    monkeypatch.setattr(runner.PowerProfile, "from_json", load_profile)
    assert runner.run(tmp_path / "run", measure_energy=True, power_profile=path)["energy"]["enabled"]
    assert "discover" not in energy_backend.events


@pytest.mark.parametrize("flags", [["--warmup-per-task", "-1"], ["--power-profile", "unused.json"]])
def test_cli_rejects_bad_options_before_output(tmp_path, flags):
    with pytest.raises(SystemExit) as exc:
        runner.main([*flags, "--output-dir", str(tmp_path / "bad")])
    assert exc.value.code == 2
    assert not (tmp_path / "bad").exists()
