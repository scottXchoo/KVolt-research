"""Generation-only adapter contracts; no model downloads or GPU required."""

import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("lm_eval.api.model")

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import get_model

from experiments.lm_eval.backend import KIVILM
from model import kivi_llama


def request(prompt="question", kwargs=None, doc_id=0):
    return Instance("generate_until", {}, (prompt, kwargs or {}), 0,
                    metadata=("local", doc_id, 1))


@pytest.fixture
def backend(monkeypatch, tmp_path):
    calls = []
    model = SimpleNamespace(device="cpu", config=SimpleNamespace(max_position_embeddings=100))
    tokenizer = SimpleNamespace(eos_token_id=2)

    def generate(model_arg, tokenizer_arg, prompt, **kwargs):
        assert model_arg is model and tokenizer_arg is tokenizer
        calls.append((prompt, kwargs))
        return {"text": "42", "raw_text": "42STOP", "input_token_ids": [10, 11],
                "output_token_ids": [20, 21, 22], "input_tokens": 2, "output_tokens": 3,
                "stop_reason": "stop_sequence", "matched_stop": "STOP",
                "energy": None, "energy_receipt": None,
                "generation": {"max_new_tokens": kwargs["max_new_tokens"]},
                "boundary": "test boundary"}

    monkeypatch.setattr(kivi_llama, "generate_with_details", generate, raising=False)
    lm = KIVILM(model=model, tokenizer=tokenizer, output_dir=tmp_path / "run",
                max_input_tokens=80, max_new_tokens=10)
    return SimpleNamespace(lm=lm, calls=calls, output=tmp_path / "run")


def test_registered_real_lm_preserves_request_order_and_raw_tokens(backend):
    assert isinstance(backend.lm, LM)
    assert get_model("kivi-llama") is KIVILM
    result = backend.lm.generate_until([
        request("longer question", {"until": "STOP", "max_gen_toks": 9}, 4),
        request("x", {"until": ["STOP"], "max_new_tokens": 8}, 2),
        request("x", {"until": ["STOP"], "max_new_tokens": 8}, 2),
    ])
    assert result == ["42", "42", "42"]
    assert [call[0] for call in backend.calls] == ["longer question", "x", "x"]
    assert [row["doc_id"] for row in backend.lm.records] == [4, 2, 2]
    rows = [json.loads(line) for line in (backend.output / "records.jsonl").read_text().splitlines()]
    assert rows == backend.lm.records
    assert rows[0]["prompt_sha256"] == hashlib.sha256(b"longer question").hexdigest()
    assert rows[0]["output_token_ids"] == [20, 21, 22]
    assert rows[0]["output_tokens"] == 3
    assert rows[0]["raw_text"] == "42STOP" and rows[0]["text"] == "42"
    assert rows[0]["matched_stop"] == "STOP"
    assert backend.calls[0][1]["until"] == ("STOP",)
    assert backend.lm.rank == 0 and backend.lm.world_size == 1
    assert backend.lm.batch_size == 1 and backend.lm.tokenizer_name.endswith("Instruct")


@pytest.mark.parametrize("kwargs", [
    {"do_sample": True}, {"num_beams": 2}, {"temperature": 0.1},
    {"top_p": 0.9}, {"batch_size": "auto"}, {"truncate": True},
    {"max_gen_toks": True}, {"max_new_tokens": 0}, {"max_new_tokens": 2.0},
    {"max_gen_toks": 2, "max_new_tokens": 3}, {"until": [""]},
    {"until": 1}, {"use_cache": False}, {"do_sample": 0}, {"num_beams": True},
    {"temperature": float("nan")},
])
def test_unsupported_generation_fails_before_model_call(backend, kwargs):
    with pytest.raises((TypeError, ValueError)):
        backend.lm.generate_until([request(kwargs=kwargs)])
    assert backend.calls == []


def test_only_generate_until_is_supported(backend):
    with pytest.raises(NotImplementedError, match="generate_until"):
        backend.lm.loglikelihood([])
    with pytest.raises(NotImplementedError, match="generate_until"):
        backend.lm.loglikelihood_rolling([])
    assert backend.lm.generate_until([]) == []
    with pytest.raises(ValueError, match="prompt"):
        backend.lm.generate_until([request("")])


@pytest.mark.parametrize("kwargs", [
    {"group_size": 7}, {"residual_length": 0}, {"residual_length": 33},
    {"max_new_tokens": False}, {"max_input_tokens": float("inf")},
    {"measure_energy": 1}, {"power_profile": "profile.json"},
    {"model": object()}, {"tokenizer": object()},
])
def test_configuration_fails_before_loading(kwargs):
    with pytest.raises((TypeError, ValueError)):
        KIVILM(**kwargs)


def test_energy_artifacts_keep_exact_trace_and_receipt_hash(backend, monkeypatch):
    raw = '{"time_ns":1,"rails":{}}\n'
    sha = hashlib.sha256(raw.encode()).hexdigest()
    original = kivi_llama.generate_with_details

    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        result["energy"] = {"valid": False, "energy_j": None, "trace_sha256": sha}
        result["energy_receipt"] = {"raw_trace_jsonl": raw, "samples": [], "trace_sha256": sha}
        return result

    monkeypatch.setattr(kivi_llama, "generate_with_details", measured)
    backend.lm.generate_until([request()])
    root = backend.output / "energy/000000"
    assert (root / "power-trace.jsonl").read_bytes() == raw.encode()
    assert (root / "power-samples.csv").exists()
    assert json.loads((root / "energy.json").read_text())["energy_j"] is None
    assert backend.lm.records[0]["energy_receipt"]["trace_sha256"] == sha


def test_failure_keeps_original_exception_and_partial_energy(backend, monkeypatch):
    error = RuntimeError("generation failed")
    error.kvolt_energy_result = {"valid": False, "energy_j": None}
    error.kvolt_energy_receipt = {"samples": [], "raw_trace_jsonl": "raw\n"}

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(kivi_llama, "generate_with_details", fail)
    with pytest.raises(RuntimeError) as caught:
        backend.lm.generate_until([request(doc_id=8)])
    assert caught.value is error
    assert (backend.output / "energy/000000/power-trace.jsonl").read_text() == "raw\n"
    rows = [json.loads(line) for line in (backend.output / "records.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "failed" and rows[0]["doc_id"] == 8
    assert rows[0]["error"]["message"] == "generation failed"


def test_in_memory_adapter_retains_raw_energy_receipt(backend, monkeypatch):
    original = kivi_llama.generate_with_details
    receipt = {"raw_trace_jsonl": "raw\n", "samples": []}

    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        result["energy"] = {"valid": False, "energy_j": None}
        result["energy_receipt"] = receipt
        return result

    monkeypatch.setattr(kivi_llama, "generate_with_details", measured)
    backend.lm.output_dir = None
    backend.lm.generate_until([request()])
    assert backend.lm.records[0]["energy_receipt"] == receipt


def test_existing_records_are_never_overwritten(tmp_path):
    (tmp_path / "records.jsonl").write_text("old\n")
    with pytest.raises(FileExistsError):
        KIVILM(model=object(), tokenizer=object(), output_dir=tmp_path)
    assert (tmp_path / "records.jsonl").read_text() == "old\n"


def test_task_generation_limit_is_capped_by_constructor(backend):
    backend.lm.generate_until([request(kwargs={"max_gen_toks": 100})])
    assert backend.calls[0][1]["max_new_tokens"] == 10
    assert backend.lm.records[0]["requested_generation"]["max_gen_toks"] == 100
    assert backend.lm.records[0]["effective_generation"]["max_new_tokens"] == 10


@pytest.mark.parametrize("mismatch", ["class", "k_bits", "v_bits", "group_size", "residual_length"])
def test_real_loader_rejects_backend_or_kv_configuration_mismatch(monkeypatch, mismatch):
    model_type = type("LlamaForCausalLM_KIVI", (),
                      {"__module__": "kvolt_runtime.backend.vendor.llama_kivi"})
    model = model_type() if mismatch != "class" else SimpleNamespace()
    model.config = SimpleNamespace(k_bits=2, v_bits=2, group_size=32, residual_length=128)
    if mismatch != "class":
        setattr(model.config, mismatch, 999)
    monkeypatch.setattr(kivi_llama, "load_model", lambda *args, **kwargs: (model, object()))
    lm = KIVILM()
    with pytest.raises(ValueError, match="loaded"):
        lm.generate_until([request()])
    assert lm.model is None


def test_real_loader_is_lazy_and_preserves_kivi_configuration(backend, monkeypatch):
    model_type = type("LlamaForCausalLM_KIVI", (),
                      {"__module__": "kvolt_runtime.backend.vendor.llama_kivi"})
    model = model_type()
    model.device = "cuda:0"
    model.config = SimpleNamespace(k_bits=2, v_bits=2, group_size=32, residual_length=128)
    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return model, backend.lm.tokenizer

    monkeypatch.setattr(kivi_llama, "load_model", load)
    lm = KIVILM()
    assert calls == []
    assert lm.device == "cuda:0"
    assert calls == [(lm.model_name, {"k_bits": 2, "v_bits": 2, "group_size": 32,
                                    "residual_length": 128})]
