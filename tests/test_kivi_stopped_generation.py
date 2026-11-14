"""CPU HF generation contracts; scripted logits are not model-quality evidence."""

from types import SimpleNamespace

import pytest

from model import kivi_llama


@pytest.fixture
def toy():
    """Use the real HF generate loop with a tiny local model and known tokens."""
    import torch
    from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel, LogitsProcessor

    state = SimpleNamespace(events=[], active=False, calls=[], generated_steps=0,
                            completion="answer<END>unnecessary", fail=None)

    class TrackedInputs(BatchEncoding):
        def to(self, *args, **kwargs):
            state.events.append(("input_to_device", state.active))
            return super().to(*args, **kwargs)

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0
        model_max_length = 64

        def __call__(self, prompt, **kwargs):
            state.events.append(("tokenize", state.active, dict(kwargs)))
            ids = [ord(character) + 3 for character in prompt]
            if kwargs.get("return_tensors") == "pt":
                return TrackedInputs({"input_ids": torch.tensor([ids], dtype=torch.long),
                                      "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)})
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        def decode(self, ids, skip_special_tokens=True, **kwargs):
            state.events.append(("decode", state.active))
            values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            return "".join(chr(value - 3) if value >= 3 else
                           "<SPECIAL>" if value == 1 and not skip_special_tokens else ""
                           for value in values)

    class ForceContinuation(LogitsProcessor):
        def __init__(self, prefix_length):
            self.prefix_length = prefix_length

        def __call__(self, input_ids, scores):
            offset = input_ids.shape[1] - self.prefix_length
            completion = ([ord(character) + 3 for character in state.completion]
                          if isinstance(state.completion, str) else state.completion)
            token = completion[offset] if offset < len(completion) else 2
            scores.fill_(-float("inf"))
            scores[:, token] = 0
            state.generated_steps += 1
            return scores

    class ScriptedModel(GPT2LMHeadModel):
        def generate(self, *args, **kwargs):
            state.calls.append(dict(kwargs))
            state.events.append(("generate", state.active))
            if state.fail is not None:
                raise state.fail
            prefix = kwargs["input_ids"].shape[1]
            kwargs["logits_processor"] = [ForceContinuation(prefix)]
            return super().generate(*args, **kwargs)

    model = ScriptedModel(GPT2Config(
        vocab_size=256, n_positions=64, n_ctx=64, n_embd=8,
        n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )).eval()
    state.model = model
    state.tokenizer = Tokenizer()
    return state


def run(toy, prompt="Q", **kwargs):
    return kivi_llama.generate_with_details(toy.model, toy.tokenizer, prompt, **kwargs)


def token_ids(text):
    return [ord(character) + 3 for character in text]


def test_multitoken_stop_halts_real_hf_loop_and_preserves_raw_evidence(toy):
    result = run(toy, max_new_tokens=40, until=("<END>",))
    assert result["text"] == "answer"
    assert result["raw_text"] == "answer<END>"
    assert result["output_token_ids"] == token_ids("answer<END>")
    assert result["input_token_ids"] == token_ids("Q")
    assert result["input_tokens"] == 1
    assert result["output_tokens"] == len("answer<END>")
    assert toy.generated_steps == len("answer<END>")
    assert result["stop_reason"] == "stop_sequence"
    assert result["matched_stop"] == "<END>"
    assert result["energy"] is None and result["energy_receipt"] is None
    assert result["generation"] == dict(max_new_tokens=40, until=["<END>"],
                                        do_sample=False, num_beams=1, use_cache=True)
    assert isinstance(result["boundary"], str) and result["boundary"]
    assert toy.calls[0]["input_ids"].shape[0] == 1
    assert toy.calls[0]["stopping_criteria"]


def test_prompt_stop_and_partial_stop_do_not_halt_continuation(toy):
    toy.completion = "<ENx<END>tail"
    result = run(toy, prompt="<END>", max_new_tokens=30, until=("<END>",))
    assert result["text"] == "<ENx"
    assert result["raw_text"] == "<ENx<END>"
    assert toy.generated_steps == len("<ENx<END>")


def test_stop_cannot_span_prompt_and_continuation(toy):
    toy.completion = "ND>abc"
    result = run(toy, prompt="<E", max_new_tokens=4, until=("<END>",))
    assert result["text"] == "ND>a"
    assert result["stop_reason"] == "max_new_tokens"


def test_special_token_stop_keeps_match_when_visible_decode_omits_it(toy):
    toy.completion = [1] + token_ids("unnecessary")
    result = run(toy, max_new_tokens=20, until=("<SPECIAL>",))
    assert result["text"] == result["raw_text"] == ""
    assert result["output_token_ids"] == [1]
    assert result["output_tokens"] == toy.generated_steps == 1
    assert result["stop_reason"] == "stop_sequence"
    assert result["matched_stop"] == "<SPECIAL>"


@pytest.mark.parametrize("completion,until,text,raw,steps,matched", [
    ("!unused", ("!",), "", "!", 1, "!"),
    ("abcSTOPtail", ("unused", "STOP"), "abc", "abcSTOP", 7, "STOP"),
    ("ab!extra", ("b!", "!"), "a", "ab!", 3, "b!"),
])
def test_first_completed_stop_and_earliest_text_cut(toy, completion, until, text, raw, steps, matched):
    toy.completion = completion
    result = run(toy, max_new_tokens=30, until=until)
    assert result["text"] == text
    assert result["raw_text"] == raw
    assert result["output_tokens"] == toy.generated_steps == steps
    assert result["matched_stop"] == matched


@pytest.mark.parametrize("completion", [[2, 100], token_ids("ok") + [2, 100]])
def test_natural_eos_preserves_actual_token_count(toy, completion):
    toy.completion = completion
    result = run(toy, max_new_tokens=10)
    assert result["stop_reason"] == "eos"
    assert result["matched_stop"] is None
    assert result["output_token_ids"] == completion[:completion.index(2) + 1]
    assert result["output_tokens"] == toy.generated_steps == completion.index(2) + 1
    assert result["text"] == result["raw_text"]


def test_generation_config_eos_list_is_respected(toy):
    toy.model.generation_config.eos_token_id = [2, 4]
    toy.completion = [4, 100]
    result = run(toy, max_new_tokens=4)
    assert result["output_token_ids"] == [4]
    assert result["stop_reason"] == "eos"


def test_tokenizer_eos_fallback_when_generation_config_is_unset(toy):
    toy.model.generation_config.eos_token_id = None
    toy.model.config.eos_token_id = None
    toy.completion = [2, 100]
    result = run(toy, max_new_tokens=4)
    assert result["output_token_ids"] == [2]
    assert result["stop_reason"] == "eos"


@pytest.mark.parametrize("cap", [1, 3])
def test_token_cap_preserves_untrimmed_continuation(toy, cap):
    toy.completion = "abcdef"
    result = run(toy, max_new_tokens=cap, until=("<END>",))
    assert result["text"] == result["raw_text"] == "abcdef"[:cap]
    assert result["output_tokens"] == toy.generated_steps == cap
    assert result["stop_reason"] == "max_new_tokens"
    assert result["matched_stop"] is None


def test_generation_overrides_mutable_model_defaults_without_mutating_them(toy):
    toy.model.generation_config.num_return_sequences = 3
    toy.model.generation_config.return_dict_in_generate = True
    toy.model.generation_config.num_beams = 3
    toy.model.generation_config.do_sample = True
    result = run(toy, max_new_tokens=1)
    assert result["output_tokens"] == 1
    assert toy.calls[-1]["num_return_sequences"] == 1
    assert toy.calls[-1]["return_dict_in_generate"] is False
    assert toy.calls[-1]["do_sample"] is False
    assert toy.calls[-1]["num_beams"] == 1
    assert toy.model.generation_config.num_return_sequences == 3
    assert toy.model.generation_config.return_dict_in_generate is True


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": 0}, {"max_new_tokens": -1}, {"max_new_tokens": True},
    {"max_new_tokens": 1.5}, {"until": "STOP"}, {"until": None},
    {"until": ("",)}, {"until": (1,)}, {"max_input_tokens": 0},
    {"max_input_tokens": True}, {"max_input_tokens": 1.5}, {"measure_energy": 1},
])
def test_invalid_arguments_fail_before_generation(toy, kwargs):
    with pytest.raises((TypeError, ValueError)):
        run(toy, **kwargs)
    assert not toy.calls


@pytest.mark.parametrize("prompt", ["", None, ["Q"], 2])
def test_invalid_prompt_or_empty_input_is_rejected(toy, prompt):
    with pytest.raises((TypeError, ValueError)):
        run(toy, prompt=prompt, max_new_tokens=1)
    assert not toy.calls


def test_input_limit_rejects_without_truncation(toy):
    with pytest.raises((TypeError, ValueError)):
        run(toy, prompt="12345", max_new_tokens=1, max_input_tokens=4)
    assert not toy.calls
    assert all(not event[2].get("truncation") for event in toy.events if event[0] == "tokenize")


def test_tokenizer_batch_larger_than_one_is_rejected(toy):
    import torch

    toy.tokenizer = lambda *args, **kwargs: {"input_ids": torch.tensor([[40], [41]])}
    with pytest.raises((TypeError, ValueError)):
        run(toy, max_new_tokens=1)
    assert not toy.calls


def test_context_rejects_input_plus_output_and_accepts_exact_boundary(toy):
    toy.model.config.max_position_embeddings = 8
    with pytest.raises((TypeError, ValueError)):
        run(toy, prompt="123456", max_new_tokens=3)
    assert not toy.calls
    result = run(toy, prompt="123456", max_new_tokens=2)
    assert result["input_tokens"] + result["output_tokens"] == 8


def test_disabled_energy_has_no_sensor_or_cuda_sync_calls(toy, monkeypatch):
    import torch
    from kvolt_runtime.telemetry import experiment

    def forbidden(*args, **kwargs):
        pytest.fail("energy-disabled generation touched sensors or CUDA synchronization")

    monkeypatch.setattr(experiment, "discover_power_profile", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    result = run(toy, max_new_tokens=1, measure_energy=False)
    assert result["energy"] is result["energy_receipt"] is None


@pytest.fixture
def fake_meter(toy, monkeypatch):
    from kvolt_runtime.telemetry import experiment

    state = SimpleNamespace(meters=[], startup_failure=None, stop_failure=None, invalid=False)

    class Meter:
        def __init__(self, profile=None, *, enabled=True, synchronize=None):
            self.result = None
            self.receipt = {"samples": [{"fixture": "not real power"}], "raw_trace_jsonl": "fixture"}
            self.synchronize = synchronize
            state.meters.append(self)

        def start(self):
            toy.events.append(("meter_start", toy.active))
            if state.startup_failure is not None:
                self.result = {"valid": False, "energy_j": None, "completed": False,
                               "reason": "start_failed"}
                raise state.startup_failure
            toy.active = True
            return self

        def stop(self, *, completed=True):
            toy.events.append(("meter_stop", toy.active))
            toy.active = False
            valid = completed and not state.invalid and state.stop_failure is None
            self.result = {"valid": valid, "energy_j": 1.0 if valid else None,
                           "completed": completed, "sensor_response_validated": False}
            if state.stop_failure is not None and completed:
                raise state.stop_failure
            return self.result

        def __enter__(self):
            return self.start()

        def __exit__(self, exc_type, exc, traceback):
            self.stop(completed=exc_type is None)
            return False

        def save(self, *args, **kwargs):
            pytest.fail("common helper must leave file IO to the caller")

    monkeypatch.setattr(experiment, "EnergyMeter", Meter)
    if hasattr(kivi_llama, "EnergyMeter"):
        monkeypatch.setattr(kivi_llama, "EnergyMeter", Meter)
    return state


def test_enabled_energy_includes_workload_and_preserves_raw_receipt(toy, fake_meter):
    result = run(toy, max_new_tokens=2, measure_energy=True)
    assert any(event[:2] == ("tokenize", True) for event in toy.events)
    assert ("generate", True) in toy.events
    assert ("input_to_device", True) in toy.events
    assert ("decode", True) in toy.events
    assert toy.events.index(("generate", True)) < toy.events.index(("meter_stop", True))
    assert result["energy"] == fake_meter.meters[0].result
    assert result["energy_receipt"] == fake_meter.meters[0].receipt
    assert not toy.active


def test_raw_id_cpu_observation_copies_happen_after_energy_interval(toy, fake_meter, monkeypatch):
    import torch

    copied_while_active = []
    original_cpu = torch.Tensor.cpu

    def observed_cpu(tensor, *args, **kwargs):
        copied_while_active.append(toy.active)
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", observed_cpu)
    result = run(toy, max_new_tokens=2, measure_energy=True)
    assert result["output_token_ids"] == token_ids("an")
    assert copied_while_active and not any(copied_while_active)


def test_workload_exception_keeps_identity_and_energy_diagnostics(toy, fake_meter):
    failure = RuntimeError("scripted original generation failure")
    toy.fail = failure
    with pytest.raises(RuntimeError) as caught:
        run(toy, max_new_tokens=2, measure_energy=True)
    assert caught.value is failure
    assert caught.value.kvolt_energy_result == fake_meter.meters[0].result
    assert caught.value.kvolt_energy_receipt == fake_meter.meters[0].receipt
    assert caught.value.kvolt_energy_result["completed"] is False
    assert not toy.active


def test_invalid_sensor_result_remains_null_without_discarding_tokens(toy, fake_meter):
    fake_meter.invalid = True
    result = run(toy, max_new_tokens=2, measure_energy=True)
    assert result["output_tokens"] == 2
    assert result["energy"]["valid"] is False
    assert result["energy"]["energy_j"] is None
    assert result["energy_receipt"] == fake_meter.meters[0].receipt


@pytest.mark.parametrize("stage", ["startup_failure", "stop_failure"])
def test_energy_failure_preserves_exception_and_diagnostics(toy, fake_meter, stage):
    failure = RuntimeError("scripted " + stage)
    setattr(fake_meter, stage, failure)
    with pytest.raises(RuntimeError) as caught:
        run(toy, max_new_tokens=2, measure_energy=True)
    assert caught.value is failure
    assert caught.value.kvolt_energy_result == fake_meter.meters[0].result
    assert caught.value.kvolt_energy_receipt == fake_meter.meters[0].receipt
    if stage == "startup_failure":
        assert not toy.calls
    assert not toy.active


def test_legacy_generate_still_returns_string(toy):
    toy.completion = "ok"
    assert kivi_llama.generate(toy.model, toy.tokenizer, "Q", max_new_tokens=4) == "ok"
