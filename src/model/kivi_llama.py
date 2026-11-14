"""Reusable KIVI Llama loading and generation; importing is CPU safe."""


def load_model(model_name, *, k_bits=2, v_bits=2, group_size=32, residual_length=128):
    """Load FP16 weights with the project's KIVI KV-cache implementation."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig

    from kvolt_runtime.backend.vendor.llama_kivi import LlamaForCausalLM_KIVI

    config = LlamaConfig.from_pretrained(model_name)
    config.k_bits = k_bits
    config.v_bits = v_bits
    config.group_size = group_size
    config.residual_length = residual_length
    config.use_cache = True
    config.use_flash = True
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"

    model = LlamaForCausalLM_KIVI.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate(model, tokenizer, prompt: str, *, max_new_tokens=128) -> str:
    """Generate one answer, preserving natural EOS and decoding only new tokens."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)


GENERATION_BOUNDARY = (
    "Synchronized single-request generation: tokenization, input transfer, KIVI "
    "model.generate including stop-string checks, raw continuation decoding and "
    "CUDA completion; excludes model load, preflight tokenization/context checks, "
    "warmup, raw token-ID observation copies, scoring and artifact I/O."
)


def generate_with_details(model, tokenizer, prompt: str, *, max_new_tokens=512,
                          until=(), max_input_tokens=None, measure_energy=False,
                          power_profile=None):
    """Generate one greedy continuation with real stopping and original token IDs.

    This opt-in API leaves ``generate`` and historical evaluation unchanged.
    Stop strings apply only to newly generated text. The returned ``text`` omits
    the earliest stop string. ``raw_text`` retains the decoded continuation;
    token IDs retain every generated token, including stop/EOS tokens. No text
    retokenization estimates the output count. Input truncation is never used.
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if max_input_tokens is not None and (type(max_input_tokens) is not int or max_input_tokens <= 0):
        raise ValueError("max_input_tokens must be a positive integer or None")
    if not isinstance(until, (list, tuple)) or any(not isinstance(s, str) or not s for s in until):
        raise ValueError("until must be a list/tuple of nonempty stop strings")
    if type(measure_energy) is not bool:
        raise ValueError("measure_energy must be bool")
    if power_profile is not None and not measure_energy:
        raise ValueError("power_profile requires measure_energy")
    stops = list(dict.fromkeys(until))
    # Validate the exact tokenizer defaults before the measurement interval.
    preflight = tokenizer(prompt, return_tensors="pt", truncation=False)
    before_ids = preflight["input_ids"]
    if len(before_ids.shape) != 2 or before_ids.shape[0] != 1 or before_ids.shape[1] == 0:
        raise ValueError("generation requires one nonempty input sequence")
    input_length = before_ids.shape[1]
    if max_input_tokens is not None and input_length > max_input_tokens:
        raise ValueError("input token limit exceeded; truncation is disabled")
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if type(context_limit) is not int or context_limit <= 0:
        raise ValueError("model max_position_embeddings must be a positive integer")
    if input_length + max_new_tokens > context_limit:
        raise ValueError("input plus output allowance exceeds model context; truncation is disabled")
    expected_input_ids = before_ids[0].tolist()

    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopContinuation(StoppingCriteria):
        matched_stop = None

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids.shape[0] != 1:
                raise ValueError("KIVI evaluation supports batch size 1 only")
            new_text = tokenizer.decode(input_ids[0, input_length:], skip_special_tokens=False)
            matches = [(new_text.find(stop), index, stop) for index, stop in enumerate(stops)
                       if stop in new_text]
            if matches:
                self.matched_stop = min(matches)[2]
                return True
            return False

    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = [] if eos is None else ([eos] if type(eos) is int else list(eos))
    if any(type(token) is not int or token < 0 for token in eos_ids):
        raise ValueError("EOS token IDs must be nonnegative integers")
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None and eos_ids:
        pad = eos_ids[0]
    stopper = StopContinuation()
    criteria = StoppingCriteriaList([stopper] if stops else [])
    meter = None
    if measure_energy:
        # Lazy import keeps the ordinary generation path free of sensor access.
        from kvolt_runtime.telemetry.experiment import EnergyMeter
        meter = EnergyMeter(power_profile,
                            synchronize=lambda: torch.cuda.synchronize(model.device))

    def workload():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
        if tuple(inputs["input_ids"].shape) != (1, input_length):
            raise ValueError("tokenizer changed input shape after preflight")
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, num_return_sequences=1, return_dict_in_generate=False,
                use_cache=True, pad_token_id=pad, eos_token_id=eos,
                stopping_criteria=criteria,
            )
        if len(output.shape) != 2 or output.shape[0] != 1:
            raise ValueError("KIVI generation must return a single token tensor")
        raw_text = tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
        return output, inputs["input_ids"], raw_text

    try:
        if meter is None:
            output, actual_inputs, raw_text = workload()
        else:
            with meter:
                output, actual_inputs, raw_text = workload()
        # Additional evidence copies and all file writes are outside the interval.
        input_ids = actual_inputs[0].detach().cpu().tolist()
        output_ids = output[0, input_length:].detach().cpu().tolist()
        if input_ids != expected_input_ids:
            raise ValueError("tokenizer changed input tokens after preflight")
        if output[0, :input_length].detach().cpu().tolist() != input_ids:
            raise ValueError("generation returned a changed input prefix")
        if not 1 <= len(output_ids) <= max_new_tokens:
            raise ValueError("generation returned an invalid output token count")
        matches = [(raw_text.find(stop), index, stop) for index, stop in enumerate(stops)
                   if stop in raw_text]
        match = min(matches) if matches else None
        is_eos = output_ids[-1] in eos_ids
        observed_stop = match[2] if match else stopper.matched_stop
        reason = "eos" if is_eos else "stop_sequence" if observed_stop else "max_new_tokens"
        if reason == "max_new_tokens" and len(output_ids) != max_new_tokens:
            raise ValueError("generation ended before EOS, stop string or token allowance")
        return {
            "text": raw_text[:match[0]] if match else raw_text,
            "raw_text": raw_text, "input_token_ids": input_ids,
            "output_token_ids": output_ids, "input_tokens": len(input_ids),
            "output_tokens": len(output_ids), "stop_reason": reason,
            "matched_stop": observed_stop if not is_eos else None,
            "energy": meter.result if meter is not None else None,
            "energy_receipt": meter.receipt if meter is not None else None,
            "generation": dict(max_new_tokens=max_new_tokens, until=stops,
                               do_sample=False, num_beams=1, use_cache=True),
            "boundary": GENERATION_BOUNDARY,
        }
    except BaseException as original:
        if meter is not None:
            original.kvolt_energy_result = meter.result
            original.kvolt_energy_receipt = meter.receipt
        raise
