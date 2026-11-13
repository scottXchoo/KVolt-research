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
