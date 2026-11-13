"""Minimal KIVI + Llama-3.1-8B quality experiment; dataset and metric are TODO."""

MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Base model, not Instruct.
K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
MAX_NEW_TOKENS = 128


def load_dataset() -> list[dict[str, str]]:
    """Return records with a complete base-model prompt and a reference answer."""
    # TODO: Choose a dataset and map its rows to {"prompt": str, "answer": str}.
    raise NotImplementedError("TODO: choose and load the quality dataset")


def load_model():
    """Load FP16 weights with the project's KIVI KV-cache implementation."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig

    from kvolt_runtime.backend.vendor.llama_kivi import LlamaForCausalLM_KIVI

    config = LlamaConfig.from_pretrained(MODEL_NAME)
    config.k_bits = K_BITS
    config.v_bits = V_BITS
    config.group_size = GROUP_SIZE
    config.residual_length = RESIDUAL_LENGTH
    config.use_cache = True
    config.use_flash = True
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"

    model = LlamaForCausalLM_KIVI.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> str:
    """Generate one answer, preserving natural EOS and decoding only new tokens."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)


def score(prediction: str, answer: str) -> float:
    """Apply the chosen dataset's metric to one prediction and reference answer."""
    # TODO: Implement the metric after choosing the dataset.
    raise NotImplementedError("TODO: define the quality metric")


def main():
    dataset = load_dataset()  # Resolve dataset TODO before GPU imports/downloads.
    if not dataset:
        raise ValueError("quality dataset must not be empty")
    model, tokenizer = load_model()
    for index, sample in enumerate(dataset, start=1):
        prediction = generate(model, tokenizer, sample["prompt"])
        result = score(prediction, sample["answer"])
        print(f"\nExample {index}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Answer: {sample['answer']}")
        print(f"Prediction: {prediction}")
        print(f"Score: {result}")


if __name__ == "__main__":
    main()
