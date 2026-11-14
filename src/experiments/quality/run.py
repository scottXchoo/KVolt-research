"""Minimal KIVI + Llama-3.1-8B-Instruct quality experiment; data and metric are TODO."""

from model.kivi_llama import load_model, generate

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
MAX_NEW_TOKENS = 128


def load_dataset() -> list[dict[str, str]]:
    """Return complete Instruct-model inputs and separate reference answers.

    The experiment loader prepares its input format, including any chat template.
    The shared generation helper receives that finished prompt unchanged.
    """
    # TODO: Choose a dataset and map its rows to {"prompt": str, "answer": str}.
    raise NotImplementedError("TODO: choose and load the quality dataset")


def score(prediction: str, answer: str) -> float:
    """Apply the chosen dataset's metric to one prediction and reference answer."""
    # TODO: Implement the metric after choosing the dataset.
    raise NotImplementedError("TODO: define the quality metric")


def main():
    dataset = load_dataset()  # Resolve dataset TODO before GPU imports/downloads.
    if not dataset:
        raise ValueError("quality dataset must not be empty")
    model, tokenizer = load_model(
        MODEL_NAME,
        k_bits=K_BITS,
        v_bits=V_BITS,
        group_size=GROUP_SIZE,
        residual_length=RESIDUAL_LENGTH,
    )
    for index, sample in enumerate(dataset, start=1):
        prediction = generate(
            model, tokenizer, sample["prompt"], max_new_tokens=MAX_NEW_TOKENS
        )
        result = score(prediction, sample["answer"])
        print(f"\nExample {index}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Answer: {sample['answer']}")
        print(f"Prediction: {prediction}")
        print(f"Score: {result}")


if __name__ == "__main__":
    main()
