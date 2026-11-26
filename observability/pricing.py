import os


MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": float(os.getenv("GPT_4O_MINI_INPUT_PRICE", "0")),
        "output": float(os.getenv("GPT_4O_MINI_OUTPUT_PRICE", "0")),
    }
}


def get_model_pricing(model_name: str) -> dict:
    pricing = MODEL_PRICING.get(model_name)

    if pricing is None:
        raise ValueError(
            f"No pricing configuration found for model: {model_name}"
        )

    return pricing