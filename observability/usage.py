from observability.callbacks import TokenUsageCallback
from observability.metrics import save_model_usage
from observability.pricing import get_model_pricing


def tracked_invoke(
    runnable,
    payload,
    *,
    run_id: str,
    node_name: str,
    model_name: str = "gpt-4o-mini",
):
    pricing = get_model_pricing(model_name)
    tracker = TokenUsageCallback()

    result = runnable.invoke(
        payload,
        config={"callbacks": [tracker]},
    )

    usage = tracker.get_summary()
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    if prompt_tokens or completion_tokens:
        save_model_usage(
            run_id=run_id,
            node_name=node_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_price_per_million=pricing["input"],
            output_price_per_million=pricing["output"],
        )

    return result