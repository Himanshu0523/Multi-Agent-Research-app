from observability.callbacks import TokenUsageCallback
from observability.metrics import save_model_usage
from observability.pricing import get_model_pricing

def invoke_with_usage(
    runnable,
    payload: dict,
    *,
    run_id: str,
    node_name: str,
    model_name: str,
    input_price_per_million: float,
    output_price_per_million: float,
):
    """
    Invoke a LangChain Runnable and persist its token usage.

    Supports AgentExecutor and LCEL Runnables that accept
    invoke(payload, config=...).
    """

    tracker = TokenUsageCallback()

    result = runnable.invoke(
        payload,
        config={"callbacks": [tracker]},
    )

    usage = tracker.get_summary()

    save_model_usage(
        run_id=run_id,
        node_name=node_name,
        model_name=model_name,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
    )

    return result


def tracked_invoke(
    runnable,
    payload: dict,
    *,
    run_id: str,
    node_name: str,
    model_name: str = "gpt-4o-mini",
):
    pricing = get_model_pricing(model_name)

    return invoke_with_usage(
        runnable,
        payload,
        run_id=run_id,
        node_name=node_name,
        model_name=model_name,
        input_price_per_million=pricing["input"],
        output_price_per_million=pricing["output"],
    )