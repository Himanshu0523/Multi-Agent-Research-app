def calculate_estimated_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:

    input_cost = (
        prompt_tokens / 1_000_000
    ) * input_price_per_million

    output_cost = (
        completion_tokens / 1_000_000
    ) * output_price_per_million

    return input_cost + output_cost