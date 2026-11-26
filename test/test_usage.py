from unittest.mock import MagicMock, patch

from observability.usage import tracked_invoke


def test_tracked_invoke_saves_usage():
    runnable = MagicMock()
    runnable.invoke.return_value = {"output": "research result"}

    fake_tracker = MagicMock()
    fake_tracker.get_summary.return_value = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }

    with (
        patch(
            "observability.usage.TokenUsageCallback",
            return_value=fake_tracker,
        ),
        patch(
            "observability.usage.get_model_pricing",
            return_value={"input": 0.15, "output": 0.60},
        ),
        patch("observability.usage.save_model_usage") as save_usage,
    ):
        result = tracked_invoke(
            runnable,
            {"input": "test"},
            run_id="test-run",
            node_name="technical_researcher",
        )

    assert result == {"output": "research result"}
    save_usage.assert_called_once_with(
        run_id="test-run",
        node_name="technical_researcher",
        model_name="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        input_price_per_million=0.15,
        output_price_per_million=0.60,
    )


def test_tracked_invoke_skips_empty_usage():
    runnable = MagicMock()
    runnable.invoke.return_value = {"output": "research result"}

    fake_tracker = MagicMock()
    fake_tracker.get_summary.return_value = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    with (
        patch(
            "observability.usage.TokenUsageCallback",
            return_value=fake_tracker,
        ),
        patch(
            "observability.usage.get_model_pricing",
            return_value={"input": 0.15, "output": 0.60},
        ),
        patch("observability.usage.save_model_usage") as save_usage,
    ):
        result = tracked_invoke(
            runnable,
            {"input": "test"},
            run_id="test-run",
            node_name="technical_researcher",
        )

    assert result == {"output": "research result"}
    save_usage.assert_not_called()


def test_tracked_invoke_records_retries_under_same_run_id():
    runnable = MagicMock()
    runnable.invoke.return_value = {"output": "retry result"}

    fake_tracker = MagicMock()
    fake_tracker.get_summary.return_value = {
        "prompt_tokens": 120,
        "completion_tokens": 40,
    }

    with (
        patch(
            "observability.usage.TokenUsageCallback",
            return_value=fake_tracker,
        ),
        patch(
            "observability.usage.get_model_pricing",
            return_value={"input": 0.15, "output": 0.60},
        ),
        patch("observability.usage.save_model_usage") as save_usage,
    ):
        tracked_invoke(
            runnable,
            {"input": "initial"},
            run_id="run-retry-01",
            node_name="technical_researcher",
        )
        tracked_invoke(
            runnable,
            {"input": "retry"},
            run_id="run-retry-01",
            node_name="technical_researcher",
        )

    assert save_usage.call_count == 2
    for call in save_usage.call_args_list:
        assert call.kwargs["run_id"] == "run-retry-01"