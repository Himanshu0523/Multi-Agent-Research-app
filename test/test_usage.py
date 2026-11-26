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
    save_usage.assert_called_once()