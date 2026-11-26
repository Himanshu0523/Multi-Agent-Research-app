from langchain_core.callbacks import BaseCallbackHandler


class TokenUsageCallback(BaseCallbackHandler):

    def __init__(self):
        self.records = []

    def on_llm_end(
        self,
        response,
        **kwargs,
    ):
        for generation_list in response.generations:

            for generation in generation_list:

                message = getattr(
                    generation,
                    "message",
                    None,
                )

                usage = getattr(
                    message,
                    "usage_metadata",
                    None,
                ) or {}

                if usage:
                    self.records.append({
                        "prompt_tokens": usage.get(
                            "input_tokens", 0
                        ),
                        "completion_tokens": usage.get(
                            "output_tokens", 0
                        ),
                        "total_tokens": usage.get(
                            "total_tokens", 0
                        ),
                    })

        if not self.records and response.llm_output:

            usage = response.llm_output.get(
                "token_usage",
                {},
            )

            if usage:
                self.records.append({
                    "prompt_tokens": usage.get(
                        "prompt_tokens", 0
                    ),
                    "completion_tokens": usage.get(
                        "completion_tokens", 0
                    ),
                    "total_tokens": usage.get(
                        "total_tokens", 0
                    ),
                })

    def get_summary(self):

        return {
            "prompt_tokens": sum(
                item["prompt_tokens"]
                for item in self.records
            ),
            "completion_tokens": sum(
                item["completion_tokens"]
                for item in self.records
            ),
            "total_tokens": sum(
                item["total_tokens"]
                for item in self.records
            ),
        }