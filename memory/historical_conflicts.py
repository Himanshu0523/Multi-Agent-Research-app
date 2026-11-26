from memory.historical_comparison import (
    compare_historical_claims,
)


def analyze_historical_changes(
    current_claims: list[dict],
    previous_claims: list[dict],
) -> dict:

    comparison = compare_historical_claims(
        current_claims,
        previous_claims,
    )

    conflicts = []

    for item in comparison["status_changed"]:

        old_status = item["previous_status"]
        new_status = item["current_status"]

        if (
            old_status == "supported"
            and new_status in {
                "contradicted",
                "insufficient",
            }
        ):
            conflicts.append({
                "claim": item["claim"],
                "previous_status": old_status,
                "current_status": new_status,
                "reason": (
                    "Previously supported claim is no longer "
                    "supported by the current verification."
                ),
            })

    return {
        "comparison": comparison,
        "historical_conflicts": conflicts,
        "requires_attention": bool(conflicts),
    }