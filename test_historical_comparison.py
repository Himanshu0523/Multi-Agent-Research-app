from memory.claim_history import fingerprint_claim

from memory.historical_comparison import (
    compare_historical_claims,
)


def test_unchanged_claim():

    claim = "Revenue increased by 10%"

    result = compare_historical_claims(
        current_claims=[
            {
                "claim": claim,
                "status": "supported",
            }
        ],
        previous_claims=[
            {
                "claim_fingerprint": fingerprint_claim(
                    claim
                ),
                "claim_text": claim,
                "verification_status": "supported",
            }
        ],
    )

    assert len(result["unchanged"]) == 1


def test_new_claim():

    result = compare_historical_claims(
        current_claims=[
            {
                "claim": "Revenue increased by 20%",
                "status": "supported",
            }
        ],
        previous_claims=[],
    )

    assert len(result["new_claims"]) == 1


def test_status_changed():

    claim = "Revenue increased by 10%"

    result = compare_historical_claims(
        current_claims=[
            {
                "claim": claim,
                "status": "contradicted",
            }
        ],
        previous_claims=[
            {
                "claim_fingerprint": fingerprint_claim(
                    claim
                ),
                "claim_text": claim,
                "verification_status": "supported",
            }
        ],
    )

    assert len(result["status_changed"]) == 1
    assert (
        result["status_changed"][0]["current_status"]
        == "contradicted"
    )