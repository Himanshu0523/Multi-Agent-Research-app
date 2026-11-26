
from contradiction import (
    generate_candidate_pairs,
    build_conflict_report,
)


def test_candidate_pair_generation():
    claims = [
        {
            "claim_id": "c1",
            "entity": "Model A",
            "attribute": "accuracy",
            "normalization_status": "success",
        },
        {
            "claim_id": "c2",
            "entity": "Model A",
            "attribute": "accuracy",
            "normalization_status": "success",
        },
        {
            "claim_id": "c3",
            "entity": "Model B",
            "attribute": "accuracy",
            "normalization_status": "success",
        },
    ]

    pairs = generate_candidate_pairs(claims)

    assert len(pairs) == 1


def test_conflict_report():
    comparisons = [
        {
            "claim_a_id": "c1",
            "claim_b_id": "c2",
            "claim_a": "Model A achieved 90% accuracy.",
            "claim_b": "Model A achieved 70% accuracy.",
            "claim_a_sources": ["src1"],
            "claim_b_sources": ["src2"],
            "relationship": "contradiction",
            "confidence": 0.90,
            "explanation": "The values differ.",
            "contextual_factors": [],
            "resolution": "Check evaluation methodology.",
        }
    ]

    report = build_conflict_report(comparisons)

    assert report["summary"]["contradictions"] == 1
    assert report["summary"]["unresolved"] == 1
    assert report["resolution_status"] == "requires_review"
