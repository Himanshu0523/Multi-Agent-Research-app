
from quality import evaluate_research_quality


def test_quality_evaluator():

    sources = [
        {
            "source_id": "src_001",
            "url": "https://example.com/research",
            "published_at": "2026-10-01T00:00:00+00:00",
        },
        {
            "source_id": "src_002",
            "url": "https://university.edu/paper",
            "published_at": "2026-09-01T00:00:00+00:00",
        },
    ]

    source_intelligence = [
        {
            "source_id": "src_001",
            "source_type": "official",
        },
        {
            "source_id": "src_002",
            "source_type": "primary",
        },
    ]

    evidence_records = [
        {
            "claim": "Example research finding",
            "status": "supported",
            "source_ids": ["src_001"],
        },
        {
            "claim": "Another research finding",
            "status": "supported",
            "source_ids": ["src_002"],
        },
    ]

    result = evaluate_research_quality(
        sources=sources,
        source_intelligence=source_intelligence,
        evidence_records=evidence_records,
    )

    assert 0 <= result["overall_score"] <= 1
    assert result["metrics"]["citation_integrity"]["valid"] == 2
    assert result["metrics"]["evidence_coverage"]["supported"] == 2

    print(result)


if __name__ == "__main__":
    test_quality_evaluator()

