from memory.refresh import inspect_source_changes
from memory.evidence_diff import compare_claims


def test_new_source():

    result = inspect_source_changes(
        old_source=None,
        new_source={
            "url": "https://example.com",
            "content": "New research content",
        },
    )

    assert result["status"] == "new"


def test_unchanged_source():

    source = {
        "url": "https://example.com",
        "content": "Same research content",
    }

    first = inspect_source_changes(
        old_source=None,
        new_source=source,
    )

    old_source = {
        "content_hash": first["new_hash"]
    }

    result = inspect_source_changes(
        old_source=old_source,
        new_source=source,
    )

    assert result["status"] == "unchanged"


def test_changed_source():

    result = inspect_source_changes(
        old_source={
            "content_hash": "old_hash"
        },
        new_source={
            "url": "https://example.com",
            "content": "Updated research content",
        },
    )

    assert result["status"] == "changed"


def test_claim_comparison():

    result = compare_claims(
        old_claims=[
            {"claim": "Python was released in 1991"},
            {"claim": "Python is dynamically typed"},
        ],
        new_claims=[
            {"claim": "Python was released in 1991"},
            {"claim": "Python supports type hints"},
        ],
    )

    assert len(result["unchanged"]) == 1
    assert len(result["added"]) == 1
    assert len(result["removed"]) == 1