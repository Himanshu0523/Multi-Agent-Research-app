from provenance import create_source_record

from memory.database import init_database
from memory.repository import (
    create_run,
    get_previous_runs,
    save_source,
    get_latest_source,
    complete_run,
)


def test_research_memory():

    init_database()

    topic = "AI research agent architecture"

    run_id = create_run(topic)

    source = create_source_record(
        url="https://example.com/research",
        title="Research Example",
        content="Research content for testing.",
    )

    _, is_new = save_source(source)

    assert is_new is True

    cached = get_latest_source(source["url"])

    assert cached is not None
    assert cached["content_hash"] == source["content_hash"]

    complete_run(run_id)

    previous = get_previous_runs(topic)

    assert any(
        item["id"] == run_id
        for item in previous
    )