from datetime import datetime, timezone


SOURCE_REFRESH_HOURS = {
    "news": 6,
    "official_documentation": 72,
    "research_paper": 168,
    "government": 72,
    "general": 48,
}


def should_refresh_source(
    retrieved_at: datetime,
    source_type: str = "general",
) -> bool:

    refresh_hours = SOURCE_REFRESH_HOURS.get(
        source_type,
        SOURCE_REFRESH_HOURS["general"],
    )

    now = datetime.now(timezone.utc)

    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.replace(
            tzinfo=timezone.utc
        )

    age_hours = (
        now - retrieved_at
    ).total_seconds() / 3600

    return age_hours >= refresh_hours