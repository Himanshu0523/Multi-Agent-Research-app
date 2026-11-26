from memory.repository import get_latest_source
from memory.refresh import inspect_source_changes


def build_incremental_plan(
    discovered_sources: list[dict],
) -> dict:

    unchanged_sources = []
    changed_sources = []
    new_sources = []
    failed_sources = []

    for source in discovered_sources:

        try:
            old_source = get_latest_source(
                source["url"]
            )

            result = inspect_source_changes(
                old_source=old_source,
                new_source=source,
            )

            if result["status"] == "unchanged":
                unchanged_sources.append({
                    **source,
                    "change_status": "unchanged",
                })

            elif result["status"] == "changed":
                changed_sources.append({
                    **source,
                    "change_status": "changed",
                })

            else:
                new_sources.append({
                    **source,
                    "change_status": "new",
                })

        except Exception as exc:
            failed_sources.append({
                "url": source.get("url"),
                "error": str(exc),
            })

    return {
        "unchanged_sources": unchanged_sources,
        "changed_sources": changed_sources,
        "new_sources": new_sources,
        "failed_sources": failed_sources,
        "requires_research": bool(
            changed_sources or new_sources
        ),
    }