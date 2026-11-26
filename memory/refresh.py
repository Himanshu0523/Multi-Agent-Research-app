import hashlib
from datetime import datetime, timezone

from provenance import (
    normalize_url,
    generate_content_hash,
)


def inspect_source_changes(
    old_source: dict | None,
    new_source: dict,
) -> dict:

    new_url = normalize_url(new_source["url"])

    new_hash = generate_content_hash(
        new_source["content"]
    )

    if old_source is None:
        return {
            "url": new_url,
            "status": "new",
            "old_hash": None,
            "new_hash": new_hash,
            "content_changed": True,
            "checked_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    old_hash = old_source["content_hash"]

    if old_hash == new_hash:
        status = "unchanged"
    else:
        status = "changed"

    return {
        "url": new_url,
        "status": status,
        "old_hash": old_hash,
        "new_hash": new_hash,
        "content_changed": old_hash != new_hash,
        "checked_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }