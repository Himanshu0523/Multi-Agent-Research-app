from memory.claim_history import fingerprint_claim


def compare_historical_claims(
    current_claims: list[dict],
    previous_claims: list[dict],
) -> dict:

    previous_map = {
        item["claim_fingerprint"]: item
        for item in previous_claims
    }

    current_map = {
        fingerprint_claim(item["claim"]): item
        for item in current_claims
    }

    current_keys = set(current_map)
    previous_keys = set(previous_map)

    unchanged = []
    new_claims = []
    status_changed = []
    no_longer_present = []

    for key in current_keys & previous_keys:

        current = current_map[key]
        previous = previous_map[key]

        old_status = previous["verification_status"]
        new_status = current.get("status", "unverified")

        if old_status == new_status:
            unchanged.append({
                "claim": current["claim"],
                "status": new_status,
            })
        else:
            status_changed.append({
                "claim": current["claim"],
                "previous_status": old_status,
                "current_status": new_status,
            })

    for key in current_keys - previous_keys:
        new_claims.append(current_map[key])

    for key in previous_keys - current_keys:
        no_longer_present.append(previous_map[key])

    return {
        "unchanged": unchanged,
        "new_claims": new_claims,
        "status_changed": status_changed,
        "no_longer_present": no_longer_present,
    }