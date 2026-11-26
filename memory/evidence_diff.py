import hashlib
import re


def normalize_claim(claim: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        claim.lower(),
    ).strip()


def claim_fingerprint(claim: str) -> str:
    normalized = normalize_claim(claim)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def compare_claims(
    old_claims: list[dict],
    new_claims: list[dict],
) -> dict:

    old_map = {
        claim_fingerprint(item["claim"]): item
        for item in old_claims
    }

    new_map = {
        claim_fingerprint(item["claim"]): item
        for item in new_claims
    }

    old_keys = set(old_map)
    new_keys = set(new_map)

    return {
        "unchanged": [
            new_map[key]
            for key in old_keys & new_keys
        ],
        "added": [
            new_map[key]
            for key in new_keys - old_keys
        ],
        "removed": [
            old_map[key]
            for key in old_keys - new_keys
        ],
    }