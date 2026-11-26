import hashlib
import re
import uuid


def normalize_claim(claim: str) -> str:
    claim = claim.lower().strip()

    claim = re.sub(
        r"\s+",
        " ",
        claim,
    )

    return claim


def fingerprint_claim(claim: str) -> str:

    normalized = normalize_claim(claim)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def create_claim_snapshot(
    run_id: str,
    claim: dict,
) -> dict:

    claim_text = claim["claim"]

    return {
        "id": str(uuid.uuid4()),
        "run_id": run_id,
        "claim_fingerprint": fingerprint_claim(
            claim_text
        ),
        "claim_text": claim_text,
        "verification_status": claim.get(
            "status",
            "unverified",
        ),
        "evidence_summary": claim.get(
            "evidence_summary",
            "",
        ),
        "is_current": True,
    }