from sqlalchemy import select

from memory.database import SessionLocal
from memory.models import ClaimSnapshot


def save_claim_snapshots(claims: list[dict]):

    with SessionLocal() as db:

        records = [
            ClaimSnapshot(**claim)
            for claim in claims
        ]

        db.add_all(records)
        db.commit()


def get_previous_claims(
    claim_fingerprints: list[str],
):

    if not claim_fingerprints:
        return []

    with SessionLocal() as db:

        statement = (
            select(ClaimSnapshot)
            .where(
                ClaimSnapshot.claim_fingerprint.in_(
                    claim_fingerprints
                )
            )
            .order_by(ClaimSnapshot.id)
        )

        records = db.scalars(statement).all()

        return [
            {
                "id": item.id,
                "run_id": item.run_id,
                "claim_fingerprint": item.claim_fingerprint,
                "claim_text": item.claim_text,
                "verification_status": item.verification_status,
                "evidence_summary": item.evidence_summary,
            }
            for item in records
        ]


def get_claim_history(fingerprint: str):

    with SessionLocal() as db:

        statement = (
            select(ClaimSnapshot)
            .where(
                ClaimSnapshot.claim_fingerprint
                == fingerprint
            )
            .order_by(ClaimSnapshot.id)
        )

        records = db.scalars(statement).all()

        return [
            {
                "run_id": item.run_id,
                "claim": item.claim_text,
                "status": item.verification_status,
                "evidence_summary": item.evidence_summary,
            }
            for item in records
        ]