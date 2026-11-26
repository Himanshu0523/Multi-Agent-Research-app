import hashlib
import re
import uuid

from sqlalchemy import select

from memory.database import SessionLocal
from memory.models import (
    ResearchRun,
    ResearchSource,
    ResearchEvidence,
)

from provenance import normalize_url, generate_source_id


def fingerprint_topic(topic: str) -> str:
    normalized = re.sub(
        r"\s+", " ", topic.lower()
    ).strip()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def create_run(topic: str) -> str:
    run_id = str(uuid.uuid4())

    with SessionLocal() as db:
        run = ResearchRun(
            id=run_id,
            topic=topic,
            topic_fingerprint=fingerprint_topic(topic),
            status="running",
        )

        db.add(run)
        db.commit()

    return run_id


def get_previous_runs(topic: str, limit: int = 5):
    fingerprint = fingerprint_topic(topic)

    with SessionLocal() as db:
        statement = (
            select(ResearchRun)
            .where(
                ResearchRun.topic_fingerprint == fingerprint,
                ResearchRun.status == "completed",
            )
            .order_by(ResearchRun.started_at.desc())
            .limit(limit)
        )

        return [
            {
                "id": run.id,
                "topic": run.topic,
                "status": run.status,
                "started_at": run.started_at.isoformat(),
            }
            for run in db.scalars(statement).all()
        ]


def save_source(source: dict):
    url = normalize_url(source["url"])
    source_id = generate_source_id(url)
    content_hash = source["content_hash"]

    with SessionLocal() as db:
        existing = db.scalar(
            select(ResearchSource).where(
                ResearchSource.source_id == source_id,
                ResearchSource.content_hash == content_hash,
            )
        )

        if existing:
            return existing.id, False

        version_id = hashlib.sha256(
            f"{source_id}:{content_hash}".encode()
        ).hexdigest()

        record = ResearchSource(
            id=version_id,
            source_id=source_id,
            url=url,
            title=source.get("title", ""),
            content_hash=content_hash,
            content=source["content"],
        )

        db.add(record)
        db.commit()

        return version_id, True


def get_latest_source(url: str):
    source_id = generate_source_id(
        normalize_url(url)
    )

    with SessionLocal() as db:
        statement = (
            select(ResearchSource)
            .where(ResearchSource.source_id == source_id)
            .order_by(ResearchSource.retrieved_at.desc())
        )

        record = db.scalars(statement).first()

        if not record:
            return None

        return {
            "source_id": record.source_id,
            "url": record.url,
            "title": record.title,
            "content_hash": record.content_hash,
            "content": record.content,
        }


def get_latest_source_record(url: str):

    source_id = generate_source_id(
        normalize_url(url)
    )

    with SessionLocal() as db:

        statement = (
            select(ResearchSource)
            .where(
                ResearchSource.source_id == source_id
            )
            .order_by(
                ResearchSource.retrieved_at.desc()
            )
        )

        record = db.scalars(statement).first()

        if not record:
            return None

        return {
            "source_id": record.source_id,
            "url": record.url,
            "title": record.title,
            "content_hash": record.content_hash,
            "content": record.content,
            "retrieved_at": record.retrieved_at,
        }


def save_evidence(
    run_id: str,
    evidence: dict,
):
    evidence_id = evidence.get("evidence_id")

    if not evidence_id:
        raise ValueError(
            "Evidence must have a stable evidence_id"
        )

    with SessionLocal() as db:
        existing = db.get(
            ResearchEvidence,
            evidence_id,
        )

        if existing:
            return evidence_id

        record = ResearchEvidence(
            id=evidence_id,
            run_id=run_id,
            source_id=evidence["source_id"],
            claim=evidence["claim"],
            quote=evidence.get("quote", ""),
            status=evidence.get("status", "unverified"),
        )

        db.add(record)
        db.commit()

    return evidence_id


def complete_run(run_id: str, status: str = "completed"):
    from datetime import datetime, timezone

    with SessionLocal() as db:
        run = db.get(ResearchRun, run_id)

        if not run:
            raise ValueError("Research run not found")

        run.status = status
        run.completed_at = datetime.now(timezone.utc)

        db.commit()