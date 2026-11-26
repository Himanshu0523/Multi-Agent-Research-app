from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Text,
    DateTime,
    ForeignKey,
    Integer,
    UniqueConstraint,
    Index,
)
from sqlalchemy import Boolean
from sqlalchemy.orm import Mapped, mapped_column
from memory.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class ResearchRun(Base):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True
    )

    topic: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    topic_fingerprint: Mapped[str] = mapped_column(
        String(64), index=True
    )

    status: Mapped[str] = mapped_column(
        String(30), default="running"
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ResearchSource(Base):
    __tablename__ = "research_sources"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True
    )

    source_id: Mapped[str] = mapped_column(
        String(64), index=True
    )

    url: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    title: Mapped[str] = mapped_column(
        Text, default=""
    )

    content_hash: Mapped[str] = mapped_column(
        String(64), index=True
    )

    content: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "content_hash",
            name="uq_source_content_version",
        ),
    )


class ResearchEvidence(Base):
    __tablename__ = "research_evidence"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id"),
        index=True,
    )

    source_id: Mapped[str] = mapped_column(
        String(64), index=True
    )

    claim: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    quote: Mapped[str] = mapped_column(
        Text, default=""
    )

    status: Mapped[str] = mapped_column(
        String(30), default="unverified"
    )
    
class ClaimSnapshot(Base):
    __tablename__ = "claim_snapshots"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id"),
        index=True,
    )

    claim_fingerprint: Mapped[str] = mapped_column(
        String(64),
        index=True,
    )

    claim_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    verification_status: Mapped[str] = mapped_column(
        String(30),
        default="unverified",
    )

    evidence_summary: Mapped[str] = mapped_column(
        Text,
        default="",
    )

    is_current: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )