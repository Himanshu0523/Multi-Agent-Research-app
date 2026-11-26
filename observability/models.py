from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Text,
    DateTime,
    Integer,
    Float,
    ForeignKey,
    JSON,
    Index,
)

from sqlalchemy.orm import Mapped, mapped_column

from memory.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class ExecutionEvent(Base):
    __tablename__ = "execution_events"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id"),
        index=True,
    )

    node_name: Mapped[str] = mapped_column(
        String(100),
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
    )

    status: Mapped[str] = mapped_column(
        String(30),
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    duration_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )


class ModelUsage(Base):
    __tablename__ = "model_usage"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id"),
        index=True,
    )

    node_name: Mapped[str] = mapped_column(
        String(100),
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
    )

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    estimated_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )


class AuditRecord(Base):
    __tablename__ = "audit_records"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id"),
        index=True,
    )

    action: Mapped[str] = mapped_column(
        String(100),
    )

    actor: Mapped[str] = mapped_column(
        String(100),
        default="system",
    )

    description: Mapped[str] = mapped_column(
        Text,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )