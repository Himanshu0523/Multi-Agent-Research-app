from sqlalchemy import select, func

from memory.database import SessionLocal
from observability.models import (
    ExecutionEvent,
    ModelUsage,
    AuditRecord,
)


def get_run_analytics(run_id: str):

    with SessionLocal() as db:

        events = db.scalars(
            select(ExecutionEvent).where(
                ExecutionEvent.run_id == run_id
            )
        ).all()

        usage = db.scalars(
            select(ModelUsage).where(
                ModelUsage.run_id == run_id
            )
        ).all()

        audit_count = db.scalar(
            select(func.count(AuditRecord.id)).where(
                AuditRecord.run_id == run_id
            )
        ) or 0

        durations = [
            event.duration_ms
            for event in events
            if event.duration_ms is not None
        ]

        return {
            "run_id": run_id,
            "total_node_executions": len(events),
            "successful_nodes": sum(
                event.status == "success"
                for event in events
            ),
            "failed_nodes": sum(
                event.status == "failed"
                for event in events
            ),
            "total_node_duration_ms": sum(durations),
            "average_node_duration_ms": (
                sum(durations) / len(durations)
                if durations else 0
            ),
            "total_prompt_tokens": sum(
                item.prompt_tokens for item in usage
            ),
            "total_completion_tokens": sum(
                item.completion_tokens for item in usage
            ),
            "total_tokens": sum(
                item.total_tokens for item in usage
            ),
            "estimated_cost": sum(
                item.estimated_cost for item in usage
            ),
            "audit_events": audit_count,
        }