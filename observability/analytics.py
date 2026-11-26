from collections import defaultdict
from sqlalchemy import select, func

from memory.database import SessionLocal
from observability.models import (
    ExecutionEvent,
    ModelUsage,
    AuditRecord,
)


def get_run_analytics(run_id: str) -> dict:
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

        audit_count = (
            db.scalar(
                select(func.count(AuditRecord.id)).where(
                    AuditRecord.run_id == run_id
                )
            )
            or 0
        )

        durations = [
            event.duration_ms
            for event in events
            if event.duration_ms is not None
        ]

        total_duration = round(sum(durations), 2)
        avg_duration = (
            round(total_duration / len(durations), 2)
            if durations
            else 0.0
        )

        total_prompt = sum(item.prompt_tokens or 0 for item in usage)
        total_completion = sum(item.completion_tokens or 0 for item in usage)
        total_tokens = sum(
            item.total_tokens
            if item.total_tokens is not None
            else ((item.prompt_tokens or 0) + (item.completion_tokens or 0))
            for item in usage
        )
        total_cost = round(
            sum(item.estimated_cost or 0.0 for item in usage),
            6,
        )

        node_usage: dict[str, dict] = defaultdict(
            lambda: {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            }
        )

        for item in usage:
            entry = node_usage[item.node_name]
            p_tok = item.prompt_tokens or 0
            c_tok = item.completion_tokens or 0
            t_tok = (
                item.total_tokens
                if item.total_tokens is not None
                else (p_tok + c_tok)
            )
            entry["calls"] += 1
            entry["prompt_tokens"] += p_tok
            entry["completion_tokens"] += c_tok
            entry["total_tokens"] += t_tok
            entry["estimated_cost"] = round(
                entry["estimated_cost"] + (item.estimated_cost or 0.0),
                6,
            )

        return {
            "run_id": run_id,
            "total_node_executions": len(events),
            "successful_nodes": sum(
                event.status == "success" for event in events
            ),
            "failed_nodes": sum(
                event.status == "failed" for event in events
            ),
            "total_node_duration_ms": total_duration,
            "average_node_duration_ms": avg_duration,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "estimated_cost": total_cost,
            "audit_events": audit_count,
            "node_usage": dict(node_usage),
        }