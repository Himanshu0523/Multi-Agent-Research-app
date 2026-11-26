import uuid
from datetime import datetime, timezone
from time import perf_counter

from memory.database import SessionLocal
from observability.models import ExecutionEvent


def utc_now():
    return datetime.now(timezone.utc)


def start_event(
    run_id: str,
    node_name: str,
) -> tuple[str, float]:

    event_id = str(uuid.uuid4())
    started = perf_counter()

    with SessionLocal() as db:
        db.add(
            ExecutionEvent(
                id=event_id,
                run_id=run_id,
                node_name=node_name,
                event_type="node_execution",
                status="running",
                started_at=utc_now(),
            )
        )
        db.commit()

    return event_id, started


def finish_event(
    event_id: str,
    started: float,
    status: str,
    metadata: dict | None = None,
):
    duration_ms = (
        perf_counter() - started
    ) * 1000

    with SessionLocal() as db:
        event = db.get(ExecutionEvent, event_id)

        if event is None:
            raise ValueError("Execution event not found")

        event.status = status
        event.completed_at = utc_now()
        event.duration_ms = duration_ms
        event.metadata_json = metadata or {}

        db.commit()

    return duration_ms


def track_node(node_name: str, node_function):

    def wrapped_node(state):

        run_id = state.get("run_id")

        if not run_id:
            raise ValueError(
                "run_id is required for observability"
            )

        event_id, started = start_event(
            run_id,
            node_name,
        )

        try:
            result = node_function(state)

            finish_event(
                event_id,
                started,
                status="success",
                metadata={
                    "result_keys": list(result.keys())
                    if isinstance(result, dict)
                    else [],
                },
            )

            return result

        except Exception as exc:
            finish_event(
                event_id,
                started,
                status="failed",
                metadata={
                    "error_type": type(exc).__name__,
                },
            )

            raise

    return wrapped_node