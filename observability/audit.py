import uuid

from memory.database import SessionLocal
from observability.models import AuditRecord


def record_audit(
    run_id: str,
    action: str,
    description: str,
    actor: str = "system",
    metadata: dict | None = None,
):

    with SessionLocal() as db:

        record = AuditRecord(
            id=str(uuid.uuid4()),
            run_id=run_id,
            action=action,
            actor=actor,
            description=description,
            metadata_json=metadata or {},
        )

        db.add(record)
        db.commit()

        return record.id