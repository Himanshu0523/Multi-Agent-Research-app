from memory.database import init_database
from memory.repository import create_run, complete_run

from observability.analytics import get_run_analytics
from observability.audit import record_audit

from graph import build_research_graph


def run_research_pipeline(
    topic: str,
    max_retries: int = 2,
) -> dict:
    init_database()
    run_id = create_run(topic)
    research_graph = build_research_graph()

    print("\n" + "=" * 65)
    print(f"MULTI-AGENT RESEARCH SYSTEM (Run ID: {run_id})")
    print("=" * 65)

    record_audit(
        run_id=run_id,
        action="research_started",
        description="Research workflow started",
        metadata={"topic": topic, "max_retries": max_retries},
    )

    try:
        result = research_graph.invoke({
            "topic": topic,
            "run_id": run_id,
            "previous_runs": [],
            "reused_sources": [],
            "memory_enabled": True,
            "incremental_plan": {},
            "unchanged_sources": [],
            "changed_sources": [],
            "new_sources": [],
            "failed_sources": [],
            "previous_claims": [],
            "claim_snapshots": [],
            "historical_comparison": {},
            "historical_conflicts": [],
            "historical_review_required": False,
            "research_objective": "",
            "research_questions": [],
            "gap_assessment": {},
            "gap_coverage": 0.0,
            "retry_count": 0,
            "max_retries": max_retries,
            "sources": [],
            "source_intelligence": [],
            "source_relationships": [],
            "evidence_records": [],
            "missing_evidence": [],
            "research_feedback": "",
            "normalized_claims": [],
            "claim_comparisons": [],
            "conflict_report": {},
            "unresolved_conflicts": [],
            "research_gaps": [],
            "research_quality": {},
            "quality_status": "needs_review",
        })

        complete_run(
            run_id,
            status="completed",
        )

        record_audit(
            run_id=run_id,
            action="report_generated",
            description="Final research report generated",
            metadata={
                "quality_status": result.get("quality_status"),
                "verification_status": result.get("verification_status"),
            },
        )

    except Exception as error:
        complete_run(
            run_id,
            status="failed",
        )

        record_audit(
            run_id=run_id,
            action="research_failed",
            description=f"Research workflow failed: {type(error).__name__}",
        )
        raise

    analytics = get_run_analytics(run_id)

    print("\n" + "=" * 65)
    print("FINAL RESEARCH REPORT")
    print("=" * 65)
    print(result.get("final_report", "No report generated."))

    print("\n" + "=" * 65)
    print("RUN ANALYTICS")
    print("=" * 65)
    print(analytics)

    return {
        "run_id": run_id,
        "topic": topic,
        "technical_research": result.get("search_a", ""),
        "industry_research": result.get("search_b", ""),
        "combined_research": result.get("combined_search", ""),
        "reader_notes": result.get("reader_notes", ""),
        "report": result.get("final_report", ""),
        "analytics": analytics,
    }


if __name__ == "__main__":
    result = run_research_pipeline(
        "Latest breakthroughs in Solid State Batteries 2026"
    )