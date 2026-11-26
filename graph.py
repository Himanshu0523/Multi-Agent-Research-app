import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from observability.tracker import track_node
from observability.callbacks import TokenUsageCallback
from observability.metrics import save_model_usage
from observability.usage import tracked_invoke
from agents import build_search_agent, build_writer_chain
from tools import fetch_document
from evidence import (
    extract_research_claims,
    build_evidence_records,
)
from source_intelligence import (
    analyze_source,
    classify_source_with_llm,
)
from source_independence import analyze_source_independence
from contradiction import (
    detect_contradictions,
    build_conflict_report,
)
from memory.incremental import build_incremental_plan
from memory.evidence_diff import compare_claims
from quality import evaluate_research_quality
from gap_planner import (
    create_research_plan,
    assess_question_coverage,
    build_gap_feedback,
)
from memory.database import init_database
from memory.repository import (
    create_run,
    get_previous_runs,
    save_source,
    save_evidence,
    complete_run,
)
from memory.claim_history import (
    create_claim_snapshot,
    fingerprint_claim,
)
from memory.history_repository import (
    save_claim_snapshots,
    get_previous_claims,
)
from memory.historical_conflicts import (
    analyze_historical_changes,
)


# ============================================================
# MODEL PRICING RATES (USD per 1M tokens for gpt-4o-mini)
# ============================================================
INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60


# ============================================================
# SHARED RESEARCH STATE
# ============================================================
class ResearchState(TypedDict, total=False):
    topic: str

    # Research planning (V3.5)
    research_objective: str
    research_questions: list[dict]
    gap_assessment: dict
    gap_coverage: float

    # Research outputs
    search_a: str
    search_b: str
    combined_search: str
    reader_notes: str

    # Source data
    sources: list[dict]
    source_intelligence: list[dict]
    source_relationships: list[dict]

    # Evidence & contradiction data
    evidence_records: list[dict]
    verification_result: dict
    verification_status: str
    missing_evidence: list[str]
    normalized_claims: list[dict]
    claim_comparisons: list[dict]
    conflict_report: dict
    unresolved_conflicts: list[str]

    # Quality & retry management
    research_gaps: list[str]
    research_quality: dict
    quality_status: str
    retry_count: int
    max_retries: int
    research_feedback: str

    # Persistent memory (V3.6)
    run_id: str
    previous_runs: list
    reused_sources: list
    memory_enabled: bool

    # Incremental research (V3.7)
    incremental_plan: dict
    unchanged_sources: list
    changed_sources: list
    new_sources: list
    failed_sources: list

    # Historical claim tracking (V3.8)
    previous_claims: list
    claim_snapshots: list
    historical_comparison: dict
    historical_conflicts: list
    historical_review_required: bool

    # Final output
    final_report: str


# ============================================================
# HELPERS
# ============================================================
def extract_urls(text: str) -> list[str]:
    """Extract unique HTTP/HTTPS URLs from research text."""
    matches = re.findall(r"""https?://[^\s<>"']+""", text or "")
    urls = []
    for url in matches:
        url = url.rstrip(".,;:!?)\]}")
        if url not in urls:
            urls.append(url)
    return urls


def get_successful_sources(sources: list[dict]) -> list[dict]:
    """Return only successfully fetched source documents."""
    return [
        source
        for source in sources
        if source.get("status") == "fetched"
    ]


def format_questions_for_prompt(research_questions: list[dict]) -> str:
    """Format planned research questions for researcher prompts."""
    if not research_questions:
        return "No specific planned questions provided."
    return "\n".join(
        f"{item['question_id']}: {item['question']} [Priority: {item['importance']}]"
        for item in research_questions
    )


# ============================================================
# NODE -1: MEMORY INITIALIZATION (V3.6)
# ============================================================
def memory_initialization_node(state: ResearchState):
    print("\n[MEMORY] Initializing research run and checking history")
    init_database()

    topic = state["topic"]
    run_id = state.get("run_id") or create_run(topic)
    previous_runs = get_previous_runs(topic)

    print(f"[MEMORY] Run ID: {run_id} | Previous runs found: {len(previous_runs)}")

    return {
        "run_id": run_id,
        "previous_runs": previous_runs,
        "reused_sources": [],
        "new_sources": [],
        "memory_enabled": True,
    }


# ============================================================
# NODE 0: RESEARCH PLANNER (V3.5 + TOKEN TRACKING)
# ============================================================
def research_planner_node(state: ResearchState):
    print("\n[PLANNER] Creating research plan")

    usage_tracker = TokenUsageCallback()

    plan = create_research_plan(
        topic=state["topic"],
        max_questions=8,
        config={"callbacks": [usage_tracker]},
    )

    usage = usage_tracker.get_summary()

    save_model_usage(
        run_id=state["run_id"],
        node_name="research_planner",
        model_name="gpt-4o-mini",
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        input_price_per_million=INPUT_PRICE_PER_MILLION,
        output_price_per_million=OUTPUT_PRICE_PER_MILLION,
    )

    print(f"[PLANNER] Created {len(plan['research_questions'])} questions")
    return plan


# ============================================================
# NODE 1: TECHNICAL RESEARCHER
# ============================================================
def technical_researcher(state: ResearchState):
    print("\n[AGENT A] Technical research started")

    agent = build_search_agent()
    feedback = state.get("research_feedback", "")
    questions_text = format_questions_for_prompt(
        state.get("research_questions", [])
    )

    prompt = f"""
Research topic: {state['topic']}

RESEARCH QUESTIONS:
{questions_text}

Research focus:
- Scientific breakthroughs
- Engineering developments
- Technical evidence
- Research papers
- Benchmarks and experimental results

Additional research requirements:
{feedback}

Find relevant sources, dates, evidence and URLs.
Distinguish peer-reviewed research, preprints, official documentation,
company announcements and speculation.

Return findings with source titles and original URLs.
"""

    result = tracked_invoke(
        agent,
        {"input": prompt},
        run_id=state["run_id"],
        node_name="technical_researcher",
    )

    return {"search_a": result["output"]}


# ============================================================
# NODE 2: INDUSTRY RESEARCHER
# ============================================================
def industry_researcher(state: ResearchState):
    print("\n[AGENT B] Industry research started")

    agent = build_search_agent()
    feedback = state.get("research_feedback", "")
    questions_text = format_questions_for_prompt(
        state.get("research_questions", [])
    )

    prompt = f"""
Research topic: {state['topic']}

RESEARCH QUESTIONS:
{questions_text}

Research focus:
- Industry developments
- Commercialization
- Market trends
- Companies and products
- Real-world implementations
- Adoption and business impact

Additional research requirements:
{feedback}

Find relevant sources, publication dates, commercial evidence
and original URLs.

Distinguish verified product releases from announcements,
marketing claims and speculation.
"""

    result = tracked_invoke(
        agent,
        {"input": prompt},
        run_id=state["run_id"],
        node_name="industry_researcher",
    )

    return {"search_b": result["output"]}


# ============================================================
# NODE 3: AGGREGATOR
# ============================================================
def aggregate_research(state: ResearchState):
    print("\n[AGGREGATOR] Combining research findings")

    technical = state.get("search_a", "")
    industry = state.get("search_b", "")

    combined = f"""
TECHNICAL RESEARCH
==================

{technical}


INDUSTRY RESEARCH
=================

{industry}
"""
    return {"combined_search": combined}


# ============================================================
# NODE 4: READER
# ============================================================
def reader_node(state: ResearchState):
    print("\n[READER] Collecting source documents")

    urls = extract_urls(state.get("combined_search", ""))
    existing_sources = state.get("sources", [])
    existing_urls = {
        source.get("url")
        for source in existing_sources
        if source.get("url")
    }

    new_urls = [url for url in urls if url not in existing_urls][:5]
    new_sources = []

    for url in new_urls:
        print(f"[FETCHING] {url}")
        try:
            document = fetch_document(url)
            if isinstance(document, dict):
                new_sources.append(document)
            else:
                print(f"[WARNING] Unexpected document format: {url}")
        except Exception as error:
            print(f"[FETCH ERROR] {url}: {error}")
            new_sources.append({
                "url": url,
                "title": "",
                "content": "",
                "status": "failed",
                "error": str(error),
            })

    all_sources = existing_sources + new_sources
    successful = get_successful_sources(all_sources)

    print(
        f"[READER] {len(successful)}/{len(all_sources)} "
        "sources fetched successfully"
    )

    reader_notes = "\n\n".join(
        (
            f"Source ID: {source.get('source_id')}\n"
            f"Title: {source.get('title')}\n"
            f"URL: {source.get('url')}\n"
            f"Status: {source.get('status')}"
        )
        for source in all_sources
    )

    return {
        "sources": all_sources,
        "reader_notes": reader_notes,
    }


# ============================================================
# NODE 4.5: INCREMENTAL PLANNER (V3.7)
# ============================================================
def incremental_planning_node(state: ResearchState):
    print("\n[INCREMENTAL] Checking sources against stored memory")

    all_sources = state.get("sources", [])
    fetched_sources = get_successful_sources(all_sources)
    fetch_failures = [
        {"url": s.get("url"), "error": s.get("error", "fetch_failed")}
        for s in all_sources
        if s.get("status") != "fetched"
    ]

    plan = build_incremental_plan(fetched_sources)
    combined_failures = fetch_failures + plan["failed_sources"]

    print(
        f"[INCREMENTAL] Unchanged: {len(plan['unchanged_sources'])} | "
        f"Changed: {len(plan['changed_sources'])} | "
        f"New: {len(plan['new_sources'])} | "
        f"Failed: {len(combined_failures)}"
    )

    return {
        "incremental_plan": plan,
        "unchanged_sources": plan["unchanged_sources"],
        "changed_sources": plan["changed_sources"],
        "new_sources": plan["new_sources"],
        "failed_sources": combined_failures,
    }


# ============================================================
# NODE 5: SOURCE INTELLIGENCE
# ============================================================
def source_intelligence_node(state: ResearchState):
    print("\n[SOURCE INTELLIGENCE] Analyzing sources")

    fetched_sources = get_successful_sources(state.get("sources", []))
    intelligence = []

    for source in fetched_sources:
        try:
            metadata = analyze_source(source)
            llm_metadata = classify_source_with_llm(source)
            intelligence.append({**metadata, **llm_metadata})
        except Exception as error:
            print(f"[SOURCE INTELLIGENCE ERROR] {source.get('url')}: {error}")
            intelligence.append({
                "source_id": source.get("source_id"),
                "analysis_status": "failed",
                "error": str(error),
            })

    relationships = analyze_source_independence(fetched_sources)

    return {
        "source_intelligence": intelligence,
        "source_relationships": relationships,
    }


# ============================================================
# NODE 6: EVIDENCE VERIFICATION
# ============================================================
def verification_node(state: ResearchState):
    print("\n[VERIFIER] Extracting claims and checking evidence")

    fetched_sources = get_successful_sources(state.get("sources", []))

    if not fetched_sources:
        return {
            "evidence_records": [],
            "verification_result": {
                "total_claims": 0,
                "supported": 0,
                "insufficient": 0,
                "contradicted": 0,
                "coverage": 0.0,
            },
            "verification_status": "research_more",
            "missing_evidence": ["No successfully fetched source documents."],
        }

    claims = extract_research_claims(
        topic=state["topic"],
        sources=fetched_sources,
    )

    evidence_records = build_evidence_records(
        claims=claims,
        sources=fetched_sources,
    )

    supported = sum(1 for item in evidence_records if item.get("status") == "supported")
    insufficient = sum(1 for item in evidence_records if item.get("status") == "insufficient")
    contradicted = sum(1 for item in evidence_records if item.get("status") == "contradicted")

    total = len(evidence_records)
    coverage = supported / total if total else 0.0

    missing = [
        item.get("claim", "Unknown claim")
        for item in evidence_records
        if item.get("status") in {"insufficient", "contradicted"}
    ]

    recommendation = (
        "accept"
        if total > 0 and coverage >= 0.70 and contradicted == 0
        else "research_more"
    )

    print(
        f"[VERIFIER] Supported: {supported} | "
        f"Insufficient: {insufficient} | "
        f"Contradicted: {contradicted} | "
        f"Coverage: {coverage:.2%}"
    )

    return {
        "evidence_records": evidence_records,
        "verification_result": {
            "total_claims": total,
            "supported": supported,
            "insufficient": insufficient,
            "contradicted": contradicted,
            "coverage": coverage,
        },
        "verification_status": recommendation,
        "missing_evidence": missing,
    }


# ============================================================
# NODE 7: CONTRADICTION DETECTION (V3.3)
# ============================================================
def contradiction_node(state: ResearchState):
    print("\n[CONTRADICTION] Detecting conflicting claims")

    detection = detect_contradictions(
        evidence_records=state.get("evidence_records", []),
        source_relationships=state.get("source_relationships", []),
    )

    conflict_report = build_conflict_report(
        claim_comparisons=detection["claim_comparisons"],
        source_relationships=state.get("source_relationships", []),
    )

    return {
        "normalized_claims": detection["normalized_claims"],
        "claim_comparisons": detection["claim_comparisons"],
        "conflict_report": conflict_report,
        "unresolved_conflicts": [
            item["explanation"]
            for item in conflict_report["unresolved_conflicts"]
        ],
    }


# ============================================================
# NODE 7.5: HISTORICAL CLAIM COMPARISON (V3.8)
# ============================================================
def historical_comparison_node(state: ResearchState):
    print("\n[HISTORY] Comparing claims against historical runs")

    current_claims = state.get("normalized_claims", [])
    fingerprints = [
        fingerprint_claim(item["claim"])
        for item in current_claims
        if item.get("claim")
    ]

    previous_claims = get_previous_claims(fingerprints)

    serialized_previous = [
        item if isinstance(item, dict) else {
            "claim_fingerprint": item.claim_fingerprint,
            "claim_text": item.claim_text,
            "verification_status": item.verification_status,
            "evidence_summary": item.evidence_summary,
        }
        for item in previous_claims
    ]

    result = analyze_historical_changes(
        current_claims=current_claims,
        previous_claims=serialized_previous,
    )

    print(
        f"[HISTORY] Unchanged: {len(result['comparison']['unchanged'])} | "
        f"New: {len(result['comparison']['new_claims'])} | "
        f"Status Changed: {len(result['comparison']['status_changed'])} | "
        f"Conflicts: {len(result['historical_conflicts'])}"
    )

    return {
        "previous_claims": serialized_previous,
        "historical_comparison": result["comparison"],
        "historical_conflicts": result["historical_conflicts"],
        "historical_review_required": result["requires_attention"],
    }


# ============================================================
# NODE 8: GAP EVALUATOR (V3.5)
# ============================================================
def gap_evaluation_node(state: ResearchState):
    print("\n[GAP EVALUATOR] Checking research coverage")

    assessment = assess_question_coverage(
        research_questions=state.get("research_questions", []),
        evidence_records=state.get("evidence_records", []),
    )

    feedback = build_gap_feedback(assessment["assessments"])

    gaps = [
        item["question"]
        for item in assessment["assessments"]
        if item["status"] != "answered"
    ]

    print(
        f"[GAP EVALUATOR] "
        f"Answered: {assessment['answered']} | "
        f"Partial: {assessment['partially_answered']} | "
        f"Unanswered: {assessment['unanswered']} | "
        f"Coverage: {assessment['coverage']:.2%}"
    )

    return {
        "gap_assessment": assessment,
        "gap_coverage": assessment["coverage"],
        "research_questions": assessment["assessments"],
        "research_gaps": gaps,
        "research_feedback": feedback,
    }


# ============================================================
# NODE 9: RESEARCH QUALITY EVALUATOR (V3.4)
# ============================================================
def quality_evaluation_node(state: ResearchState):
    print("\n[QUALITY] Evaluating research quality")

    quality = evaluate_research_quality(
        sources=state.get("sources", []),
        source_intelligence=state.get("source_intelligence", []),
        evidence_records=state.get("evidence_records", []),
        research_gaps=state.get("research_gaps", []),
        unresolved_conflicts=state.get("unresolved_conflicts", []),
        freshness_days=365,
    )

    print(f"[QUALITY] Score: {quality['overall_percentage']}%")
    print(f"[QUALITY] Status: {quality['status']}")

    return {
        "research_quality": quality,
        "quality_status": quality["status"],
    }


# ============================================================
# NODE 10: PREPARE RETRY
# ============================================================
def prepare_retry_node(state: ResearchState):
    next_retry = state.get("retry_count", 0) + 1

    gap_feedback = state.get("research_feedback", "").strip()
    missing = state.get("missing_evidence", [])

    if gap_feedback:
        feedback = gap_feedback
    elif missing:
        feedback = (
            "The previous research attempt lacked sufficient "
            "verified evidence for these claims.\n\n"
            "Find additional primary or independent sources "
            "that support, refute, or clarify:\n\n- "
            + "\n- ".join(missing[:5])
        )
    else:
        feedback = (
            "Find additional primary sources, independent "
            "confirmations and verifiable quantitative evidence."
        )

    print(f"\n[RETRY {next_retry}] Preparing targeted follow-up research")

    return {
        "retry_count": next_retry,
        "research_feedback": feedback,
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================
def route_after_verification(state: ResearchState):
    verification_status = state.get("verification_status", "research_more")
    quality_status = state.get("quality_status", "needs_review")
    gap_assessment = state.get("gap_assessment", {})
    high_priority_gaps = gap_assessment.get("high_priority_unanswered", [])

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    needs_more_research = (
        verification_status == "research_more"
        or quality_status == "low"
        or bool(high_priority_gaps)
    )

    if needs_more_research and retry_count < max_retries:
        print(
            "\n[ROUTER] Research gaps or quality issues found. "
            "Starting targeted retry."
        )
        return "prepare_retry"

    if needs_more_research:
        print(
            "\n[ROUTER] Retry limit reached. "
            "Generating report with unresolved gaps disclosed."
        )

    return "writer"


# ============================================================
# NODE 11: FINAL WRITER
# ============================================================
def writer_node(state: ResearchState):
    print("\n[WRITER] Generating evidence-linked report")

    writer = build_writer_chain()

    report_notes = f"""
RESEARCH OBJECTIVE
==================

{state.get('research_objective', '')}


RESEARCH QUESTIONS AND COVERAGE
===============================

{state.get('research_questions', [])}


GAP ASSESSMENT
==============

{state.get('gap_assessment', {})}


REMAINING RESEARCH GAPS
=======================

{state.get('research_gaps', [])}


INCREMENTAL SOURCE SUMMARY
==========================

Unchanged Sources: {len(state.get('unchanged_sources', []))}
Changed Sources: {len(state.get('changed_sources', []))}
New Sources: {len(state.get('new_sources', []))}
Failed Sources: {len(state.get('failed_sources', []))}


HISTORICAL RESEARCH ANALYSIS
============================

Previous Research Findings: {state.get('previous_claims', [])}
Historical Comparison: {state.get('historical_comparison', {})}
Historical Conflicts: {state.get('historical_conflicts', [])}
Historical Review Required: {state.get('historical_review_required', False)}


RESEARCH FINDINGS
=================

{state.get('combined_search', '')}


SOURCE DOCUMENTS
================

{state.get('reader_notes', '')}


SOURCE INTELLIGENCE
===================

{state.get('source_intelligence', [])}


SOURCE INDEPENDENCE ANALYSIS
============================

{state.get('source_relationships', [])}


VERIFIED EVIDENCE RECORDS
=========================

{state.get('evidence_records', [])}


VERIFICATION SUMMARY
====================

{state.get('verification_result', {})}


CONTRADICTION ANALYSIS
======================

{state.get('conflict_report', {})}


UNRESOLVED CONFLICTS
====================

{state.get('unresolved_conflicts', [])}


RESEARCH QUALITY EVALUATION
===========================

{state.get('research_quality', {})}


REPORT REQUIREMENTS
===================

1. Every important factual claim must cite a verified source ID
   and its original URL.

2. Distinguish primary sources from secondary sources.

3. Do not count syndicated articles as independent confirmations.

4. Do not invent citations or source details.

5. Separate confirmed findings from uncertain findings.

6. Clearly describe contradictions and unresolved disagreements.

7. Do not present unsupported claims as established facts.

8. Include a structured References section.

9. State research limitations and missing evidence.

10. Address every planned research question.

11. Explicitly mark unanswered or partially answered questions.

12. Do not claim complete research when important gaps remain.

13. Explain when the retry limit prevented further investigation.

14. Include a "Historical Changes" section explaining unchanged findings,
    newly discovered findings, and previously supported claims that are now
    contradicted or insufficiently supported.
"""

    report = tracked_invoke(
        writer,
        {
            "topic": state["topic"],
            "notes": report_notes,
            "previous_claims": state.get("previous_claims", []),
            "historical_comparison": state.get("historical_comparison", {}),
            "historical_conflicts": state.get("historical_conflicts", []),
        },
        run_id=state["run_id"],
        node_name="writer",
    )

    return {"final_report": report}


# ============================================================
# NODE 11.5: SAVE CLAIM HISTORY (V3.8)
# ============================================================
def save_claim_history_node(state: ResearchState):
    print("\n[HISTORY] Saving final claim snapshots")

    run_id = state["run_id"]
    snapshots = [
        create_claim_snapshot(
            run_id=run_id,
            claim=claim,
        )
        for claim in state.get("normalized_claims", [])
        if claim.get("claim")
    ]

    if snapshots:
        save_claim_snapshots(snapshots)

    print(f"[HISTORY] Saved {len(snapshots)} claim snapshots")
    return {"claim_snapshots": snapshots}


# ============================================================
# NODE 12: PERSIST RESEARCH MEMORY (V3.6)
# ============================================================
def persist_research_memory_node(state: ResearchState):
    print("\n[MEMORY] Persisting sources, evidence, and run status")

    run_id = state["run_id"]
    saved_sources = []

    for source in state.get("sources", []):
        if source.get("status") != "fetched":
            continue
        if not source.get("content"):
            continue

        _, is_new = save_source(source)
        saved_sources.append({
            "url": source["url"],
            "is_new": is_new,
        })

    for evidence in state.get("evidence_records", []):
        if evidence.get("source_id"):
            save_evidence(
                run_id=run_id,
                evidence=evidence,
            )

    complete_run(run_id)

    persisted_new = [item for item in saved_sources if item["is_new"]]
    print(
        f"[MEMORY] Saved {len(saved_sources)} sources "
        f"({len(persisted_new)} new) and {len(state.get('evidence_records', []))} evidence records"
    )

    return {"reused_sources": state.get("unchanged_sources", [])}


# ============================================================
# BUILD RESEARCH GRAPH
# ============================================================
def build_research_graph():
    graph = StateGraph(ResearchState)

    # Register all nodes (instrumented with track_node)
    graph.add_node("memory_initialization", memory_initialization_node)
    graph.add_node("research_planner", track_node("research_planner", research_planner_node))
    graph.add_node("technical_researcher", track_node("technical_researcher", technical_researcher))
    graph.add_node("industry_researcher", track_node("industry_researcher", industry_researcher))
    graph.add_node("aggregator", track_node("aggregator", aggregate_research))
    graph.add_node("reader", track_node("reader", reader_node))
    graph.add_node("incremental_planner", track_node("incremental_planner", incremental_planning_node))
    graph.add_node("source_intelligence", track_node("source_intelligence", source_intelligence_node))
    graph.add_node("verifier", track_node("verifier", verification_node))
    graph.add_node("contradiction", track_node("contradiction", contradiction_node))
    graph.add_node("historical_comparison", track_node("historical_comparison", historical_comparison_node))
    graph.add_node("gap_evaluator", track_node("gap_evaluator", gap_evaluation_node))
    graph.add_node("quality_evaluator", track_node("quality_evaluator", quality_evaluation_node))
    graph.add_node("prepare_retry", track_node("prepare_retry", prepare_retry_node))
    graph.add_node("writer", track_node("writer", writer_node))
    graph.add_node("save_claim_history", track_node("save_claim_history", save_claim_history_node))
    graph.add_node("persist_memory", track_node("persist_memory", persist_research_memory_node))

    # 1. Initialize persistent memory, then plan
    graph.add_edge(START, "memory_initialization")
    graph.add_edge("memory_initialization", "research_planner")

    # 2. Fan-out to parallel researchers after planning
    graph.add_edge("research_planner", "technical_researcher")
    graph.add_edge("research_planner", "industry_researcher")

    # 3. Wait for both researchers before aggregating
    graph.add_edge(
        ["technical_researcher", "industry_researcher"],
        "aggregator",
    )

    # 4. Sequential verification, contradiction, historical comparison, gap & quality pipeline
    graph.add_edge("aggregator", "reader")
    graph.add_edge("reader", "incremental_planner")
    graph.add_edge("incremental_planner", "source_intelligence")
    graph.add_edge("source_intelligence", "verifier")
    graph.add_edge("verifier", "contradiction")
    graph.add_edge("contradiction", "historical_comparison")
    graph.add_edge("historical_comparison", "gap_evaluator")
    graph.add_edge("gap_evaluator", "quality_evaluator")

    # 5. Conditional retry routing after quality evaluation
    graph.add_conditional_edges(
        "quality_evaluator",
        route_after_verification,
        {
            "prepare_retry": "prepare_retry",
            "writer": "writer",
        },
    )

    # 6. Retry: re-run researchers with targeted gap feedback (keeps original plan)
    graph.add_edge("prepare_retry", "technical_researcher")
    graph.add_edge("prepare_retry", "industry_researcher")

    # 7. Save final claim snapshots & persist memory only after final report
    graph.add_edge("writer", "save_claim_history")
    graph.add_edge("save_claim_history", "persist_memory")
    graph.add_edge("persist_memory", END)

    return graph.compile()


# ============================================================
# RUN RESEARCH
# ============================================================
def run_research(
    topic: str,
    max_retries: int = 2,
):
    init_database()
    run_id = create_run(topic)
    research_graph = build_research_graph()

    initial_state: ResearchState = {
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
    }

    try:
        return research_graph.invoke(initial_state)
    except Exception:
        # Mark run failed if an exception interrupts execution
        from memory.database import SessionLocal
        from memory.models import ResearchRun

        with SessionLocal() as db:
            run = db.get(ResearchRun, run_id)
            if run and run.status == "running":
                run.status = "failed"
                db.commit()
        raise


if __name__ == "__main__":
    result = run_research(
        topic="Latest developments in AI agent architectures",
        max_retries=2,
    )

    print("\n\n========== FINAL REPORT ==========\n")
    print(result.get("final_report", "No report generated."))