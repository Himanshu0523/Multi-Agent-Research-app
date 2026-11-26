import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agents import build_search_agent, build_writer_chain
from tools import fetch_document
from evidence import (
    extract_research_claims,
    build_evidence_records,
)
from test.source_intelligence import (
    analyze_source,
    classify_source_with_llm,
)
from source_independence import analyze_source_independence
from contradiction import (
    detect_contradictions,
    build_conflict_report,
)
from quality import evaluate_research_quality
from gap_planner import (
    create_research_plan,
    assess_question_coverage,
    build_gap_feedback,
)


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
# NODE 0: RESEARCH PLANNER (V3.5)
# ============================================================
def research_planner_node(state: ResearchState):
    print("\n[PLANNER] Creating research plan")

    plan = create_research_plan(
        topic=state["topic"],
        max_questions=8,
    )

    print(f"[PLANNER] Created {len(plan['research_questions'])} questions")
    return plan


# NODE 1: TECHNICAL RESEARCHER
def technical_researcher(state: ResearchState):
    print("\n[AGENT A] Technical research started")

    agent = build_search_agent()
    feedback = state.get("research_feedback", "")
    questions_text = format_questions_for_prompt(
        state.get("research_questions", [])
    )

    prompt = f"""
Research topic: {state['topic']}

RESEARCH QUESTIONS TO INVESTIGATE:
{questions_text}

Research these questions directly.
Prioritize high-importance questions.
Identify which questions have insufficient evidence.

Your research focus:
- Scientific breakthroughs
- Engineering developments
- Technical evidence
- Research papers
- Benchmarks and experimental results

Additional research requirements:
{feedback}

Find relevant sources, dates, evidence and URLs.

Distinguish:
- Peer-reviewed research
- Preprints
- Official technical documentation
- Company announcements
- Speculation

Return findings with source titles and original URLs.
"""

    result = agent.invoke({"input": prompt})
    return {"search_a": result["output"]}


# NODE 2: INDUSTRY RESEARCHER
def industry_researcher(state: ResearchState):
    print("\n[AGENT B] Industry research started")

    agent = build_search_agent()
    feedback = state.get("research_feedback", "")
    questions_text = format_questions_for_prompt(
        state.get("research_questions", [])
    )

    prompt = f"""
Research topic: {state['topic']}

RESEARCH QUESTIONS TO INVESTIGATE:
{questions_text}

Research these questions directly.
Prioritize high-importance questions.
Identify which questions have insufficient evidence.

Your research focus:
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

    result = agent.invoke({"input": prompt})
    return {"search_b": result["output"]}


# NODE 3: AGGREGATOR
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


# NODE 4: READER
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


# NODE 5: SOURCE INTELLIGENCE
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


# NODE 6: EVIDENCE VERIFICATION
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


# NODE 7: CONTRADICTION DETECTION (V3.3)
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


# NODE 8: GAP EVALUATOR (V3.5)
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


# NODE 9: RESEARCH QUALITY EVALUATOR (V3.4)
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


# NODE 10: PREPARE RETRY
def prepare_retry_node(state: ResearchState):
    next_retry = state.get("retry_count", 0) + 1

    # Prefer gap-driven feedback from gap_evaluator; fall back to missing_evidence
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


# CONDITIONAL ROUTER
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


# NODE 11: FINAL WRITER
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
"""

    report = writer.invoke({
        "topic": state["topic"],
        "notes": report_notes,
    })

    return {"final_report": report}


# ============================================================
# BUILD RESEARCH GRAPH
# ============================================================
def build_research_graph():
    graph = StateGraph(ResearchState)

    # Register all nodes
    graph.add_node("research_planner", research_planner_node)
    graph.add_node("technical_researcher", technical_researcher)
    graph.add_node("industry_researcher", industry_researcher)
    graph.add_node("aggregator", aggregate_research)
    graph.add_node("reader", reader_node)
    graph.add_node("source_intelligence", source_intelligence_node)
    graph.add_node("verifier", verification_node)
    graph.add_node("contradiction", contradiction_node)
    graph.add_node("gap_evaluator", gap_evaluation_node)
    graph.add_node("quality_evaluator", quality_evaluation_node)
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node("writer", writer_node)

    # 1. Plan first
    graph.add_edge(START, "research_planner")

    # 2. Fan-out to parallel researchers after planning
    graph.add_edge("research_planner", "technical_researcher")
    graph.add_edge("research_planner", "industry_researcher")

    # 3. Wait for both researchers before aggregating
    graph.add_edge(
        ["technical_researcher", "industry_researcher"],
        "aggregator",
    )

    # 4. Sequential verification, contradiction, gap & quality pipeline
    graph.add_edge("aggregator", "reader")
    graph.add_edge("reader", "source_intelligence")
    graph.add_edge("source_intelligence", "verifier")
    graph.add_edge("verifier", "contradiction")
    graph.add_edge("contradiction", "gap_evaluator")
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

    # 7. Finish
    graph.add_edge("writer", END)

    return graph.compile()



# RUN RESEARCH
def run_research(
    topic: str,
    max_retries: int = 2,
):
    research_graph = build_research_graph()

    initial_state: ResearchState = {
        "topic": topic,
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

    return research_graph.invoke(initial_state)


if __name__ == "__main__":
    result = run_research(
        topic="Latest developments in AI agent architectures",
        max_retries=2,
    )

    print("\n\n========== FINAL REPORT ==========\n")
    print(result.get("final_report", "No report generated."))