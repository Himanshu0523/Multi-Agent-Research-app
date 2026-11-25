from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agents import (
    build_search_agent,
    build_reader_agent,
    build_writer_chain,
)
from tools import fetch_document
from evidence import (
    extract_research_claims,
    build_evidence_records,
)
import re


from verification import verify_research


# SHARED RESEARCH STATE
class ResearchState(TypedDict, total=False):

    topic: str

    search_a: str
    search_b: str
    combined_search: str

    reader_notes: str

    verification_result: dict
    verification_status: str
    missing_evidence: list[str]
    contradictions: list[str]

    retry_count: int
    max_retries: int
    research_feedback: str

    final_report: str
    sources: list[dict]
    evidence_records: list[dict]


# NODE 1: TECHNICAL RESEARCHER
def technical_researcher(state: ResearchState):

    print("\n[AGENT A] Technical research started")

    agent = build_search_agent(
        focus="Scientific breakthroughs, engineering "
              "developments, technical evidence and research papers"
    )

    feedback = state.get("research_feedback", "")

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"""
            Research topic: {state['topic']}

            Additional research requirements:
            {feedback}

            Investigate technical developments.
            Find relevant sources, dates, evidence and URLs.
            Distinguish published results from announcements.
            """
        }]
    })

    return {
        "search_a": result["messages"][-1].content
    }


# NODE 2: INDUSTRY RESEARCHER
def industry_researcher(state: ResearchState):

    print("\n[AGENT B] Industry research started")

    agent = build_search_agent(
        focus="Industry developments, commercialization, "
              "market trends, companies and product announcements"
    )

    feedback = state.get("research_feedback", "")

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"""
            Research topic: {state['topic']}

            Additional research requirements:
            {feedback}

            Investigate industry developments.
            Find relevant sources, dates, commercial evidence
            and URLs.
            """
        }]
    })

    return {
        "search_b": result["messages"][-1].content
    }


# NODE 3: AGGREGATOR
def aggregate_research(state: ResearchState):

    print("\n[AGGREGATOR] Combining research findings")

    combined = (
        "TECHNICAL RESEARCH:\n"
        + state.get("search_a", "")
        + "\n\nINDUSTRY RESEARCH:\n"
        + state.get("search_b", "")
    )

    return {
        "combined_search": combined
    }


# NODE 4: READER AGENT

def reader_node(state: ResearchState):

    print("\n[READER] Collecting source documents")

    # Extract URLs from both search results
    urls = re.findall(
        r'https?://[^\s<>"\']+',
        state["combined_search"],
    )

    # Normalize and deduplicate
    unique_urls = list(dict.fromkeys(
        url.rstrip(".,);]")
        for url in urls
    ))

    # Limit the number of pages per research cycle
    selected_urls = unique_urls[:5]

    sources = []

    for url in selected_urls:

        print(f"[FETCHING] {url}")

        document = fetch_document(url)

        sources.append(document)

    successful = sum(
        1 for source in sources
        if source.get("status") == "fetched"
    )

    print(
        f"[READER] {successful}/{len(sources)} "
        "sources fetched successfully"
    )

    return {
        "sources": sources,
        "reader_notes": "\n".join(
            f"Title: {s.get('title')}\n"
            f"URL: {s.get('url')}\n"
            f"Status: {s.get('status')}"
            for s in sources
        ),
    }



# NODE 5: VERIFICATION ENGINE

def verification_node(state: ResearchState):

    print("\n[VERIFIER] Extracting claims and checking provenance")

    claims = extract_research_claims(
        topic=state["topic"],
        sources=state.get("sources", []),
    )

    evidence_records = build_evidence_records(
        claims=claims,
        sources=state.get("sources", []),
    )

    supported = sum(
        1 for item in evidence_records
        if item["status"] == "supported"
    )

    insufficient = sum(
        1 for item in evidence_records
        if item["status"] == "insufficient"
    )

    contradicted = sum(
        1 for item in evidence_records
        if item["status"] == "contradicted"
    )

    print(f"Supported: {supported}")
    print(f"Insufficient: {insufficient}")
    print(f"Contradicted: {contradicted}")

    missing = [
        item["claim"]
        for item in evidence_records
        if item["status"] == "insufficient"
    ]

    # A simple evidence-coverage gate.
    # This is not a guarantee of factual correctness.
    total = len(evidence_records)

    coverage = (
        supported / total
        if total else 0
    )

    recommendation = (
        "accept"
        if total > 0
        and coverage >= 0.70
        and contradicted == 0
        else "research_more"
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

# NODE 6: CONDITIONAL ROUTER
def route_after_verification(state: ResearchState):

    status = state.get("verification_status", "research_more")

    retry_count = state.get("retry_count", 0)

    max_retries = state.get("max_retries", 1)

    if status == "research_more" and retry_count <= max_retries:

        print("\n[ROUTER] Insufficient evidence. Researching again.")

        return [
            "technical_researcher",
            "industry_researcher",
        ]

    print("\n[ROUTER] Moving to report generation")

    return ["writer"]


# NODE 7: FINAL WRITER

def writer_node(state: ResearchState):
    print("\n[WRITER] Generating evidence-linked report")
    writer = build_writer_chain()

    report_notes = f"""
    RESEARCH FINDINGS:

    {state.get('combined_search', '')}

    SOURCE DOCUMENTS:

    {state.get('reader_notes', '')}

    VERIFIED EVIDENCE RECORDS:

    {state.get('evidence_records', [])}

    VERIFICATION SUMMARY:

    {state.get('verification_result', {})}

    REPORT REQUIREMENTS:

    1. Every important factual claim must have a source.
    2. Use source IDs and original URLs.
    3. Do not invent citations.
    4. Do not present unsupported claims as verified.
    5. Clearly identify contradictions.
    6. Separate confirmed findings from uncertain findings.
    7. Include a references section.
    """

    report = writer.invoke({
        "topic": state["topic"],
        "notes": report_notes,
    })

    return {
        "final_report": report
    }

# BUILD RESEARCH GRAPH
def build_research_graph():

    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node(
        "technical_researcher",
        technical_researcher
    )

    graph.add_node(
        "industry_researcher",
        industry_researcher
    )

    graph.add_node(
        "aggregator",
        aggregate_research
    )

    graph.add_node(
        "reader",
        reader_node
    )

    graph.add_node(
        "verifier",
        verification_node
    )

    graph.add_node(
        "writer",
        writer_node
    )

    # Initial parallel research
    graph.add_edge(
        START,
        "technical_researcher"
    )

    graph.add_edge(
        START,
        "industry_researcher"
    )

    # Wait for both researchers
    graph.add_edge(
        [
            "technical_researcher",
            "industry_researcher",
        ],
        "aggregator"
    )

    # Sequential research
    graph.add_edge("aggregator", "reader")

    graph.add_edge("reader", "verifier")

    # Conditional verification routing
    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
    )

    graph.add_edge("writer", END)

    return graph.compile()
