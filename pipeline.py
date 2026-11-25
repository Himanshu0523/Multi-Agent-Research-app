
from graph import build_research_graph


def run_research_pipeline(topic: str) -> dict:

    research_graph = build_research_graph()

    print("\n" + "=" * 65)
    print("MULTI-AGENT RESEARCH SYSTEM")
    print("=" * 65)

    result = research_graph.invoke(
        {"topic": topic}
    )

    print("\n" + "=" * 65)
    print("FINAL RESEARCH REPORT")
    print("=" * 65)

    print(result["final_report"])

    return {
        "topic": topic,
        "technical_research": result["search_a"],
        "industry_research": result["search_b"],
        "combined_research": result["combined_search"],
        "reader_notes": result["reader_notes"],
        "report": result["final_report"],
    }


if __name__ == "__main__":

    result = run_research_pipeline(
        "Latest breakthroughs in Solid State Batteries 2026"
    )
