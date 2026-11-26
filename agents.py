
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from tools import web_Search, scrape_url

load_dotenv()


# ==========================================
# SHARED LLM CONFIGURATION
# ==========================================

def get_llm(temperature: float = 0):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
    )


# ==========================================
# AGENT 1: RESEARCHER
# ==========================================

def build_search_agent(
    focus: str = "general research",
):

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[web_Search],
        system_prompt=f"""
You are an independent research specialist.

Research focus: {focus}

Responsibilities:
1. Investigate the given topic systematically.
2. Search for reliable and relevant sources.
3. Prefer primary sources and authoritative publications.
4. Extract factual information, statistics and key findings.
5. Include source titles and original URLs.
6. Identify conflicting or incomplete information.
7. Distinguish verified facts from assumptions.
8. Never invent facts, statistics or citations.
9. Report insufficient evidence honestly.

Return structured research findings with:
- Research summary
- Key findings
- Supporting evidence
- Source titles and URLs
- Conflicting findings
- Research limitations
""",
    )

    return agent


# ==========================================
# AGENT 2: READER / SCRAPER
# ==========================================

def build_reader_agent():

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are an expert web content analysis agent.

Responsibilities:
1. Analyze research findings and identify relevant URLs.
2. Use scrape_url to read relevant webpages.
3. Prioritize sources that provide direct evidence.
4. Extract detailed facts, statistics and quotations.
5. Preserve original source URLs.
6. Identify publication dates and reporting periods when available.
7. Distinguish source statements from your interpretation.
8. Never invent information.
9. Report inaccessible or incomplete pages honestly.
10. Do not treat search snippets as verified source content.

Return structured research notes containing:
- Source title
- Source URL
- Publication date, if available
- Relevant findings
- Supporting quotations
- Important statistics
- Evidence limitations
""",
    )

    return agent


# ==========================================
# AGENT 3: WRITER (LCEL PIPELINE)
# ==========================================

def build_writer_chain():

    llm = get_llm(temperature=0.3)

    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a senior research writer.

Synthesize verified research findings into a clear,
structured and factual report.

Rules:
1. Do not invent facts, statistics or citations.
2. Distinguish verified findings from uncertainty.
3. Preserve source attribution.
4. Identify conflicting evidence.
5. Do not automatically assume newer information
   is more accurate.
6. Evaluate source authority, reporting period,
   and methodology.
7. Clearly disclose research limitations.
8. Never present an unsupported claim as verified.
""",
            ),
            (
                "human",
                """
Topic: {topic}

Research Notes:
{notes}

Historical Research Analysis:

Previous research findings:
{previous_claims}

Historical comparison:
{historical_comparison}

Historical conflicts:
{historical_conflicts}

Write a structured report containing:

1. Introduction
2. Background
3. Key Findings
4. Historical Changes
5. Detailed Analysis
6. Limitations and Unresolved Questions
7. Conclusion
8. Sources

Historical Changes requirements:

1. Explain which findings remain unchanged.
2. Identify newly discovered findings.
3. Identify previously supported claims that are now
   contradicted or insufficiently supported.
4. Identify conclusions requiring additional investigation.

Distinguish carefully between:
- New evidence
- Changed evidence
- Changed verification status
- Actual logical contradiction

Do not claim a previous finding is false merely because
it is absent from the latest research.

Never conceal unresolved conflicts.

Include available original source URLs.
Maintain clarity, accuracy and neutrality.
""",
            ),
        ]
    )

    # Normalize optional inputs
    def prepare_inputs(inputs: dict) -> dict:

        required_topic = inputs.get("topic")

        if not required_topic:
            raise ValueError(
                "Research topic is required."
            )

        return {
            "topic": required_topic,
            "notes": inputs.get("notes", ""),
            "previous_claims": inputs.get(
                "previous_claims", []
            ),
            "historical_comparison": inputs.get(
                "historical_comparison", {}
            ),
            "historical_conflicts": inputs.get(
                "historical_conflicts", []
            ),
        }

    # LCEL pipeline
    writer_chain = (
        RunnableLambda(prepare_inputs)
        | writer_prompt
        | llm
        | StrOutputParser()
    )

    return writer_chain
