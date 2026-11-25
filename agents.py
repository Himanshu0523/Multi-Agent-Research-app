from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_Search, scrape_url

load_dotenv()


# Shared LLM Configuration
def get_llm(temperature=0):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
    )


# AGENT 1: RESEARCHER

def build_search_agent(focus: str = "general research"):

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[web_Search],
        system_prompt=f"""
        You are an independent research specialist.

        Your research focus: {focus}

        Responsibilities:
        1. Investigate the given topic.
        2. Search for reliable and relevant sources.
        3. Extract factual information and important findings.
        4. Include source titles and URLs.
        5. Identify conflicting information.
        6. Never invent facts or citations.

        Return structured research findings.
        """,
    )

    return agent


# AGENT 2: READER / SCRAPER
def build_reader_agent():

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
        You are an expert web content analysis agent.

        Your responsibilities:
        1. Analyze the research findings.
        2. Identify the most relevant URLs.
        3. Use scrape_url to read relevant webpages.
        4. Extract detailed facts, statistics and evidence.
        5. Preserve source URLs.
        6. Never invent information.
        7. Report inaccessible pages honestly.

        Return detailed research notes.
        """,
    )

    return agent


# AGENT 3: WRITER (LCEL PIPELINE)
def build_writer_chain():

    llm = get_llm(temperature=0.3)

    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a senior research writer.

                Synthesize verified research findings into
                a clear, structured and factual report.

                Do not invent facts, statistics or citations.
                Distinguish verified findings from uncertainty.
                """,
            ),
            (
                "human",
                """
                Topic: {topic}

                Research Notes:
                {notes}

                Write a structured report containing:

                1. Introduction
                2. Background
                3. Key Findings
                4. Detailed Analysis
                5. Conclusion
                6. Sources (include available URLs)

                Maintain clarity, accuracy and neutrality.
                """,
            ),
        ]
    )

    # LCEL Runnable Pipeline
    writer_chain = (
        writer_prompt
        | llm
        | StrOutputParser()
    )

    return writer_chain
