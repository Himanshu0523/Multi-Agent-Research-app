from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI



# VERIFICATION SCHEMAS
class VerifiedClaim(BaseModel):

    claim: str = Field(
        description="A specific factual claim from the research"
    )

    evidence: str = Field(
        description="Supporting or contradicting evidence from scraped content"
    )

    source_url: str = Field(
        description="URL of the source supporting the evidence"
    )

    status: Literal[
        "supported",
        "contradicted",
        "insufficient"
    ] = Field(
        description="Whether the supplied evidence supports the claim"
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in the evidence assessment"
    )

    explanation: str = Field(
        description="Why this verification status was assigned"
    )


class VerificationResult(BaseModel):

    claims: list[VerifiedClaim] = Field(
        description="Individual claims and their evidence assessments"
    )

    missing_evidence: list[str] = Field(
        description="Important research questions that lack sufficient evidence"
    )

    contradictions: list[str] = Field(
        description="Conflicting findings between supplied sources"
    )

    overall_quality: Literal[
        "high",
        "medium",
        "low"
    ] = Field(
        description="Overall evidence quality of the research"
    )

    recommendation: Literal[
        "accept",
        "research_more"
    ] = Field(
        description="Whether more research is necessary"
    )


# VERIFICATION ENGINE
def build_verification_chain():

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    structured_llm = llm.with_structured_output(
        VerificationResult
    )

    return structured_llm


def verify_research(
    topic: str,
    search_notes: str,
    reader_notes: str,
):

    verifier = build_verification_chain()

    prompt = f"""
    You are an independent research verification specialist.

    Research Topic:
    {topic}

    Web Search Findings:
    {search_notes}

    Scraped Source Content:
    {reader_notes}

    TASK:

    1. Identify important factual claims.
    2. Compare each claim with the supplied scraped evidence.
    3. Mark each claim as:
       - supported
       - contradicted
       - insufficient
    4. Include the exact supporting evidence where available.
    5. Preserve the actual source URL.
    6. Identify contradictions between sources.
    7. Identify missing evidence.
    8. Assess overall evidence quality.

    RULES:

    - Never invent source URLs.
    - Never invent quotations.
    - Search snippets alone are weaker evidence than full source content.
    - Do not mark a claim supported merely because an LLM
      generated it in the research notes.
    - A missing source or missing evidence must remain insufficient.
    - Do not interpret model confidence as statistical probability.
    - Do not confuse a source's reputation with proof of a claim.

    Accept the research only when the important claims have
    adequate supporting evidence.
    """

    return verifier.invoke(prompt)



# SOURCE QUALITY HEURISTIC
def estimate_source_tier(url: str) -> str:

    """
    A basic domain-category heuristic, NOT a factual
    credibility score.
    """

    url = url.lower()

    if url.endswith(".gov") or ".gov." in url:
        return "government_domain"

    if url.endswith(".edu") or ".edu." in url:
        return "educational_domain"

    if "doi.org" in url or "pubmed.ncbi.nlm.nih.gov" in url:
        return "research_identifier_or_database"

    return "general_web_domain"
