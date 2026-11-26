
import hashlib
import re

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal


# CLAIM SCHEMA
class ResearchClaim(BaseModel):

    claim: str

    source_id: str

    evidence_quote: str

    status: Literal[
        "supported",
        "contradicted",
        "insufficient",
    ]

    explanation: str


class ClaimExtraction(BaseModel):

    claims: list[ResearchClaim] = Field(
        description="Claims extracted from the research"
    )


def generate_evidence_id(
    source_id: str,
    claim: str,
    quote: str,
) -> str:

    def normalize(text):
        return re.sub(
            r"\s+", " ", text.lower()
        ).strip()

    identity = "|".join([
        source_id,
        normalize(claim),
        normalize(quote),
    ])

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    return f"ev_{digest[:24]}"
    
    
def extract_research_claims(
    topic: str,
    sources: list[dict],
) -> list[ResearchClaim]:

    available_sources = [
        source for source in sources
        if source.get("status") == "fetched"
    ]

    if not available_sources:
        return []

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    structured_llm = llm.with_structured_output(
        ClaimExtraction
    )

    source_context = "\n\n".join(
        f"""
        SOURCE ID: {source['source_id']}
        TITLE: {source['title']}
        URL: {source['url']}
        CONTENT:
        {source['content']}
        """
        for source in available_sources
    )

    prompt = f"""
    You are a research evidence extraction specialist.

    Topic:
    {topic}

    Available source documents:
    {source_context}

    TASK:

    1. Extract important factual claims relevant to the topic.
    2. Associate each claim with an existing source ID.
    3. Provide an exact verbatim evidence passage.
    4. Identify whether the passage supports, contradicts,
       or does not sufficiently address the claim.

    STRICT RULES:

    - Use only the supplied source IDs.
    - Never create source IDs or URLs.
    - Evidence quotes must be copied exactly.
    - Do not invent statistics or publication dates.
    - Do not treat a source mentioning a topic as proof of a claim.
    - If evidence is insufficient, use insufficient status.
    - Keep claims atomic: one factual assertion per claim.
    """

    result = structured_llm.invoke(prompt)

    return result.claims



# STABLE CLAIM ID
def generate_claim_id(claim: str) -> str:

    normalized = re.sub(
        r"\s+",
        " ",
        claim.strip().lower(),
    )

    digest = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

    return f"clm_{digest[:16]}"


# NORMALIZE TEXT
def normalize_text(text: str) -> str:

    return " ".join(text.lower().split())


# VERIFY EXACT PASSAGE
def validate_evidence_quote(
    quote: str,
    source_content: str,
) -> bool:

    if not quote.strip():
        return False

    normalized_quote = normalize_text(quote)

    normalized_content = normalize_text(
        source_content
    )

    return normalized_quote in normalized_content


# BUILD EVIDENCE RECORDS
def build_evidence_records(
    claims: list[ResearchClaim],
    sources: list[dict],
) -> list[dict]:

    source_map = {
        source["source_id"]: source
        for source in sources
        if source.get("status") == "fetched"
    }

    records = []

    for claim in claims:

        source = source_map.get(claim.source_id)

        quote_valid = False

        if source:

            quote_valid = validate_evidence_quote(
                claim.evidence_quote,
                source["content"],
            )

        if (
            claim.status == "supported"
            and not quote_valid
        ):
            final_status = "insufficient"

        elif not source:
            final_status = "insufficient"

        else:
            final_status = claim.status

        evidence_record = {
            "evidence_id": generate_evidence_id(
                source_id=claim.source_id,
                claim=claim.claim,
                quote=claim.evidence_quote,
            ),

            "claim_id": generate_claim_id(
                claim.claim
            ),

            "claim": claim.claim,

            "source_id": claim.source_id,

            "source_url": (
                source["url"] if source else None
            ),

            "evidence_quote": claim.evidence_quote,

            "quote_match": quote_valid,

            "status": final_status,

            "explanation": claim.explanation,
        }

        records.append(evidence_record)

    return records