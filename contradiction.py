
import hashlib
import json
import os
import re

from itertools import combinations
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Only comparisons at or above this confidence are considered
# high-confidence contradiction candidates.
CONTRADICTION_CONFIDENCE_THRESHOLD = 0.75


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class NormalizedClaim(BaseModel):
    """A normalized representation of a factual research claim."""

    entity: str = Field(
        description="The main entity being discussed."
    )

    attribute: str = Field(
        description="The property or characteristic being measured."
    )

    value: str = Field(
        description="The factual value or assertion."
    )

    unit: str | None = Field(
        default=None,
        description="Measurement unit, if applicable."
    )

    time_scope: str | None = Field(
        default=None,
        description="Relevant date, period, or time range."
    )

    geography: str | None = Field(
        default=None,
        description="Geographic scope of the claim."
    )

    population: str | None = Field(
        default=None,
        description="Population, sample, or group covered."
    )

    conditions: str | None = Field(
        default=None,
        description="Conditions, methodology, or qualifications."
    )

    normalized_statement: str = Field(
        description="A concise normalized version of the original claim."
    )


class ClaimComparison(BaseModel):
    """LLM assessment of two claims."""

    relationship: Literal[
        "contradiction",
        "contextual_difference",
        "compatible",
        "not_comparable",
        "uncertain",
    ]

    explanation: str = Field(
        description="Why the two claims have this relationship."
    )

    contextual_factors: list[str] = Field(
        default_factory=list,
        description=(
            "Relevant differences in time, geography, population, "
            "methodology, definitions, or conditions."
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the comparison, from 0 to 1."
    )

    resolution: str = Field(
        description=(
            "What additional evidence or context is needed, "
            "or why the claims can be reconciled."
        )
    )


# ============================================================
# HELPERS
# ============================================================

def create_claim_id(claim: str) -> str:
    """Generate a deterministic ID for a claim."""

    digest = hashlib.sha256(
        claim.strip().lower().encode("utf-8")
    ).hexdigest()

    return f"claim_{digest[:16]}"


def safe_json(data) -> str:
    """Serialize data safely for LLM prompts."""

    return json.dumps(
        data,
        ensure_ascii=False,
        default=str,
    )


def clean_text(value) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def get_claim_text(record: dict) -> str:
    """Extract claim text from an evidence record."""

    return clean_text(
        record.get("claim")
        or record.get("statement")
        or record.get("text")
    )


def get_source_ids(record: dict) -> list[str]:
    """Extract source identifiers from common evidence structures."""

    source_ids = []

    direct_id = record.get("source_id")

    if direct_id:
        source_ids.append(str(direct_id))

    evidence = record.get("evidence", [])

    if isinstance(evidence, dict):
        evidence = [evidence]

    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                source_id = item.get("source_id")

                if source_id:
                    source_ids.append(str(source_id))

    return list(dict.fromkeys(source_ids))


def normalize_entity(value: str) -> str:
    """Basic deterministic normalization for candidate filtering."""

    return clean_text(value).lower()


def normalize_attribute(value: str) -> str:
    return clean_text(value).lower()


# ============================================================
# NODE 1: NORMALIZE CLAIMS
# ============================================================

def normalize_claims(
    evidence_records: list[dict],
    llm=None,
) -> list[dict]:
    """
    Convert evidence records into normalized factual claims.

    Normalization makes it easier to compare claims referring
    to the same entity and attribute.
    """

    if llm is None:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
        )

    normalizer = llm.with_structured_output(
        NormalizedClaim
    )

    normalized_claims = []

    for record in evidence_records:

        claim_text = get_claim_text(record)

        if not claim_text:
            continue

        claim_id = create_claim_id(claim_text)

        print(
            f"[CONTRADICTION] Normalizing {claim_id}"
        )

        try:
            prompt = f"""
You are a factual claim normalization engine.

Convert the following research claim into structured fields.

Rules:
1. Preserve the original factual meaning.
2. Do not introduce facts that are not present.
3. Identify the entity and attribute.
4. Preserve units, dates, geography and population scope.
5. If a field is not available, return null.
6. Do not interpret uncertainty as certainty.
7. Do not change the claim's numerical value.

Original claim:
{claim_text}

Evidence record:
{safe_json(record)}
"""

            result = normalizer.invoke(prompt)

            normalized_claims.append({
                "claim_id": claim_id,
                "original_claim": claim_text,
                "entity": result.entity,
                "attribute": result.attribute,
                "value": result.value,
                "unit": result.unit,
                "time_scope": result.time_scope,
                "geography": result.geography,
                "population": result.population,
                "conditions": result.conditions,
                "normalized_statement": (
                    result.normalized_statement
                ),
                "source_ids": get_source_ids(record),
                "evidence_status": record.get("status"),
                "normalization_status": "success",
            })

        except Exception as error:

            print(
                f"[NORMALIZATION ERROR] {claim_id}: {error}"
            )

            normalized_claims.append({
                "claim_id": claim_id,
                "original_claim": claim_text,
                "source_ids": get_source_ids(record),
                "evidence_status": record.get("status"),
                "normalization_status": "failed",
                "error": str(error),
            })

    return normalized_claims


# ============================================================
# NODE 2: GENERATE COMPARISON CANDIDATES
# ============================================================

def generate_candidate_pairs(
    normalized_claims: list[dict],
) -> list[tuple[dict, dict]]:
    """
    Select pairs that discuss the same entity and attribute.

    This is a preliminary filter, not proof of comparability.
    The LLM performs the contextual comparison afterward.
    """

    valid_claims = [
        claim
        for claim in normalized_claims
        if claim.get("normalization_status") == "success"
        and claim.get("entity")
        and claim.get("attribute")
    ]

    candidates = []

    for first, second in combinations(valid_claims, 2):

        same_entity = (
            normalize_entity(first["entity"])
            == normalize_entity(second["entity"])
        )

        same_attribute = (
            normalize_attribute(first["attribute"])
            == normalize_attribute(second["attribute"])
        )

        if same_entity and same_attribute:

            candidates.append((first, second))

    return candidates


# ============================================================
# NODE 3: COMPARE CLAIM PAIRS
# ============================================================

def compare_claim_pair(
    first: dict,
    second: dict,
    llm,
) -> dict:
    """Determine whether two claims genuinely conflict."""

    comparator = llm.with_structured_output(
        ClaimComparison
    )

    prompt = f"""
You are an expert research contradiction detection system.

Compare two factual claims.

Your task is NOT to assume that different values automatically
mean a contradiction.

Check these factors carefully:

1. Entity identity
2. Attribute definition
3. Time period
4. Geographic scope
5. Population or sample
6. Measurement unit
7. Research methodology
8. Experimental conditions
9. Definitions and terminology

Classification:

- contradiction:
  Both claims refer to comparable conditions but assert
  incompatible factual values.

- contextual_difference:
  The claims differ because their time, location, population,
  methodology or other context differs.

- compatible:
  Both claims can be true simultaneously.

- not_comparable:
  The claims do not describe sufficiently comparable facts.

- uncertain:
  Available information is insufficient to determine
  the relationship.

Do not invent missing context.

CLAIM A:
{safe_json(first)}

CLAIM B:
{safe_json(second)}

Return a structured comparison.
"""

    result = comparator.invoke(prompt)

    return {
        "claim_a_id": first["claim_id"],
        "claim_b_id": second["claim_id"],

        "claim_a": first["original_claim"],
        "claim_b": second["original_claim"],

        "claim_a_sources": first.get("source_ids", []),
        "claim_b_sources": second.get("source_ids", []),

        "relationship": result.relationship,
        "explanation": result.explanation,
        "contextual_factors": result.contextual_factors,
        "confidence": result.confidence,
        "resolution": result.resolution,
    }


# ============================================================
# NODE 4: DETECT CONTRADICTIONS
# ============================================================

def detect_contradictions(
    evidence_records: list[dict],
    source_relationships: list[dict] | None = None,
) -> dict:
    """
    Main contradiction detection entry point.

    Returns:
        {
            "normalized_claims": [...],
            "claim_comparisons": [...]
        }
    """

    print("\n[CONTRADICTION] Starting detection")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
    )

    normalized = normalize_claims(
        evidence_records=evidence_records,
        llm=llm,
    )

    candidates = generate_candidate_pairs(normalized)

    print(
        f"[CONTRADICTION] "
        f"{len(normalized)} claims normalized"
    )

    print(
        f"[CONTRADICTION] "
        f"{len(candidates)} candidate pairs generated"
    )

    comparisons = []

    for first, second in candidates:

        try:
            comparison = compare_claim_pair(
                first=first,
                second=second,
                llm=llm,
            )

            comparisons.append(comparison)

            print(
                f"[COMPARISON] "
                f"{comparison['claim_a_id']} vs "
                f"{comparison['claim_b_id']}: "
                f"{comparison['relationship']}"
            )

        except Exception as error:

            print(
                f"[COMPARISON ERROR] "
                f"{first['claim_id']} / "
                f"{second['claim_id']}: {error}"
            )

            comparisons.append({
                "claim_a_id": first["claim_id"],
                "claim_b_id": second["claim_id"],
                "claim_a": first["original_claim"],
                "claim_b": second["original_claim"],
                "claim_a_sources": first.get("source_ids", []),
                "claim_b_sources": second.get("source_ids", []),
                "relationship": "uncertain",
                "explanation": (
                    "Automated comparison failed. "
                    "Manual review is required."
                ),
                "contextual_factors": [],
                "confidence": 0.0,
                "resolution": str(error),
            })

    return {
        "normalized_claims": normalized,
        "claim_comparisons": comparisons,
    }


# ============================================================
# SOURCE RELATIONSHIP HELPERS
# ============================================================

def find_shared_source_relationships(
    source_ids: list[str],
    source_relationships: list[dict],
) -> list[dict]:
    """
    Find known source relationships involving the sources
    supporting a comparison.

    Supports relationship records containing source_a/source_b
    or source_id_a/source_id_b.
    """

    relevant = []
    source_set = set(source_ids)

    for relationship in source_relationships:

        first = (
            relationship.get("source_a")
            or relationship.get("source_id_a")
        )

        second = (
            relationship.get("source_b")
            or relationship.get("source_id_b")
        )

        if first in source_set and second in source_set:
            relevant.append(relationship)

        elif second in source_set and first in source_set:
            relevant.append(relationship)

    return relevant


# ============================================================
# NODE 5: BUILD CONFLICT REPORT
# ============================================================

def build_conflict_report(
    claim_comparisons: list[dict],
    source_relationships: list[dict] | None = None,
) -> dict:
    """
    Build a structured contradiction report.

    A contradiction is not automatically resolved just because
    an LLM proposes a possible explanation.
    """

    source_relationships = source_relationships or []

    contradictions = []
    contextual_differences = []
    compatible = []
    uncertain = []
    not_comparable = []

    for comparison in claim_comparisons:

        relationship = comparison.get("relationship")

        source_ids = (
            comparison.get("claim_a_sources", [])
            + comparison.get("claim_b_sources", [])
        )

        related_sources = find_shared_source_relationships(
            source_ids,
            source_relationships,
        )

        item = {
            **comparison,
            "source_relationships": related_sources,
        }

        if relationship == "contradiction":

            confidence = comparison.get("confidence", 0.0)

            if confidence >= CONTRADICTION_CONFIDENCE_THRESHOLD:
                contradictions.append(item)
            else:
                uncertain.append(item)

        elif relationship == "contextual_difference":
            contextual_differences.append(item)

        elif relationship == "compatible":
            compatible.append(item)

        elif relationship == "not_comparable":
            not_comparable.append(item)

        else:
            uncertain.append(item)

    unresolved = contradictions + uncertain

    return {
        "summary": {
            "total_comparisons": len(claim_comparisons),
            "contradictions": len(contradictions),
            "contextual_differences": len(contextual_differences),
            "compatible": len(compatible),
            "not_comparable": len(not_comparable),
            "uncertain": len(uncertain),
            "unresolved": len(unresolved),
        },

        "contradictions": contradictions,

        "contextual_differences": contextual_differences,

        "compatible": compatible,

        "not_comparable": not_comparable,

        "uncertain": uncertain,

        "unresolved_conflicts": unresolved,

        "resolution_status": (
            "requires_review"
            if unresolved
            else "no_conflicts_detected"
        ),
    }


# ============================================================
# OPTIONAL: HUMAN-READABLE REPORT
# ============================================================

def format_conflict_report(report: dict) -> str:
    """Convert the structured report into readable text."""

    summary = report.get("summary", {})

    lines = [
        "CONTRADICTION ANALYSIS",
        "=" * 60,
        "",
        "SUMMARY",
        safe_json(summary),
        "",
    ]

    sections = [
        ("CONFIRMED CONTRADICTION CANDIDATES", "contradictions"),
        ("CONTEXTUAL DIFFERENCES", "contextual_differences"),
        ("COMPATIBLE CLAIMS", "compatible"),
        ("NOT COMPARABLE", "not_comparable"),
        ("UNCERTAIN COMPARISONS", "uncertain"),
    ]

    for heading, key in sections:

        lines.extend([
            heading,
            "-" * 60,
        ])

        items = report.get(key, [])

        if not items:
            lines.append("None")
        else:
            for item in items:
                lines.extend([
                    f"Claim A: {item.get('claim_a')}",
                    f"Claim B: {item.get('claim_b')}",
                    f"Relationship: {item.get('relationship')}",
                    f"Confidence: {item.get('confidence')}",
                    f"Explanation: {item.get('explanation')}",
                    f"Resolution: {item.get('resolution')}",
                    "",
                ])

        lines.append("")

    return "\n".join(lines)
