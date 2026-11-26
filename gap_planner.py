
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# STRUCTURED OUTPUT SCHEMAS
class ResearchQuestion(BaseModel):
    question_id: str = Field(
        description="Unique ID such as Q1, Q2, Q3"
    )

    question: str = Field(
        description="A specific, answerable research question"
    )

    importance: Literal["high", "medium", "low"]

    research_angle: str = Field(
        description="What evidence or perspective is needed"
    )


class ResearchPlan(BaseModel):
    objective: str

    questions: list[ResearchQuestion]


class QuestionAssessment(BaseModel):
    question_id: str

    status: Literal[
        "answered",
        "partially_answered",
        "unanswered",
    ]

    evidence_ids: list[str] = Field(default_factory=list)

    missing_information: str = ""


class CoverageAssessment(BaseModel):
    assessments: list[QuestionAssessment]


# LLM
def get_planning_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    return llm


# NODE SUPPORT: CREATE RESEARCH PLAN
def create_research_plan(
    topic: str,
    max_questions: int = 8,
    config=None,
) -> dict:
    llm = get_planning_llm().with_structured_output(
        ResearchPlan
    )

    prompt = f"""
    You are an expert research planning system.

    Create a research plan for this topic:
    {topic}

    Requirements:
    1. Create no more than {max_questions} questions.
    2. Questions must be specific, answerable and non-duplicative.
    3. Cover technical foundations and mechanisms.
    4. Cover recent developments and evidence.
    5. Cover practical applications and industry adoption.
    6. Cover limitations, risks and unresolved problems.
    7. Include comparisons or benchmarks when relevant.
    8. Prioritize important questions as high, medium or low.
    9. Do not assume facts that have not been researched.

    Return a structured research plan.
    """

    result = llm.invoke(prompt, config=config)

    questions = []
    for index, item in enumerate(result.questions, start=1):
        questions.append({
            "question_id": f"Q{index}",
            "question": item.question,
            "importance": item.importance,
            "research_angle": item.research_angle,
            "status": "unanswered",
            "evidence_ids": [],
            "missing_information": "",
        })

    return {
        "research_objective": result.objective,
        "research_questions": questions,
    }
    
    
    
# EVIDENCE ID NORMALIZATION
def normalize_evidence_records(
    evidence_records: list[dict],
) -> list[dict]:

    normalized = []

    for index, record in enumerate(evidence_records, start=1):
        item = dict(record)

        # Keep an existing ID where available.
        evidence_id = item.get("evidence_id")

        if not evidence_id:
            evidence_id = f"E{index:03d}"

        item["evidence_id"] = evidence_id

        normalized.append(item)

    return normalized


# NODE SUPPORT: ASSESS QUESTION COVERAGE
def assess_question_coverage(
    research_questions: list[dict],
    evidence_records: list[dict],
) -> dict:

    if not research_questions:
        return {
            "assessments": [],
            "answered": 0,
            "partially_answered": 0,
            "unanswered": 0,
            "coverage": 0.0,
            "high_priority_unanswered": [],
        }

    evidence = normalize_evidence_records(
        evidence_records
    )

    valid_evidence_ids = {
        item["evidence_id"]
        for item in evidence
    }

    # Only provide evidence records, not unrestricted instructions
    # from source documents. Source content is untrusted data.
    llm = get_planning_llm().with_structured_output(
        CoverageAssessment
    )

    prompt = f"""
You are a research coverage evaluator.

Assess whether each research question is answered by the
provided evidence records.

RESEARCH QUESTIONS:
{research_questions}

EVIDENCE RECORDS:
{evidence}

Rules:

1. Use only the supplied evidence records.
2. Do not use outside knowledge.
3. A question is answered only when evidence sufficiently
   addresses its central requirements.
4. Use partially_answered when evidence addresses only
   part of the question.
5. Use unanswered when evidence does not adequately answer it.
6. Cite only evidence_id values from the supplied records.
7. Do not treat an evidence record marked insufficient or
   contradicted as supporting evidence.
8. Explain missing information briefly.
9. Return exactly one assessment for every question.
"""

    result = llm.invoke(prompt)

    returned = {
        item.question_id: item
        for item in result.assessments
    }

    assessments = []

    for question in research_questions:
        question_id = question["question_id"]

        item = returned.get(question_id)

        if item is None:
            assessment = {
                **question,
                "status": "unanswered",
                "evidence_ids": [],
                "missing_information": (
                    "No assessment was returned for this question."
                ),
            }

        else:
            valid_ids = [
                evidence_id
                for evidence_id in item.evidence_ids
                if evidence_id in valid_evidence_ids
            ]

            status = item.status

            # Never accept an answered status without at least
            # one valid evidence reference.
            if status == "answered" and not valid_ids:
                status = "unanswered"

            assessment = {
                **question,
                "status": status,
                "evidence_ids": valid_ids,
                "missing_information": item.missing_information,
            }

        assessments.append(assessment)

    answered = sum(
        item["status"] == "answered"
        for item in assessments
    )

    partially_answered = sum(
        item["status"] == "partially_answered"
        for item in assessments
    )

    unanswered = sum(
        item["status"] == "unanswered"
        for item in assessments
    )

    # Partial answers count as half coverage.
    total = len(assessments)

    coverage = (
        (answered + 0.5 * partially_answered) / total
        if total
        else 0.0
    )

    high_priority_unanswered = [
        item
        for item in assessments
        if item["importance"] == "high"
        and item["status"] != "answered"
    ]

    return {
        "assessments": assessments,
        "answered": answered,
        "partially_answered": partially_answered,
        "unanswered": unanswered,
        "coverage": round(coverage, 4),
        "high_priority_unanswered": high_priority_unanswered,
    }


# BUILD TARGETED RETRY FEEDBACK
def build_gap_feedback(
    assessments: list[dict],
) -> str:

    gaps = [
        item
        for item in assessments
        if item["status"] != "answered"
    ]

    if not gaps:
        return "No unanswered research questions remain."

    lines = [
        "Targeted research gaps:",
        "",
    ]

    for item in gaps:
        lines.append(
            f"- Question: {item['question']}"
        )

        lines.append(
            f"  Priority: {item['importance']}"
        )

        lines.append(
            f"  Status: {item['status']}"
        )

        lines.append(
            f"  Missing: {item.get('missing_information', '')}"
        )

        lines.append(
            f"  Research angle: {item.get('research_angle', '')}"
        )

        lines.append("")

    lines.append(
        "Find additional primary or independent sources "
        "that directly address these gaps. Do not repeat "
        "already established findings without new evidence."
    )

    return "\n".join(lines)
