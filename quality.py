
from datetime import datetime, timezone
from urllib.parse import urlsplit


# HELPERS
def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def get_domain(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except (ValueError, TypeError):
        return None


def parse_date(value: str | None):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed

    except (ValueError, TypeError):
        return None


def is_primary_source(metadata: dict) -> bool:
    """
    Uses explicit source classification fields.
    Unknown classifications are not treated as primary.
    """

    values = [
        metadata.get("source_type"),
        metadata.get("publisher_type"),
        metadata.get("source_class"),
    ]

    normalized = {
        str(value).strip().lower()
        for value in values
        if value
    }

    primary_labels = {
        "primary",
        "primary_source",
        "official",
        "research_paper",
        "government",
        "academic",
        "original_research",
    }

    return bool(normalized & primary_labels)


# METRIC 1: EVIDENCE COVERAGE
def calculate_evidence_coverage(evidence_records: list[dict]) -> dict:
    total = len(evidence_records)

    supported = sum(
        1
        for record in evidence_records
        if record.get("status") == "supported"
    )

    insufficient = sum(
        1
        for record in evidence_records
        if record.get("status") == "insufficient"
    )

    contradicted = sum(
        1
        for record in evidence_records
        if record.get("status") == "contradicted"
    )

    return {
        "score": safe_ratio(supported, total),
        "supported": supported,
        "insufficient": insufficient,
        "contradicted": contradicted,
        "total": total,
    }


# METRIC 2: SOURCE DIVERSITY
def calculate_source_diversity(sources: list[dict]) -> dict:
    domains = {
        domain
        for source in sources
        if (domain := get_domain(source.get("url", "")))
    }

    total_sources = len(sources)
    unique_domains = len(domains)

    return {
        "score": safe_ratio(unique_domains, total_sources),
        "unique_domains": unique_domains,
        "total_sources": total_sources,
        "domains": sorted(domains),
    }


# METRIC 3: PRIMARY SOURCE RATIO
def calculate_primary_source_ratio(
    sources: list[dict],
    source_intelligence: list[dict],
) -> dict:

    metadata_by_id = {
        item.get("source_id"): item
        for item in source_intelligence
        if item.get("source_id")
    }

    classified = 0
    primary = 0

    for source in sources:
        metadata = metadata_by_id.get(
            source.get("source_id"),
            {},
        )

        if not any(
            metadata.get(key)
            for key in (
                "source_type",
                "publisher_type",
                "source_class",
            )
        ):
            continue

        classified += 1

        if is_primary_source(metadata):
            primary += 1

    return {
        "score": safe_ratio(primary, classified),
        "primary_sources": primary,
        "classified_sources": classified,
        "total_sources": len(sources),
    }


# METRIC 4: CITATION INTEGRITY
def calculate_citation_integrity(
    sources: list[dict],
    evidence_records: list[dict],
) -> dict:

    valid_source_ids = {
        source.get("source_id")
        for source in sources
        if source.get("source_id")
    }

    total = len(evidence_records)
    valid = 0
    invalid = 0
    missing = 0

    for record in evidence_records:

        # Adapt these field names if your evidence.py
        # uses a different citation structure.
        source_ids = record.get("source_ids")

        if source_ids is None:
            source_id = record.get("source_id")
            source_ids = [source_id] if source_id else []

        if isinstance(source_ids, str):
            source_ids = [source_ids]

        if not source_ids:
            missing += 1
            continue

        if all(
            source_id in valid_source_ids
            for source_id in source_ids
        ):
            valid += 1
        else:
            invalid += 1

    return {
        "score": safe_ratio(valid, total),
        "valid": valid,
        "invalid": invalid,
        "missing": missing,
        "total": total,
    }


# METRIC 5: CONTRADICTION RATE
def calculate_contradiction_rate(
    evidence_records: list[dict],
) -> dict:

    total = len(evidence_records)

    contradicted = sum(
        1
        for record in evidence_records
        if record.get("status") == "contradicted"
    )

    return {
        "score": safe_ratio(contradicted, total),
        "contradicted": contradicted,
        "total": total,
    }


# METRIC 6: RESEARCH COMPLETENESS
def calculate_research_completeness(
    research_gaps: list[str],
    unresolved_conflicts: list[str],
) -> dict:

    total_gaps = len(research_gaps)
    unresolved = len(unresolved_conflicts)

    # A gap list represents known missing work.
    # Without a separate resolved-gap record, we cannot
    # reliably calculate how many gaps were addressed.
    if total_gaps == 0:
        return {
            "score": None,
            "status": "not_measured",
            "known_gaps": 0,
            "unresolved_conflicts": unresolved,
        }

    return {
        "score": 0.0,
        "status": "incomplete",
        "known_gaps": total_gaps,
        "unresolved_conflicts": unresolved,
        "note": (
            "No resolved-gap tracking is available yet. "
            "A completeness score cannot be calculated "
            "from an unresolved-gap list alone."
        ),
    }


# METRIC 7: SOURCE FRESHNESS
def calculate_source_freshness(
    sources: list[dict],
    freshness_days: int = 365,
) -> dict:

    now = datetime.now(timezone.utc)

    dated = 0
    fresh = 0
    undated = 0

    for source in sources:
        published_at = parse_date(
            source.get("published_at")
        )

        if published_at is None:
            undated += 1
            continue

        dated += 1

        age_days = (now - published_at).days

        if 0 <= age_days <= freshness_days:
            fresh += 1

    return {
        "score": safe_ratio(fresh, dated),
        "fresh_sources": fresh,
        "dated_sources": dated,
        "undated_sources": undated,
        "total_sources": len(sources),
        "freshness_window_days": freshness_days,
    }


# OVERALL QUALITY EVALUATION
def evaluate_research_quality(
    sources: list[dict],
    source_intelligence: list[dict],
    evidence_records: list[dict],
    research_gaps: list[str] | None = None,
    unresolved_conflicts: list[str] | None = None,
    freshness_days: int = 365,
) -> dict:

    research_gaps = research_gaps or []
    unresolved_conflicts = unresolved_conflicts or []

    metrics = {
        "evidence_coverage": calculate_evidence_coverage(
            evidence_records
        ),

        "source_diversity": calculate_source_diversity(
            sources
        ),

        "primary_source_ratio": calculate_primary_source_ratio(
            sources,
            source_intelligence,
        ),

        "citation_integrity": calculate_citation_integrity(
            sources,
            evidence_records,
        ),

        "contradiction_rate": calculate_contradiction_rate(
            evidence_records
        ),

        "research_completeness": calculate_research_completeness(
            research_gaps,
            unresolved_conflicts,
        ),

        "source_freshness": calculate_source_freshness(
            sources,
            freshness_days,
        ),
    }

    # Only metrics with a numeric score participate.
    # Contradiction rate is inverted: fewer contradictions
    # produce a higher quality contribution.
    weighted_metrics = {
        "evidence_coverage": (
            metrics["evidence_coverage"]["score"],
            0.30,
        ),
        "source_diversity": (
            metrics["source_diversity"]["score"],
            0.15,
        ),
        "primary_source_ratio": (
            metrics["primary_source_ratio"]["score"],
            0.15,
        ),
        "citation_integrity": (
            metrics["citation_integrity"]["score"],
            0.20,
        ),
        "contradiction_rate": (
            1.0 - metrics["contradiction_rate"]["score"],
            0.10,
        ),
        "source_freshness": (
            metrics["source_freshness"]["score"],
            0.10,
        ),
    }

    available = {
        name: (score, weight)
        for name, (score, weight) in weighted_metrics.items()
        if score is not None
    }

    total_weight = sum(
        weight
        for _, weight in available.values()
    )

    overall_score = (
        sum(
            score * weight
            for score, weight in available.values()
        ) / total_weight
        if total_weight
        else 0.0
    )

    # A score alone must not hide critical failures.
    critical_failures = []

    if metrics["citation_integrity"]["invalid"] > 0:
        critical_failures.append("invalid_source_references")

    if metrics["evidence_coverage"]["total"] == 0:
        critical_failures.append("no_evidence_records")

    if metrics["source_diversity"]["total_sources"] == 0:
        critical_failures.append("no_sources")

    if critical_failures:
        status = "needs_review"
    elif overall_score >= 0.80:
        status = "high"
    elif overall_score >= 0.60:
        status = "moderate"
    else:
        status = "low"

    return {
        "overall_score": round(overall_score, 4),
        "overall_percentage": round(overall_score * 100, 2),
        "status": status,
        "metrics": metrics,
        "critical_failures": critical_failures,
        "scoring_weights": {
            name: weight
            for name, (_, weight) in weighted_metrics.items()
        },
        "limitations": [
            "This is a heuristic score, not a factual correctness guarantee.",
            "Source diversity by domain does not prove editorial independence.",
            "Primary-source classification depends on source intelligence metadata.",
            "Freshness uses publication dates, not the date relevance of each claim.",
            "Research completeness is not scored without resolved-gap tracking.",
        ],
    }
