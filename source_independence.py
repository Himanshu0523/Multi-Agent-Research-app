import re
from urllib.parse import urlparse


WIRE_SERVICES = (
    "reuters",
    "associated press",
    "ap news",
    "afp",
    "agence france-presse",
    "bloomberg",
    "pr newswire",
    "business wire",
    "globe newswire",
)


def _extract_domain(url: str) -> str:
    """Extract normalized root domain without www. prefix."""
    if not url:
        return ""
    return urlparse(url).netloc.lower().removeprefix("www.")


def _tokenize(text: str) -> set[str]:
    """Tokenize text into normalized words for Jaccard similarity checks."""
    return set(re.findall(r"\b[a-z0-9]{4,}\b", (text or "").lower()))


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def _detect_wire_attribution(content: str) -> str | None:
    """Detect whether an article credits a wire service or press release feed."""
    header = (content or "")[:600].lower()
    for wire in WIRE_SERVICES:
        if wire in header:
            return wire
    return None


def analyze_source_independence(sources: list[dict]) -> list[dict]:
    """
    Evaluate pairwise independence across fetched sources to identify:
    - Same-domain articles (non-independent publisher)
    - Syndicated wire-service reprints
    - Near-duplicate or mirrored content
    """
    relationships: list[dict] = []

    prepared = []
    for index, source in enumerate(sources):
        source_id = source.get("source_id") or f"S{index + 1:03d}"
        url = source.get("url", "")
        content = source.get("content", "") or ""
        prepared.append({
            "source_id": source_id,
            "url": url,
            "domain": _extract_domain(url),
            "tokens": _tokenize(content),
            "wire": _detect_wire_attribution(content),
        })

    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            a = prepared[i]
            b = prepared[j]

            similarity = _jaccard_similarity(a["tokens"], b["tokens"])
            same_domain = bool(a["domain"]) and a["domain"] == b["domain"]
            shared_wire = (
                a["wire"]
                if a["wire"] and a["wire"] == b["wire"]
                else None
            )

            if similarity >= 0.65:
                relation_type = "syndicated_or_mirror"
                is_independent = False
                reason = (
                    f"High content overlap ({similarity:.0%}) between "
                    f"{a['source_id']} and {b['source_id']}."
                )
            elif shared_wire:
                relation_type = "shared_wire_source"
                is_independent = False
                reason = (
                    f"Both {a['source_id']} and {b['source_id']} "
                    f"reference wire/release source '{shared_wire}'."
                )
            elif same_domain:
                relation_type = "same_domain"
                is_independent = False
                reason = (
                    f"Both {a['source_id']} and {b['source_id']} "
                    f"originate from domain '{a['domain']}'."
                )
            else:
                relation_type = "independent"
                is_independent = True
                reason = (
                    f"Distinct domains ({a['domain']} vs {b['domain']}) "
                    f"with low content overlap ({similarity:.0%})."
                )

            relationships.append({
                "source_a": a["source_id"],
                "source_b": b["source_id"],
                "domain_a": a["domain"],
                "domain_b": b["domain"],
                "relationship": relation_type,
                "is_independent": is_independent,
                "similarity_score": round(similarity, 3),
                "reason": reason,
            })

    return relationships