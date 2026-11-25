import hashlib
import json

from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode



# URL Normalizations
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}


def normalize_url(url : str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.schema.lower()
    hostname = (parts.hostname or "").lower()
    
    if scheme not in {"http" , "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are allowed")
    
    if not hostname or parts.username or parts.password:
        raise ValueError('Invalid URL')
    
    port = parts.port
    
    netloc = hostname
    
    if port and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
        
    query = [
        (key , value) 
        for key , value in parse_qsl(
            parts.query,
            keep_blank_values=True
        )
        if key.lower() not in TRACKING_PARAMS
    ]
    
    query_string = urlencode(query)

    return urlunsplit((
        scheme,
        netloc,
        parts.path or "/",
        query_string,
        ""
    ))    
    
    
def generate_source_id(url: str) -> str:
    
    canonical_url = normalize_url(url)
    
    digest = hashlib.sha256(
        canonical_url.encode("utf-8")
    ).hexdigest()
    
    return f"src_{digest[:16]}"



def generate_content_hash(content: str) -> str:
    
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    
    
def create_source_record(
    url: str,
    title: str,
    content: str,
    published_at: str | None = None,
):

    canonical_url = normalize_url(url)

    return {
        "source_id": generate_source_id(canonical_url),

        "url": canonical_url,

        "domain": urlsplit(canonical_url).hostname,

        "title": title,

        "published_at": published_at,

        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "content_hash": generate_content_hash(content),

        "content": content,

        "status": "fetched",
    }
    

def serialize_source(source: dict) -> str:
    return json.dumps(
        source,
        ensure_ascii=False,
        indent=2
    )