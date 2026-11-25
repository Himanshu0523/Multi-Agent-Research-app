
import requests

from bs4 import BeautifulSoup
from urllib.parse import urlsplit
from langchain_core.tools import tool

from provenance import (
    create_source_record,
    serialize_source,
)


# DOCUMENT FETCHER
def fetch_document(url: str) -> dict:

    try:
        # Basic URL validation
        parts = urlsplit(url)

        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
        ):
            raise ValueError("Invalid HTTP/HTTPS URL")

        response = requests.get(
            url,
            timeout=(5, 20),
            headers={
                "User-Agent": "ResearchBot/1.0"
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        # Only process HTML pages
        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" not in content_type:
            raise ValueError(
                f"Unsupported content type: {content_type}"
            )

        soup = BeautifulSoup(
            response.content,
            "lxml",
        )

        # Remove irrelevant elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "noscript",
        ]):
            tag.decompose()

        # Extract title
        title = (
            soup.title.get_text(strip=True)
            if soup.title
            else "Untitled"
        )

        # Extract publication date if available
        published_at = None

        date_tag = soup.find(
            "meta",
            attrs={"property": "article:published_time"},
        )

        if date_tag:
            published_at = date_tag.get("content")

        # Extract main content
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.body
            or soup
        )

        content = main_content.get_text(
            separator=" ",
            strip=True,
        )

        content = content[:15000]

        source = create_source_record(
            url=response.url,
            title=title,
            content=content,
            published_at=published_at,
        )

        source["http_status"] = response.status_code

        return source

    except Exception as e:

        return {
            "url": url,
            "status": "failed",
            "error": str(e),
        }


# LANGCHAIN SCRAPING TOOL
@tool
def scrape_url(url: str) -> str:
    """Fetch a webpage and return its structured source provenance."""

    document = fetch_document(url)

    return serialize_source(document)


# QUICK TEST
if __name__ == "__main__":

    result = fetch_document(
        "https://www.nasa.gov/humans-in-space/artemis/"
    )

    print(serialize_source(result))
