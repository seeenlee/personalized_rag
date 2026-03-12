"""Fetch plain-text article content from a Wikipedia URL."""

import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, unquote
from urllib.request import Request, urlopen


USER_AGENT = "WikipediaExtractScript/1.0 (lee3966@purdue.edu)"
EXPECTED_HOST = "en.wikipedia.org"
EXPECTED_PATH_PREFIX = "/wiki/"
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "wikipedia"


class WikipediaExtractError(Exception):
    """Raised when the script cannot retrieve a Wikipedia extract."""


def parse_article_identifier(article_url: str) -> str:
    """Extract the article identifier from an en.wikipedia.org article URL."""
    parsed_url = urlparse(article_url)

    if parsed_url.scheme not in {"http", "https"}:
        raise WikipediaExtractError("URL must start with http:// or https://.")

    if parsed_url.netloc != EXPECTED_HOST:
        raise WikipediaExtractError(
            f"URL must use the {EXPECTED_HOST} domain."
        )

    if not parsed_url.path.startswith(EXPECTED_PATH_PREFIX):
        raise WikipediaExtractError(
            f"URL path must start with {EXPECTED_PATH_PREFIX}."
        )

    raw_title = parsed_url.path[len(EXPECTED_PATH_PREFIX) :]
    if not raw_title:
        raise WikipediaExtractError("Wikipedia article title is missing from the URL.")

    return unquote(raw_title)


def parse_article_title(article_url: str) -> str:
    """Convert the article URL into the MediaWiki page title."""
    return parse_article_identifier(article_url).replace("_", " ")


def build_output_path(article_url: str) -> Path:
    """Build the output path for the downloaded article text."""
    article_identifier = parse_article_identifier(article_url)
    safe_file_name = article_identifier.replace("/", "_")
    return OUTPUT_DIRECTORY / f"{safe_file_name}.txt"


def fetch_article_text(article_url: str) -> str:
    """Fetch the plain-text article extract using the TextExtracts API."""
    parsed_url = urlparse(article_url)
    article_title = parse_article_title(article_url)
    api_url = f"{parsed_url.scheme}://{parsed_url.netloc}/w/api.php"

    query_string = urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": article_title,
            "explaintext": "1",
            "exsectionformat": "plain",
        }
    )
    request = Request(
        f"{api_url}?{query_string}",
        headers={"User-Agent": USER_AGENT},
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload: dict[str, Any] = json.load(response)
    except HTTPError as exc:
        raise WikipediaExtractError(
            f"Wikipedia API returned HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise WikipediaExtractError(
            f"Could not reach the Wikipedia API: {exc.reason}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise WikipediaExtractError("Wikipedia API returned invalid JSON.") from exc

    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if not isinstance(page, dict):
        raise WikipediaExtractError("Wikipedia API response did not include a page.")

    if "missing" in page:
        raise WikipediaExtractError("The requested Wikipedia article does not exist.")

    article_text = page.get("extract")
    if not article_text:
        raise WikipediaExtractError(
            "The article did not include extract text in the API response."
        )

    return article_text


def main(argv: list[str]) -> int:
    """Run the command-line interface."""
    if len(argv) != 2:
        print(f"Usage: python {argv[0]} <wikipedia-url>", file=sys.stderr)
        return 1

    try:
        article_text = fetch_article_text(argv[1])
        output_path = build_output_path(argv[1])
    except WikipediaExtractError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(article_text, encoding="utf-8")
    except OSError as exc:
        print(f"Error: Could not write article text to {output_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Saved article text to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
