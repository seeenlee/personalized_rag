"""Evaluate Brown bear retrieval in Pinecone using mean reciprocal rank."""

import argparse
import os
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.llms.google_genai import GoogleGenAI
from pinecone import Pinecone

REPO_ROOT = Path(__file__).resolve().parents[2]
from ..insert_wikipedia_page import (
    METADATA_TEXT_FIELD,
    TEXT_FIELD_NAME,
    split_paragraphs,
)

PINECONE_ENV_VAR_NAME = "PINECONE_API_KEY"
GEMINI_ENV_VAR_NAME = "GEMINI_API_KEY"
DEFAULT_ARTICLE_PATH = REPO_ROOT / "data" / "wikipedia" / "Brown_bear.txt"
DEFAULT_QUESTION = "when do bears hibernate?"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_TOP_K = 5
GROUND_TRUTH_CHUNK_ID = "Brown_bear-26"


@dataclass
class RetrievedChunk:
    """Normalized retrieval result for reporting and evaluation."""

    chunk_id: str
    text: str
    score: float | None


def load_required_env_vars() -> tuple[str, str]:
    """Load the Pinecone and Gemini API keys from the repo .env or env."""
    load_dotenv(REPO_ROOT / ".env")

    pinecone_api_key = os.getenv(PINECONE_ENV_VAR_NAME)
    if not pinecone_api_key:
        raise RuntimeError(
            f"Missing {PINECONE_ENV_VAR_NAME}. Add it to the repo .env or your "
            "environment."
        )

    gemini_api_key = os.getenv(GEMINI_ENV_VAR_NAME)
    if not gemini_api_key:
        raise RuntimeError(
            f"Missing {GEMINI_ENV_VAR_NAME}. Add it to the repo .env or your "
            "environment."
        )

    return pinecone_api_key, gemini_api_key


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Brown bear retrieval in Pinecone using RAGAS."
    )
    parser.add_argument("index_name", help="Name of the Pinecone index to query")
    parser.add_argument("namespace", help="Namespace in the Pinecone index to query")
    parser.add_argument(
        "--article-path",
        default=str(DEFAULT_ARTICLE_PATH),
        help="Path to the Brown bear source text file",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to ask against the Brown bear article",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of Pinecone chunks to retrieve",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model to use through LlamaIndex",
    )
    return parser.parse_args(argv)


def ensure_file_exists(path: Path) -> Path:
    """Resolve and validate that a file exists."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Article file not found: {resolved}")
    return resolved


def load_llamaindex_components() -> tuple[Any, Any, Any]:
    """Import optional LlamaIndex components with a helpful error."""
    return PromptTemplate, TextNode, GoogleGenAI


def build_brown_bear_nodes(article_path: Path) -> tuple[list[Any], Any]:
    """Create paragraph-level LlamaIndex nodes aligned to the ingestion script."""
    _, TextNode, _ = load_llamaindex_components()

    article_text = article_path.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(article_text)
    if not paragraphs:
        raise RuntimeError(f"No paragraphs found in {article_path}.")

    nodes: list[Any] = []
    for paragraph_number, paragraph in enumerate(paragraphs, start=1):
        chunk_id = f"{article_path.stem}-{paragraph_number}"
        nodes.append(
            TextNode(
                id_=chunk_id,
                text=paragraph,
                metadata={
                    "chunk_id": chunk_id,
                    "source_file": article_path.name,
                    "paragraph_number": paragraph_number,
                },
            )
        )

    ground_truth_node = next(
        (node for node in nodes if node.node_id == GROUND_TRUTH_CHUNK_ID),
        None,
    )
    if ground_truth_node is None:
        raise RuntimeError(
            f"Could not find the expected ground-truth chunk {GROUND_TRUTH_CHUNK_ID} "
            f"in {article_path.name}."
        )

    return nodes, ground_truth_node


def connect_to_index(api_key: str, index_name: str) -> Any:
    """Connect to an existing Pinecone index."""
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        raise RuntimeError(f"Pinecone index '{index_name}' does not exist.")

    index_host = pc.describe_index(name=index_name).host
    return pc.Index(host=index_host)


def search_pinecone(index: Any, namespace: str, question: str, top_k: int) -> Any:
    """Run a text search against an integrated-embedding Pinecone index."""
    query = {
        "inputs": {"text": question},
        "top_k": top_k,
    }
    fields = [TEXT_FIELD_NAME, METADATA_TEXT_FIELD]

    if hasattr(index, "search"):
        return index.search(namespace=namespace, query=query, fields=fields)

    if hasattr(index, "search_records"):
        return index.search_records(namespace=namespace, query=query, fields=fields)

    raise RuntimeError(
        "The installed Pinecone client does not expose a supported search method."
    )


def _safe_get(obj: Any, key: str) -> Any:
    """Read a key from either a mapping or an object attribute."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_sequence(value: Any) -> list[Any]:
    """Normalize a list-like value while ignoring strings."""
    if value is None or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    return []


def extract_hits(search_result: Any) -> list[Any]:
    """Best-effort extraction of Pinecone hits across response shapes."""
    pending = [search_result]
    visited: set[int] = set()

    while pending:
        current = pending.pop(0)
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        hits = _as_sequence(_safe_get(current, "hits"))
        if hits:
            return hits

        matches = _as_sequence(_safe_get(current, "matches"))
        if matches:
            return matches

        for child_key in ("result", "results", "data"):
            child = _safe_get(current, child_key)
            if child is not None:
                pending.append(child)

    return []


def normalize_hits(raw_hits: list[Any]) -> list[RetrievedChunk]:
    """Convert Pinecone hits into a stable structure."""
    normalized: list[RetrievedChunk] = []

    for raw_hit in raw_hits:
        field_sources = [
            _safe_get(raw_hit, "fields"),
            _safe_get(raw_hit, "metadata"),
            raw_hit,
        ]

        text = ""
        for source in field_sources:
            if source is None:
                continue
            text = (
                _safe_get(source, TEXT_FIELD_NAME)
                or _safe_get(source, METADATA_TEXT_FIELD)
                or _safe_get(source, "content")
                or _safe_get(source, "page_content")
                or ""
            )
            if text:
                break

        chunk_id = _safe_get(raw_hit, "_id") or _safe_get(raw_hit, "id") or "unknown"
        score = _safe_get(raw_hit, "_score") or _safe_get(raw_hit, "score")
        normalized.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                text=str(text).strip(),
                score=float(score) if score is not None else None,
            )
        )

    return normalized


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str,
    gemini_api_key: str,
) -> str:
    """Use Gemini through LlamaIndex to answer from retrieved paragraph chunks."""
    PromptTemplate, _, GoogleGenAI = load_llamaindex_components()

    llm = GoogleGenAI(model=model, api_key=gemini_api_key)
    context = "\n\n".join(
        f"[{chunk.chunk_id}] {chunk.text}" for chunk in retrieved_chunks if chunk.text
    )
    prompt = PromptTemplate(
        "You are answering questions about the Brown bear article.\n"
        "Use only the provided paragraph chunks.\n"
        "If the answer is not supported by the chunks, say that clearly.\n\n"
        "Question: {question}\n\n"
        "Paragraph chunks:\n{context}\n\n"
        "Answer in 1-2 sentences and cite the chunk IDs you relied on."
    )
    response = llm.complete(prompt.format(question=question, context=context))
    return str(getattr(response, "text", response)).strip()


def compute_mrr(retrieved_chunks: list[RetrievedChunk]) -> float:
    """Compute reciprocal rank for this single-query evaluation."""
    ground_truth_rank = find_ground_truth_rank(retrieved_chunks)
    if ground_truth_rank is None:
        return 0.0
    return 1.0 / ground_truth_rank


def snippet(text: str, width: int = 120) -> str:
    """Build a compact one-line preview for console output."""
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="...")


def find_ground_truth_rank(retrieved_chunks: list[RetrievedChunk]) -> int | None:
    """Return the 1-based rank of the known ground-truth chunk if retrieved."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.chunk_id == GROUND_TRUTH_CHUNK_ID:
            return rank
    return None


def print_report(
    question: str,
    reference_context: str,
    retrieved_chunks: list[RetrievedChunk],
    response: str,
    mrr: float,
) -> None:
    """Print an inspection-friendly retrieval and evaluation report."""
    ground_truth_rank = find_ground_truth_rank(retrieved_chunks)

    print("Brown Bear Pinecone Evaluation")
    print("==============================")
    print(f"Question: {question}")
    print(f"Ground-truth chunk: {GROUND_TRUTH_CHUNK_ID}")
    print(f"Ground-truth retrieved: {'yes' if ground_truth_rank else 'no'}")
    print(f"Ground-truth rank: {ground_truth_rank if ground_truth_rank else 'not retrieved'}")
    print()
    print("Ground-truth context")
    print("--------------------")
    print(snippet(reference_context, width=160))
    print()
    print("Retrieved chunks")
    print("----------------")
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        score = f"{chunk.score:.4f}" if chunk.score is not None else "n/a"
        marker = " <- ground truth" if chunk.chunk_id == GROUND_TRUTH_CHUNK_ID else ""
        print(f"{rank}. {chunk.chunk_id} | score={score}{marker}")
        print(f"   {snippet(chunk.text, width=160)}")
    print()
    print("Gemini answer")
    print("-------------")
    print(response)
    print()
    print("Retrieval score")
    print("---------------")
    print(f"Mean reciprocal rank (single query): {mrr:.4f}")


def main(argv: list[str] | None = None) -> int:
    """Run the Brown bear Pinecone evaluation workflow."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        article_path = ensure_file_exists(Path(args.article_path))
        pinecone_api_key, gemini_api_key = load_required_env_vars()
        _, ground_truth_node = build_brown_bear_nodes(article_path)
        index = connect_to_index(pinecone_api_key, args.index_name)
        raw_search_result = search_pinecone(
            index=index,
            namespace=args.namespace,
            question=args.question,
            top_k=args.top_k,
        )
        retrieved_chunks = normalize_hits(extract_hits(raw_search_result))
        if not retrieved_chunks:
            raise RuntimeError(
                "Pinecone returned no retrieval hits. Confirm the namespace and "
                "that the Brown bear paragraphs were inserted into the index."
            )

        response = generate_answer(
            args.question,
            retrieved_chunks,
            args.gemini_model,
            gemini_api_key,
        )
        mrr = compute_mrr(retrieved_chunks)
        print_report(
            question=args.question,
            reference_context=ground_truth_node.get_content(),
            retrieved_chunks=retrieved_chunks,
            response=response,
            mrr=mrr,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
