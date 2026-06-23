"""Local article chunking utilities for the Medium Article RAG assignment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# 512 tokens is approximated as about 380 words for this assignment.
CHUNK_SIZE_WORDS = 380
OVERLAP_RATIO = 0.10

# Small default limit for safe local testing.
MAX_ARTICLES = None

DEFAULT_CSV_PATH = "data/medium-english-50mb-with-ids.csv"
CHUNKS_OUTPUT_PATH = Path("data/chunks_all_articles_380w_10ov.jsonl")


def clean_text(text: str) -> str:
    """Normalize article text by collapsing repeated whitespace."""
    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


def split_text_into_chunks(
    text: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[str]:
    """Split text into overlapping word-based chunks."""
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0.")

    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be at least 0 and less than 1.")

    overlap_words = int(chunk_size_words * overlap_ratio)
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be less than chunk_size_words.")

    words = cleaned_text.split()
    if len(words) <= chunk_size_words:
        return [cleaned_text]

    step_size = chunk_size_words - overlap_words
    chunks = []

    for start_index in range(0, len(words), step_size):
        end_index = start_index + chunk_size_words
        chunk_words = words[start_index:end_index]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if end_index >= len(words):
            break

    return chunks


def _is_missing_value(value: Any) -> bool:
    """Return True for common missing values such as NaN or None."""
    return value is None or value != value


def _safe_metadata_value(row: Any, column_name: str) -> Any:
    """Return a metadata value from a row, or an empty string when unavailable."""
    value = row.get(column_name, "")
    if _is_missing_value(value):
        return ""
    return value


def create_chunks_from_dataframe(
    df: Any,
    max_articles: int | None = MAX_ARTICLES,
) -> list[dict]:
    """Create article chunks with repeated metadata for each chunk."""
    chunks = []

    rows_to_process = df
    if max_articles is not None:
        rows_to_process = df.head(max_articles)

    for _, row in rows_to_process.iterrows():
        article_id = _safe_metadata_value(row, "article_id")
        if article_id == "":
            continue

        text = row.get("text", "")
        article_chunks = split_text_into_chunks(text)
        if not article_chunks:
            continue

        article_id = str(article_id)
        title = _safe_metadata_value(row, "title")
        authors = _safe_metadata_value(row, "authors")
        url = _safe_metadata_value(row, "url")
        timestamp = _safe_metadata_value(row, "timestamp")
        tags = _safe_metadata_value(row, "tags")

        for chunk_index, chunk_text in enumerate(article_chunks):
            chunks.append(
                {
                    "chunk_id": f"{article_id}_{chunk_index}",
                    "article_id": article_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "authors": authors,
                    "url": url,
                    "timestamp": timestamp,
                    "tags": tags,
                    "chunk": chunk_text,
                }
            )

    return chunks


def print_chunking_summary(chunks: list[dict]) -> None:
    """Print a short summary of generated chunks."""
    unique_article_ids = {chunk["article_id"] for chunk in chunks}

    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of unique articles represented: {len(unique_article_ids)}")
    print()

    for chunk in chunks[:3]:
        preview = chunk["chunk"][:300]
        print(f"article_id: {chunk['article_id']}")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"title: {chunk['title']}")
        print(f"chunk preview: {preview}")
        print("-" * 80)


def save_chunks_to_jsonl(
    chunks: list[dict],
    output_path: Path = CHUNKS_OUTPUT_PATH,
) -> None:
    """Save chunks as JSONL, with one complete chunk dictionary per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSONL is useful for embedding later because another script can read
    # chunks one line at a time and process them in batches.
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks_from_jsonl(input_path: Path = CHUNKS_OUTPUT_PATH) -> list[dict]:
    """Load chunks from a JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(f"JSONL chunks file not found: {input_path}")

    chunks = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


def build_embedding_text(chunk: dict) -> str:
    """Build the text that will later be sent to the embedding model."""
    title = str(chunk.get("title", "") or "").strip()
    tags = str(chunk.get("tags", "") or "").strip()
    chunk_text = str(chunk.get("chunk", "") or "").strip()

    return f"Title: {title}\nTags: {tags}\nContent: {chunk_text}".strip()


def preview_embedding_texts(chunks: list[dict], limit: int = 3) -> None:
    """Print a few embedding text examples for local inspection."""
    for chunk in chunks[:limit]:
        print(build_embedding_text(chunk))
        print("-" * 80)


if __name__ == "__main__":
    import pandas as pd

    dataframe = pd.read_csv(DEFAULT_CSV_PATH, encoding="utf-8")
    generated_chunks = create_chunks_from_dataframe(dataframe, max_articles=MAX_ARTICLES)
    print_chunking_summary(generated_chunks)
    save_chunks_to_jsonl(generated_chunks)

    loaded_chunks = load_chunks_from_jsonl()
    same_first_chunk_id = (
        bool(generated_chunks)
        and bool(loaded_chunks)
        and generated_chunks[0]["chunk_id"] == loaded_chunks[0]["chunk_id"]
    )

    print()
    print(f"Chunks created: {len(generated_chunks)}")
    print(f"Chunks loaded: {len(loaded_chunks)}")
    print(f"First chunk_id matches: {same_first_chunk_id}")
    print()
    preview_embedding_texts(loaded_chunks, limit=3)
