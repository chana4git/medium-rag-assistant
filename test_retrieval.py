# You may need to install these packages first:
# pip install openai python-dotenv pinecone

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


# Edit these constants and rerun this file to test different retrieval queries.
TEST_QUESTION = "how to market a Medium article"
TOP_K = 5
NAMESPACE = "medium-50-test-v2"
EMBEDDING_MODEL = os.getenv("LLMOD_EMBEDDING_MODEL", "NBUECSE-text-embedding-3-small")
EXPECTED_EMBEDDING_DIMENSION = 1536


client = None
index = None


def get_required_env_var(name: str) -> str:
    """Read an environment variable and raise a clear error if it is missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def validate_environment() -> None:
    """Validate the environment variables needed for LLMod.ai and Pinecone."""
    get_required_env_var("LLMOD_API_KEY")
    get_required_env_var("LLMOD_BASE_URL")
    get_required_env_var("PINECONE_API_KEY")

    pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if not pinecone_index_host and not pinecone_index_name:
        raise RuntimeError(
            "Missing Pinecone index target. Set PINECONE_INDEX_HOST or "
            "PINECONE_INDEX_NAME in your .env file."
        )


def connect_to_llmod() -> OpenAI:
    """Create an OpenAI-compatible client for LLMod.ai."""
    return OpenAI(
        api_key=os.getenv("LLMOD_API_KEY"),
        base_url=os.getenv("LLMOD_BASE_URL"),
    )


def connect_to_pinecone_index():
    """Connect to an existing Pinecone index without creating or modifying it."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if pinecone_index_host:
        print("Connecting to Pinecone index using PINECONE_INDEX_HOST.")
        return pc.Index(host=pinecone_index_host)

    print("Connecting to Pinecone index using PINECONE_INDEX_NAME.")
    return pc.Index(pinecone_index_name)


def embed_question(question: str) -> List[float]:
    """Embed the user question with the same model used for the stored chunks."""
    if client is None:
        raise RuntimeError("LLMod.ai client is not initialized.")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    question_embedding = response.data[0].embedding

    if len(question_embedding) != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Unexpected question embedding dimension: expected "
            f"{EXPECTED_EMBEDDING_DIMENSION}, got {len(question_embedding)}."
        )

    return question_embedding


def query_pinecone(question_embedding: List[float], top_k: int = TOP_K):
    """Query Pinecone for similar chunks. This does not upsert or delete anything."""
    if index is None:
        raise RuntimeError("Pinecone index is not initialized.")

    return index.query(
        vector=question_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
    )


def metadata_value(metadata, key: str, default: str = ""):
    """Read metadata from Pinecone's returned metadata object or dict."""
    if metadata is None:
        return default

    if isinstance(metadata, dict):
        return metadata.get(key, default)

    return getattr(metadata, key, default)


def format_list_value(value) -> str:
    """Make list metadata, such as authors or tags, easy to read."""
    if value is None or value == "":
        return ""

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    return str(value)


def print_retrieval_results(question: str, results) -> None:
    """Print retrieved chunks so retrieval quality can be inspected manually."""
    print("\nRetrieval results")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Top K: {TOP_K}")

    matches = getattr(results, "matches", [])

    if not matches:
        print("\nNo matches returned.")
        return

    for rank, match in enumerate(matches, start=1):
        metadata = getattr(match, "metadata", {}) or {}

        match_id = getattr(match, "id", "")
        score = getattr(match, "score", "")
        chunk_id = metadata_value(metadata, "chunk_id", match_id)
        article_id = metadata_value(metadata, "article_id")
        chunk_index = metadata_value(metadata, "chunk_index")
        title = metadata_value(metadata, "title")
        authors = format_list_value(metadata_value(metadata, "authors"))
        tags = format_list_value(metadata_value(metadata, "tags"))
        chunk_text = str(metadata_value(metadata, "chunk", ""))

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"Score: {score}")
        print(f"ID / chunk_id: {match_id} / {chunk_id}")
        print(f"Article ID: {article_id}")
        print(f"Chunk index: {chunk_index}")
        print(f"Title: {title}")
        print(f"Authors: {authors}")
        print(f"Tags: {tags}")
        print("\nChunk text preview:")
        print(chunk_text[:700])


if __name__ == "__main__":
    # This script tests retrieval quality before using a final LLM answer model.
    # It only embeds the question and queries Pinecone. It does not call a chat
    # model, read the full CSV, re-chunk, upsert, or delete vectors.
    print("Starting retrieval-only test.")
    print(f"Test question: {TEST_QUESTION}")

    load_dotenv()
    validate_environment()

    client = connect_to_llmod()
    index = connect_to_pinecone_index()

    embedding = embed_question(TEST_QUESTION)
    retrieval_results = query_pinecone(embedding, top_k=TOP_K)
    print_retrieval_results(TEST_QUESTION, retrieval_results)

    print("\nManually inspect whether the returned chunks are related to the question.")
