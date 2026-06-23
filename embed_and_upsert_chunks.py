# You may need to install these packages first:
# pip install openai python-dotenv pinecone

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from chunking import build_embedding_text, load_chunks_from_jsonl


CHUNKS_PATH = Path("data/chunks_all_articles_380w_10ov.jsonl")
EMBEDDING_MODEL = os.getenv("LLMOD_EMBEDDING_MODEL", "NBUECSE-text-embedding-3-small")
EXPECTED_EMBEDDING_DIMENSION = 1536
PINECONE_NAMESPACE = "medium-all-380w-10ov"
BATCH_SIZE = 250


def get_required_env_var(name):
    """Read an environment variable and raise a clear error if it is missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def validate_environment():
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


def connect_to_llmod():
    """Create an OpenAI-compatible client for LLMod.ai."""
    return OpenAI(
        api_key=os.getenv("LLMOD_API_KEY"),
        base_url=os.getenv("LLMOD_BASE_URL"),
    )


def connect_to_pinecone_index():
    """Connect to an existing Pinecone index without creating a new one."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if pinecone_index_host:
        print("Connecting to Pinecone index using PINECONE_INDEX_HOST.")
        return pc.Index(host=pinecone_index_host)

    print("Connecting to Pinecone index using PINECONE_INDEX_NAME.")
    return pc.Index(pinecone_index_name)


def batched(items, batch_size):
    """Yield lists of items in fixed-size batches."""
    for start_index in range(0, len(items), batch_size):
        yield items[start_index : start_index + batch_size]


def metadata_for_chunk(chunk):
    """Build Pinecone metadata for one chunk."""
    return {
        "article_id": chunk["article_id"],
        "chunk_index": chunk["chunk_index"],
        "title": chunk["title"],
        "authors": chunk["authors"],
        "url": chunk["url"],
        "timestamp": chunk["timestamp"],
        "tags": chunk["tags"],
        "chunk": chunk["chunk"],
    }


def embed_texts(client, texts):
    """Embed a batch of texts with one embeddings API call."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, but received {len(embeddings)}."
        )

    for embedding_index, embedding in enumerate(embeddings):
        if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Unexpected embedding dimension for item "
                f"{embedding_index}: expected {EXPECTED_EMBEDDING_DIMENSION}, "
                f"got {len(embedding)}."
            )

    return embeddings


def upsert_batch(index, chunks, embeddings):
    """Upsert one batch of embedded chunks into Pinecone."""
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):
        vectors.append(
            {
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": metadata_for_chunk(chunk),
            }
        )

    response = index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
    return response, len(vectors)


def print_index_stats(index):
    """Print Pinecone index stats after upload, if the call succeeds."""
    try:
        stats = index.describe_index_stats()
    except Exception as error:
        print(f"Could not fetch Pinecone index stats: {error}")
        return

    print("Pinecone index stats:")
    print(stats)


def main():
    """Embed local Medium article chunks and upsert them into Pinecone."""
    # Load variables from the .env file in the project root.
    load_dotenv()
    validate_environment()

    llmod_client = connect_to_llmod()
    pinecone_index = connect_to_pinecone_index()

    # This only reads the prepared small JSONL file. It does not read the CSV
    # and does not re-chunk the articles.
    chunks = load_chunks_from_jsonl(CHUNKS_PATH)
    print(f"Number of chunks loaded: {len(chunks)}")

    if not chunks:
        raise RuntimeError(f"No chunks found in {CHUNKS_PATH}. Nothing to upload.")

    total_uploaded = 0
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_number, chunk_batch in enumerate(batched(chunks, BATCH_SIZE), start=1):
        print(f"Processing batch {batch_number}/{total_batches}.")

        embedding_texts = [build_embedding_text(chunk) for chunk in chunk_batch]
        embeddings = embed_texts(llmod_client, embedding_texts)
        print(f"Embeddings received: {len(embeddings)}")

        _, vectors_upserted = upsert_batch(pinecone_index, chunk_batch, embeddings)
        total_uploaded += vectors_upserted

        print(f"Vectors upserted in this batch: {vectors_upserted}")
        print(f"Total vectors uploaded: {total_uploaded}")

        if batch_number in {1, 5}:
            input(f"Batch {batch_number} completed. Check the output. Press Enter to continue, or Ctrl+C to stop.")

    print(f"Finished uploading {total_uploaded} vectors to namespace '{PINECONE_NAMESPACE}'.")
    print_index_stats(pinecone_index)


if __name__ == "__main__":
    main()
