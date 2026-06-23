# You may need to install these packages first:
# pip install openai python-dotenv

import os

from dotenv import load_dotenv
from openai import OpenAI


EMBEDDING_MODEL = os.getenv("LLMOD_EMBEDDING_MODEL", "NBUECSE-text-embedding-3-small")
EXPECTED_DIMENSION = 1536

TEST_INPUT = """Title: Test article
Tags: test, rag
Content: This is a short test to confirm that the embedding API returns a vector."""


def get_required_env_var(name):
    """Read an environment variable and raise a clear error if it is missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main():
    """Run one small embedding request to confirm the LLMod API works."""
    # Load variables from the .env file in the project root.
    load_dotenv()

    # Never print the API key. Only pass it directly to the client.
    api_key = get_required_env_var("LLMOD_API_KEY")
    base_url = get_required_env_var("LLMOD_BASE_URL")

    # LLMod.ai is assumed to be OpenAI-compatible, so we use the OpenAI client.
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Make exactly one embedding API call for this smoke test.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=TEST_INPUT,
    )

    # The embedding vector is stored in the first item returned by the API.
    embedding = response.data[0].embedding
    embedding_length = len(embedding)

    print("Embedding API smoke test succeeded.")
    print(f"Embedding vector length: {embedding_length}")
    print(f"First 5 numbers: {embedding[:5]}")

    if embedding_length == EXPECTED_DIMENSION:
        print("Embedding dimension is 1536 as expected.")
    else:
        print(f"Unexpected embedding dimension: {embedding_length}")


if __name__ == "__main__":
    main()
