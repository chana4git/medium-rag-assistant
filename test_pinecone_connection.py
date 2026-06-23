# You may need to install these packages first:
# pip install pinecone python-dotenv

import os

from dotenv import load_dotenv
from pinecone import Pinecone


VECTOR_ID = "smoke-test-vector"
NAMESPACE = "smoke-test"
VECTOR_DIMENSION = 1536


def main():
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index_host = os.getenv("PINECONE_INDEX_HOST")

    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY in your .env file.")

    if not index_host and not index_name:
        raise ValueError(
            "Missing Pinecone index target. Set PINECONE_INDEX_HOST or PINECONE_INDEX_NAME in your .env file."
        )

    pc = Pinecone(api_key=api_key)

    if index_host:
        index = pc.Index(host=index_host)
        print("Connected to Pinecone index using PINECONE_INDEX_HOST.")
    else:
        index = pc.Index(index_name)
        print("Connected to Pinecone index using PINECONE_INDEX_NAME.")

    test_vector = [0.001] * VECTOR_DIMENSION

    index.upsert(
        vectors=[
            {
                "id": VECTOR_ID,
                "values": test_vector,
                "metadata": {"source": "pinecone_smoke_test"},
            }
        ],
        namespace=NAMESPACE,
    )
    print("Successfully upserted smoke test vector.")

    query_result = index.query(
        vector=test_vector,
        top_k=1,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    if not query_result.matches:
        raise RuntimeError("Query succeeded, but no matches were returned.")

    match = query_result.matches[0]
    print(f"Query match id: {match.id}")
    print(f"Query score: {match.score}")
    print(f"Query metadata: {match.metadata}")

    index.delete(ids=[VECTOR_ID], namespace=NAMESPACE)
    print("Deleted smoke test vector.")

    print("Pinecone smoke test completed successfully.")


if __name__ == "__main__":
    main()
