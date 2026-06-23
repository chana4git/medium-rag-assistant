import os
from typing import Dict, List, Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone


load_dotenv()

# ----------------------------
# Configuration
# ----------------------------

EMBEDDING_MODEL = os.getenv("LLMOD_EMBEDDING_MODEL", "NBUECSE-text-embedding-3-small")
CHAT_MODEL = os.getenv("LLMOD_CHAT_MODEL", "NBUECSE-gpt-5-mini")

LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = os.getenv("LLMOD_BASE_URL")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

NAMESPACE = "medium-all-380w-10ov"

APPROX_CHUNK_SIZE_TOKENS = 512
OVERLAP_RATIO = 0.10
RETRIEVAL_TOP_K = 30
MAX_ARTICLES_IN_PROMPT = 4
MAX_CONTEXT_ITEMS = 12

EXPECTED_EMBEDDING_DIMENSION = 1536

UNKNOWN_RESPONSE = "I don’t know based on the provided Medium articles data."

REQUIRED_SYSTEM_PROMPT_SECTION = f"""
You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “{UNKNOWN_RESPONSE}” Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.
""".strip()


# ----------------------------
# App setup
# ----------------------------

app = FastAPI(title="Medium Article RAG Assistant")


class PromptRequest(BaseModel):
    question: str


# ----------------------------
# Clients
# ----------------------------

def get_llmod_client() -> OpenAI:
    if not LLMOD_API_KEY:
        raise RuntimeError("Missing LLMOD_API_KEY")

    if not LLMOD_BASE_URL:
        raise RuntimeError("Missing LLMOD_BASE_URL")

    return OpenAI(
        api_key=LLMOD_API_KEY,
        base_url=LLMOD_BASE_URL,
    )


def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")

    if not PINECONE_INDEX_HOST:
        raise RuntimeError("Missing PINECONE_INDEX_HOST")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(host=PINECONE_INDEX_HOST)


# ----------------------------
# Helper functions
# ----------------------------

def metadata_value(metadata: Dict[str, Any], key: str, default: str = "") -> Any:
    value = metadata.get(key, default)
    if value is None:
        return default
    return value


def embed_question(client: OpenAI, question: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    embedding = response.data[0].embedding

    if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected embedding dimension: {len(embedding)}. "
            f"Expected {EXPECTED_EMBEDDING_DIMENSION}."
        )

    return embedding


def retrieve_context(question_embedding: List[float], top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
    index = get_pinecone_index()

    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
    )

    matches = getattr(results, "matches", [])

    retrieved_items = []

    for match in matches:
        metadata = getattr(match, "metadata", {}) or {}
        chunk_id = metadata_value(metadata, "chunk_id") or getattr(match, "id", "")

        retrieved_items.append(
            {
                "score": getattr(match, "score", None),
                "chunk_id": chunk_id,
                "article_id": metadata_value(metadata, "article_id"),
                "chunk_index": metadata_value(metadata, "chunk_index"),
                "title": metadata_value(metadata, "title"),
                "authors": metadata_value(metadata, "authors"),
                "url": metadata_value(metadata, "url"),
                "timestamp": metadata_value(metadata, "timestamp"),
                "tags": metadata_value(metadata, "tags"),
                "chunk": metadata_value(metadata, "chunk"),
            }
        )

    return retrieved_items


def group_context_by_article(
    retrieved_items: List[Dict],
    max_articles: int = MAX_ARTICLES_IN_PROMPT,
    max_context_items: int = MAX_CONTEXT_ITEMS,
) -> List[Dict]:
    """
    Select context chunks while balancing relevance, article diversity, and prompt size.

    Strategy:
    1. Pinecone returns retrieved_items in score order.
    2. Choose up to max_articles distinct articles from those results.
    3. Guarantee one best chunk from each selected article.
    4. Fill the remaining context slots with the best remaining chunks
       from those selected articles, still in score order.
    5. Return grouped article context for the LLM prompt.
    """

    selected_article_ids = []

    # First pass: choose up to max_articles distinct articles.
    for item in retrieved_items:
        article_id = item.get("article_id")

        if not article_id:
            continue

        if article_id not in selected_article_ids:
            selected_article_ids.append(article_id)

        if len(selected_article_ids) >= max_articles:
            break

    selected_article_id_set = set(selected_article_ids)

    selected_items = []
    selected_chunk_ids = set()

    # Second pass: guarantee one best chunk from each selected article.
    for article_id in selected_article_ids:
        for item in retrieved_items:
            chunk_id = item.get("chunk_id")

            if item.get("article_id") == article_id and chunk_id not in selected_chunk_ids:
                selected_items.append(item)
                selected_chunk_ids.add(chunk_id)
                break

    # Third pass: fill remaining slots with the best remaining chunks
    # from the selected articles.
    for item in retrieved_items:
        article_id = item.get("article_id")
        chunk_id = item.get("chunk_id")

        if article_id not in selected_article_id_set:
            continue

        if chunk_id in selected_chunk_ids:
            continue

        selected_items.append(item)
        selected_chunk_ids.add(chunk_id)

        if len(selected_items) >= max_context_items:
            break

    groups_by_article_id = {}

    for item in selected_items:
        article_id = item.get("article_id")

        if article_id not in groups_by_article_id:
            groups_by_article_id[article_id] = {
                "article_id": article_id,
                "title": item.get("title", ""),
                "authors": item.get("authors", ""),
                "url": item.get("url", ""),
                "timestamp": item.get("timestamp", ""),
                "tags": item.get("tags", ""),
                "best_score": item.get("score"),
                "chunks": [],
            }

        article_group = groups_by_article_id[article_id]

        score = item.get("score")
        best_score = article_group.get("best_score")

        if isinstance(score, (int, float)) and (
            not isinstance(best_score, (int, float)) or score > best_score
        ):
            article_group["best_score"] = score

        article_group["chunks"].append(
            {
                "chunk_id": item.get("chunk_id", ""),
                "chunk_index": item.get("chunk_index", ""),
                "score": score,
                "chunk": item.get("chunk", ""),
            }
        )

    article_groups = [
        groups_by_article_id[article_id]
        for article_id in selected_article_ids
        if article_id in groups_by_article_id
    ]

    return article_groups


def build_context_text(article_groups: List[Dict]) -> str:
    context_parts = []

    for article_number, article_group in enumerate(article_groups, start=1):
        article_text = f"""
[Article {article_number}]
Title: {article_group.get("title", "")}
Authors: {article_group.get("authors", "")}

Relevant passages:
""".strip()

        for chunk in article_group.get("chunks", []):
            article_text += f"""

Relevant excerpt:
{chunk.get("chunk", "")}
"""

        context_parts.append(article_text)

    return "\n\n".join(context_parts)


def build_augmented_prompt(question: str, article_groups: List[Dict]) -> Dict[str, str]:
    context_text = build_context_text(article_groups)

    system_prompt = f"""
Required assignment system-prompt section:
{REQUIRED_SYSTEM_PROMPT_SECTION}

Additional implementation instructions:

Use the retrieved article content and metadata to answer the user's question.

You may summarize, explain, list titles, or recommend articles only when the answer is directly supported by the retrieved context.

If the retrieved context does not directly answer the question, respond exactly:
{UNKNOWN_RESPONSE}

Do not invent article titles, authors, facts, or article content.

Do not add unsupported advice, generic knowledge, or assumptions.

Do not mention chunk IDs, passage numbers, excerpt labels, similarity scores, article IDs, retrieval ranks, or internal context labels in the final answer.

Do not offer follow-up help, checklists, next steps, or answers from other sources.

Answer concisely and directly.
""".strip()

    user_prompt = f"""
The retrieved context is grouped by article. Multiple passages under the same article belong to the same Medium article.

If the user asks for multiple article titles, choose the best matching distinct article titles from the article groups and return at most 3, even if more than 3 article groups are provided.

If the user asks to return only titles, return only the titles. Do not add explanations, quotes, bullets with extra text, or justifications for title-only requests.

If the user asks about one article, use the relevant article group and its passages.

If the user asks for a recommendation, compare the retrieved article groups and recommend the article that best matches the user's need based on the provided passages. Do not assume the first article is automatically the best if another article is better supported.

If the user asks for a recommendation, recommend one best article unless the user explicitly asks for multiple options. Do not add secondary recommendations, extra notes, or alternative suggestions.

Retrieved Medium article context:

{context_text}

User question:
{question}
""".strip()

    return {
        "System": system_prompt,
        "User": user_prompt,
    }


def call_chat_model(client: OpenAI, augmented_prompt: Dict[str, str]) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": augmented_prompt["System"]},
            {"role": "user", "content": augmented_prompt["User"]},
        ],
    )

    return response.choices[0].message.content


def flatten_article_groups_for_api_context(article_groups: List[Dict]) -> List[Dict]:
    api_context = []

    for article_group in article_groups:
        for chunk in article_group.get("chunks", []):
            api_context.append(
                {
                    "article_id": article_group.get("article_id", ""),
                    "title": article_group.get("title", ""),
                    "chunk": chunk.get("chunk", ""),
                    "score": chunk.get("score"),
                }
            )

    return api_context[:MAX_CONTEXT_ITEMS]


# ----------------------------
# API endpoints
# ----------------------------

@app.get("/api/stats")
def get_stats():
    return {
        "chunk_size": APPROX_CHUNK_SIZE_TOKENS,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": RETRIEVAL_TOP_K,
    }


@app.post("/api/prompt")
def prompt(request: PromptRequest):
    question = request.question.strip()

    client = get_llmod_client()

    question_embedding = embed_question(client, question)
    retrieved_items = retrieve_context(question_embedding, top_k=RETRIEVAL_TOP_K)
    article_groups = group_context_by_article(retrieved_items)

    augmented_prompt = build_augmented_prompt(question, article_groups)
    final_response = call_chat_model(client, augmented_prompt)

    api_context = flatten_article_groups_for_api_context(article_groups)

    return {
        "response": final_response,
        "context": api_context,
        "Augmented_prompt": augmented_prompt,
    }
