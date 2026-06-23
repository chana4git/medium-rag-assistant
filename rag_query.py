# You may need to install these packages first:
# pip install openai python-dotenv pinecone

import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load .env before reading model names from environment variables.
load_dotenv()


# Edit these constants and rerun this file to test different RAG questions.
#TEST_QUESTION = "Who wrote the article 'How a Single Medium Article Received 100,000 Views'?"
#TEST_QUESTION = "List up to 3 articles about writing headlines. Return only the article titles."
#TEST_QUESTION = "What is the central idea of the article 'Avoid Clickbait: Headline Techniques Used by Six Reputable Media Sites'?"
#TEST_QUESTION = "Recommend an article for someone who wants to improve their Medium headlines, and explain why based on the article."
#TEST_QUESTION = "Find an article that argues what long term effects can happen by a pandemic, and summarise its central argument"
#TEST_QUESTION = "Tell me about quantum physics and black holes."
#TEST_QUESTION = "Who wrote the article 'Hot’n’Pop Song Machine: end-to-end Machine Learning classificator project'?"
#TEST_QUESTION = "What is the central idea of the article 'Hot’n’Pop Song Machine: end-to-end Machine Learning classificator project'?"
#TEST_QUESTION = "According to the article 'Hot’n’Pop Song Machine: end-to-end Machine Learning classificator project', what data sources did the project use?"
#TEST_QUESTION = "Recommend an article for someone who wants to build and deploy a machine learning music prediction app, and explain why."
#TEST_QUESTION = "In the Hot’n’Pop Song Machine project, which final model was chosen and why?"
#TEST_QUESTION = "List up to 3 articles about writing headlines. Return only the article titles."
#TEST_QUESTION = "List up to 3 articles about building and deploying machine learning projects. Return only the article titles."
TEST_QUESTION = "List up to 3 articles about using machine learning for music, songs, or audio. Return only the article titles."
#TEST_QUESTION = "List up to 3 articles about artificial intelligence in the energy sector or renewable energy. Return only the article titles."
#TEST_QUESTION = "List up to 3 articles about Kubernetes, Istio, service mesh, or traffic management. Return only the article titles."
RETRIEVAL_TOP_K = 30
MAX_ARTICLES_IN_PROMPT = 4
MAX_CONTEXT_ITEMS = 12
NAMESPACE = "medium-all-380w-10ov"
EXPECTED_EMBEDDING_DIMENSION = 1536
UNKNOWN_RESPONSE = "I don’t know based on the provided Medium articles data."

REQUIRED_SYSTEM_PROMPT_SECTION = f"""
You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “{UNKNOWN_RESPONSE}” Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.
""".strip()

EMBEDDING_MODEL = os.getenv("LLMOD_EMBEDDING_MODEL", "NBUECSE-text-embedding-3-small")
CHAT_MODEL = os.getenv("LLMOD_CHAT_MODEL", "NBUECSE-gpt-5-mini")


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
    """Connect to the existing Pinecone index without creating or modifying it."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if pinecone_index_host:
        print("Connecting to Pinecone index using PINECONE_INDEX_HOST.")
        return pc.Index(host=pinecone_index_host)

    print("Connecting to Pinecone index using PINECONE_INDEX_NAME.")
    return pc.Index(pinecone_index_name)


def metadata_value(metadata, key: str, default=""):
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


def embed_question(question: str) -> List[float]:
    """Embed only the user question once for this local RAG test."""
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


def retrieve_context(
    question_embedding: List[float],
    top_k: int = RETRIEVAL_TOP_K,
) -> List[Dict]:
    """Retrieve top matching chunks from Pinecone and keep compact metadata."""
    if index is None:
        raise RuntimeError("Pinecone index is not initialized.")

    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
    )

    matches = getattr(results, "matches", [])

    context_items = []

    for match in getattr(results, "matches", []):
        metadata = getattr(match, "metadata", {}) or {}
        match_id = getattr(match, "id", "")

        context_items.append(
            {
                "score": getattr(match, "score", None),
                "chunk_id": metadata_value(metadata, "chunk_id", match_id),
                "article_id": metadata_value(metadata, "article_id"),
                "chunk_index": metadata_value(metadata, "chunk_index"),
                "title": metadata_value(metadata, "title"),
                "authors": metadata_value(metadata, "authors"),
                "tags": metadata_value(metadata, "tags"),
                "chunk": metadata_value(metadata, "chunk"),
            }
        )

    return context_items


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

    # Preserve article order by first appearance in the retrieved results.
    article_groups = [
        groups_by_article_id[article_id]
        for article_id in selected_article_ids
        if article_id in groups_by_article_id
    ]

    return article_groups


def flatten_article_groups_for_context(article_groups: List[Dict]) -> List[Dict]:
    """Flatten grouped article context for the API-shaped response."""
    context_items = []

    for article_group in article_groups:
        for chunk in article_group.get("chunks", []):
            context_items.append(
                {
                    "article_id": article_group.get("article_id", ""),
                    "title": article_group.get("title", ""),
                    "authors": article_group.get("authors", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "chunk_index": chunk.get("chunk_index", ""),
                    "score": chunk.get("score"),
                    "chunk": chunk.get("chunk", ""),
                }
            )

            if len(context_items) >= MAX_CONTEXT_ITEMS:
                return context_items

    return context_items


def build_context_text(article_groups: List[Dict]) -> str:
    """Build article-grouped context for the chat model."""
    if not article_groups:
        return "No retrieved Medium article context was found."

    context_sections = []

    # The returned context can include extra metadata for transparency/debugging.
    # The augmented prompt should stay compact and include only information the model needs to answer.
    for result_number, article_group in enumerate(article_groups, start=1):
        lines = [
            f"[Article {result_number}]",
            f"Title: {article_group.get('title', '')}",
            f"Authors: {format_list_value(article_group.get('authors'))}",
            "",
            "Relevant passages:",
        ]

        for chunk in article_group.get("chunks", []):
            lines.extend(
                [
                    "Relevant excerpt:",
                    str(chunk.get("chunk", "")),
                    "",
                ]
            )

        context_sections.append("\n".join(lines).strip())

    return "\n\n".join(context_sections)


def build_augmented_prompt(
    question: str,
    article_groups: List[Dict],
) -> Dict[str, str]:
    """Build the system and user messages for the RAG chat call."""
    context_text = build_context_text(article_groups)

    system_message = (
        "Required assignment system-prompt section:\n"
        f"{REQUIRED_SYSTEM_PROMPT_SECTION}\n\n"
        "Additional implementation instructions:\n\n"
        "Use the retrieved article content and metadata to answer the user's "
        "question.\n\n"
        "You may summarize, explain, list titles, or recommend articles only "
        "when the answer is directly supported by the retrieved context.\n\n"
        "If the retrieved context does not directly answer the question, respond "
        f"exactly:\n{UNKNOWN_RESPONSE}\n\n"
        "Do not invent article titles, authors, facts, or article content.\n\n"
        "Do not add unsupported advice, generic knowledge, or assumptions.\n\n"
        "Do not mention chunk IDs, passage numbers, excerpt labels, similarity scores, article IDs, retrieval "
        "ranks, or internal context labels in the final answer.\n\n"
        "Do not offer follow-up help, checklists, next steps, or answers from other sources.\n\n"
        "Answer concisely and directly."
    )

    user_message_parts = [
        "The retrieved context is grouped by article. Multiple passages under "
        "the same article belong to the same Medium article.",
        "",
        "If the user asks for multiple article titles, "
        "choose the best matching distinct article titles from "
        "the article groups and return at most 3, "
        "even if more than 3 article groups are provided. ",
        "",
        "If the user asks to return only titles, return only the titles. "
        "Do not add explanations, quotes, bullets with extra text, or "
        "justifications for title-only requests.",
        "",
        "If the user asks about one article, use the relevant article group and "
        "its passages.",
        "",
        "If the user asks for a recommendation, compare the retrieved article groups "
        "and recommend the article that best matches the user's need based on the "
        "provided passages. Do not assume the first article is automatically the best "
        "if another article is better supported.",
        "",
        "If the user asks for a recommendation, recommend one best article unless the "
        "user explicitly asks for multiple options. Do not add secondary recommendations, "
        "extra notes, or alternative suggestions.",
        "",
    ]

    user_message_parts.extend(
        [
            "Retrieved Medium article context:",
            "",
            context_text,
            "",
            "User question:",
            question,
        ]
    )

    user_message = "\n".join(user_message_parts)

    return {
        "System": system_message,
        "User": user_message,
    }


def call_chat_model(augmented_prompt: Dict[str, str]) -> str:
    """Call the chat model exactly once and return its response text."""
    if client is None:
        raise RuntimeError("LLMod.ai client is not initialized.")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": augmented_prompt["System"]},
            {"role": "user", "content": augmented_prompt["User"]},
        ],
    )

    final_text = response.choices[0].message.content.strip()

    if UNKNOWN_RESPONSE in final_text:
        return UNKNOWN_RESPONSE

    return final_text


def print_article_group_summary(article_groups: List[Dict]) -> None:
    """Print compact grouped-article metadata for local debugging."""
    if not article_groups:
        print("No article groups returned.")
        return

    for rank, article_group in enumerate(article_groups, start=1):
        best_score = article_group.get("best_score")
        rounded_best_score = (
            round(best_score, 3) if isinstance(best_score, (int, float)) else best_score
        )
        chunks = article_group.get("chunks", [])
        first_chunk_preview = ""

        if chunks:
            first_chunk_preview = str(chunks[0].get("chunk", ""))[:200]

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"Best score: {rounded_best_score}")
        print(f"Article ID: {article_group.get('article_id', '')}")
        print(f"Title: {article_group.get('title', '')}")
        print(f"Authors: {format_list_value(article_group.get('authors'))}")
        print(f"Chunks included: {len(chunks)}")
        print("First chunk preview:")
        print(first_chunk_preview)


if __name__ == "__main__":
    # Local end-to-end RAG test:
    # 1. Embed the question once.
    # 2. Retrieve a wider set of existing chunks from Pinecone.
    # 3. Group the retrieved chunks by article for a compact prompt.
    # 4. Send the grouped context and question to the chat model once.
    # This does not read the CSV, re-chunk, upsert, or delete vectors.
    print("Starting local RAG query test.")
    print(f"Test question: {TEST_QUESTION}")

    validate_environment()

    client = connect_to_llmod()
    index = connect_to_pinecone_index()

    embedding = embed_question(TEST_QUESTION)
    retrieved_items = retrieve_context(
        embedding,
        top_k=RETRIEVAL_TOP_K,
    )
    article_groups = group_context_by_article(retrieved_items)
    context_items = flatten_article_groups_for_context(article_groups)
    augmented_prompt = build_augmented_prompt(
        TEST_QUESTION,
        article_groups,
    )
    final_response = call_chat_model(augmented_prompt)

    print("\n" + "=" * 80)
    print("Final response")
    print("=" * 80)
    print(final_response)

    print("\n" + "=" * 80)
    print("Article group summary")
    print("=" * 80)
    print(f"Retrieved candidates: {len(retrieved_items)}")
    print(f"Article groups in prompt: {len(article_groups)}")
    print(f"Context items in API output: {len(context_items)}")
    print_article_group_summary(article_groups)

    print("\n" + "=" * 80)
    print("Augmented prompt")
    print("=" * 80)
    print("\nSystem:")
    print(augmented_prompt["System"])
    print("\nUser:")
    print(augmented_prompt["User"])

    api_shaped_output = {
        "response": final_response,
        "context": context_items,
        "Augmented_prompt": augmented_prompt,
    }

    print("\n" + "=" * 80)
    print("Final API-shaped output")
    print("=" * 80)
    print("Final API-shaped output created successfully.")
    print("Keys: response, context, Augmented_prompt")
    print(f"Context items: {len(api_shaped_output['context'])}")
    print(
        "Augmented prompt keys: "
        + ", ".join(api_shaped_output["Augmented_prompt"].keys())
    )
