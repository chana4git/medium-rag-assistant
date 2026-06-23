# Medium Article RAG Assistant

**Student:** Chana Gutenmacher
**Course:** AI Agents – Individual Assignment
## Design Decisions and Implementation Notes

### 1. Chunk retrieval vs. article-level answers

A main issue I ran into was that Pinecone retrieves **chunks**, while some questions need answers at the **article** level.

For example, if the user asks for 3 article titles, the top retrieved chunks may all come from the same long article. So using the raw Pinecone results directly can give too little article diversity.

To solve this, I added a `group_context_by_article()` function. It takes the ranked chunks from Pinecone and builds article-level context:

1. Retrieve the top `30` chunks.
2. Group chunks by `article_id`.
3. Select up to `4` different articles.
4. Make sure each selected article gets at least one strong chunk.
5. Add more high-scoring chunks from those articles until reaching the `12` chunk limit.

This way the model gets both article diversity and enough supporting text.

### 2. Why top-k is 30, but the prompt gets only 12 chunks

I use `top_k = 30` in Pinecone because Medium articles can be much longer than one chunk. With a smaller top-k, one relevant article can take up most of the retrieval results.

But I do not send all 30 chunks to the model. The code first retrieves broadly, then `group_context_by_article()` builds a smaller prompt context: up to `4` articles and up to `12` chunks total.

The fourth article is intentional. For questions asking for 3 titles, it gives the model one extra candidate article to compare, while still telling it to return at most 3 titles.

### 3. Chunk size experiment

I tested two chunking settings:

* `380 words / 10% overlap`
* `550 words / 15% overlap`

I chose `380 words / 10% overlap`.

The assignment defines chunk size in tokens, so I used the approximation: `512 tokens ≈ 380 words`.

The 550-word chunks also worked, but they did not clearly improve the answers. They gave more surrounding text, but also made the prompt larger and less focused. The 380-word chunks were more focused and worked well in the tests, so I kept them.

### 4. Why I did not add a classifier

I considered adding a question classifier, but decided not to.

The same retrieval flow worked for the different cases once the context was grouped by article. This kept the implementation simpler and avoided another model call, while still supporting cases where the user asks for one article, several titles, a summary, or a recommendation.

## Final Configuration

```json id="1xw22n"
{
  "chunk_size": 512,
  "overlap_ratio": 0.1,
  "top_k": 30
}
```

Additional internal settings:

* `MAX_ARTICLES_IN_PROMPT = 4`
* `MAX_CONTEXT_ITEMS = 12`
* Pinecone namespace: `medium-all-380w-10ov`

## General Assignment / API Notes

### `GET /api/stats`

Returns:

```json id="qxl3zr"
{
  "chunk_size": 512,
  "overlap_ratio": 0.1,
  "top_k": 30
}
```

### `POST /api/prompt`

Request example:

```json id="hmlz7j"
{
  "question": "List up to 3 articles about writing headlines. Return only the article titles."
}
```

Response includes:

```json id="vjx085"
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "article_id": "4083",
      "title": "How to Write a Headline",
      "chunk": "retrieved article chunk",
      "score": 0.6119
    }
  ],
  "Augmented_prompt": {
    "System": "system prompt used to query the chat model",
    "User": "user prompt used to query the chat model"
  }
}
```

### Models

* Embedding model: `NBUECSE-text-embedding-3-small`
* Chat model: `NBUECSE-gpt-5-mini`
* Expected embedding dimension: `1536`

### Environment Variables

```text id="4pkegt"
LLMOD_API_KEY
LLMOD_BASE_URL
LLMOD_EMBEDDING_MODEL
LLMOD_CHAT_MODEL
PINECONE_API_KEY
PINECONE_INDEX_HOST
```

### Local Run

```bash id="az0dk7"
uvicorn main:app --reload
```

### Deployment

The app is deployed as a FastAPI app. For Vercel, `api/index.py` imports the FastAPI app from `main.py`.

### Repository Notes

The dataset files, generated chunk files, `.env`, and virtual environment are excluded from GitHub using `.gitignore`.

The Pinecone index should remain active for grading.
