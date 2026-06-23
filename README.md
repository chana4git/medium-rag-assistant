# Medium Article RAG Assistant

This project implements a Retrieval-Augmented Generation assistant over a Medium articles dataset. The assistant answers questions only from retrieved Medium article context and returns the context used for the answer.

## API Endpoints

### GET /api/stats

Returns the RAG configuration:

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.1,
  "top_k": 30
}
POST /api/prompt

Request:

{
  "question": "List up to 3 articles about writing headlines. Return only the article titles."
}

Response:

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
    "System": "system prompt used for the chat model",
    "User": "user prompt with retrieved context"
  }
}
RAG Configuration

Final configuration:

Approximate chunk size: 512 tokens
Implemented chunk size: 380 words
Overlap ratio: 0.1
Retrieval top-k: 30
Pinecone namespace: medium-all-380w-10ov

The assignment defines chunk size in tokens. To avoid adding a tokenizer dependency, this project approximates 512 tokens as about 380 words.

Context Selection Strategy

The system retrieves the top 30 most similar chunks from Pinecone. Since Pinecone retrieves chunks rather than full articles, several high-ranking chunks may come from the same article.

To balance relevance and article diversity, the code groups retrieved chunks by article_id. It selects up to 4 distinct article groups, guarantees one strong chunk from each selected article, and then fills the remaining context slots with the best remaining chunks from those selected articles, up to 12 chunks total.

For multi-article listing questions, the model may see up to 4 candidate articles, but it is instructed to return at most 3 distinct article titles.

Models

Embedding model: NBUECSE-text-embedding-3-small

Chat model: NBUECSE-gpt-5-mini

Environment Variables

The app expects the following environment variables:

LLMOD_API_KEY
LLMOD_BASE_URL
LLMOD_EMBEDDING_MODEL
LLMOD_CHAT_MODEL
PINECONE_API_KEY
PINECONE_INDEX_HOST
Local Run
uvicorn main:app --reload
Deployment

The project is deployed as a FastAPI app. The Vercel entrypoint is api/index.py, which imports the FastAPI app from main.py.