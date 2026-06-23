## Initial Retrieval Smoke Tests

Before running the full RAG pipeline, I tested whether Pinecone retrieval could return relevant chunks for different kinds of questions.

### Exact-title retrieval

Question:

`What is the central idea of the article "Avoid Clickbait: Headline Techniques Used by Six Reputable Media Sites"?`

Result:

The top retrieved results came from the correct article, with strong similarity scores around 0.85 and 0.83.

Judgment:

This showed strong exact-title retrieval. When the user question contains a specific article title, the vector search is able to retrieve chunks from the correct article.

### Semantic topic retrieval

Question:

`How to market a Medium article`

Result:

The top retrieved results included a relevant article about how a single Medium article received 100,000 views. Scores were around 0.59–0.52.

Judgment:

This showed good semantic retrieval. The system was able to retrieve article chunks related to Medium marketing and traffic generation even though the user question did not exactly match an article title.

### Unrelated / unsupported question

Question:

`Quantum physics and black holes`

Result:

The top score was low, around 0.23, and the retrieved chunks were weak or unrelated.

Judgment:

This showed a case where the system should probably respond that it does not know based on the provided Medium articles data. This helped motivate the system prompt rule that the assistant must not answer from outside knowledge when the retrieved context does not support the question.

# RAG Experiment Notes

## Goal

The goal of this experiment was to evaluate whether the RAG system retrieves relevant Medium article passages and produces grounded answers for the assignment’s required question types:

1. Precise fact retrieval
2. Multi-result topic listing
3. Key idea summary extraction
4. Recommendation with evidence-based justification

The system answers only from retrieved Medium article context and does not use external knowledge.

---

## Retrieval and Context Selection Strategy

The system retrieves the top 15 matching chunks from Pinecone for each user question.

Because Pinecone retrieves chunks rather than full articles, multiple retrieved chunks may belong to the same article. To avoid duplicate titles and to provide coherent article-level context, the retrieved chunks are grouped by `article_id`.

The grouping logic works as follows:

1. Retrieve `RETRIEVAL_TOP_K = 15` candidate chunks.
2. Group retrieved chunks by `article_id`.
3. Sort chunks inside each article group by similarity score.
4. Keep up to `MAX_CHUNKS_PER_ARTICLE = 4` chunks per article.
5. Rank article groups by their best retrieved chunk score.
6. Keep up to `MAX_ARTICLES_IN_PROMPT = 3` article groups.
7. Send the grouped article context to the chat model.

The model sees only the article title, authors, and selected article excerpts. Internal retrieval metadata such as chunk IDs, article IDs, similarity scores, and retrieval ranks is kept in the API/debug output but is not included in the final answer.

---

## Chunking Configuration A — Baseline

* Namespace: `medium-50-test-v2`
* Article subset: first 50 articles
* Chunk size: 380 words
* Approximate token size: about 512 tokens
* Overlap ratio: 0.10
* Retrieval top_k: 15
* Max article groups in prompt: 3
* Max chunks per article: 4

### Results

#### 1. Precise fact retrieval

Question:

`Who wrote the article 'How a Single Medium Article Received 100,000 Views'?`

Result:

The system correctly returned Casey Botticello.

Notes:

The target article was retrieved as the top article group. The final answer was concise and did not mention chunks, scores, article IDs, or retrieval labels.

#### 2. Multi-result topic listing

Question:

`List up to 3 articles about writing headlines. Return only the article titles.`

Result:

The system returned three distinct relevant titles:

* An Effective Five-Step Process for Writing Captivating Headlines
* Avoid Clickbait: Headline Techniques Used by Six Reputable Media Sites
* How a Single Medium Article Received 100,000 Views

Notes:

The answer returned distinct article titles and did not include explanations or internal retrieval labels.

#### 3. Key idea summary

Question:

`What is the central idea of the article 'Avoid Clickbait: Headline Techniques Used by Six Reputable Media Sites'?`

Result:

The system correctly summarized that the article argues against clickbait because it harms credibility and reader trust, and explains that writers can learn honest headline techniques from reputable media outlets.

Notes:

The model used the relevant target article group and did not mix in ideas from the other retrieved article groups.

#### 4. Recommendation

Question:

`Recommend an article for someone who wants to improve their Medium headlines, and explain why based on the article.`

Result:

The system recommended “An Effective Five-Step Process for Writing Captivating Headlines” by Nicole Bianchi.

Notes:

The recommendation was grounded in retrieved article content: one-sentence synopsis, specificity, curiosity, power words, reader benefit, multiple headline drafts, and headline formulas. The answer did not expose internal retrieval details.

---

## Chunking Configuration B — Larger Chunks

* Namespace: `medium-50-550w-15ov`
* Article subset: first 50 articles
* Chunk size: 550 words
* Approximate token size: about 700–750 tokens
* Overlap ratio: 0.15
* Retrieval top_k: 15
* Max article groups in prompt: 3
* Max chunks per article: 4

### Results

The larger-chunk configuration also answered the four test questions correctly. It provided more surrounding context per retrieved excerpt and worked well for summary and recommendation questions.

However, the larger chunks did not materially improve the answers compared with the baseline. They also produced larger prompts. In one recommendation run, the model referenced internal passage labels, which showed that larger, more verbose prompts can make it easier for the model to copy prompt structure into the final answer.

After this observation, the prompt was updated to avoid numbered passage labels and instead use neutral “Relevant excerpt” labels. The system instruction was also strengthened to tell the model not to mention passage numbers, excerpt labels, similarity scores, article IDs, retrieval ranks, or internal context labels in the final answer.

---

## Final Larger-Subset Test

* Namespace: `medium-250-380w-10ov`
* Article subset: first 250 articles
* Chunk size: 380 words
* Approximate token size: about 512 tokens
* Overlap ratio: 0.10
* Retrieval top_k: 15
* Max article groups in prompt: 3
* Max chunks per article: 4

### Results

#### 1. Multi-result topic listing

Question:

`List up to 3 articles about writing headlines. Return only the article titles.`

Result:

The system returned:

* An Effective Five-Step Process for Writing Captivating Headlines
* Why You Need to Write More Than One Headline
* Avoid Clickbait: Headline Techniques Used by Six Reputable Media Sites

Notes:

With 250 articles, the retrieved set changed slightly compared with the 50-article run. This is expected because the vector database contains more candidate articles. The answer still returned three distinct and relevant headline-related titles.

#### 2. Key idea summary

Question:

`What is the central idea of the article 'Avoid Clickbait: Headline Techniques Used by Six Reputable Media Sites'?`

Result:

The system correctly summarized the target article: clickbait harms credibility and reader trust, and writers can learn from reputable media outlets how to create engaging but non-misleading headlines.

Notes:

The target article was retrieved as the top article group, and the final answer was based only on that article.

#### 3. Recommendation

Question:

`Recommend an article for someone who wants to improve their Medium headlines, and explain why based on the article.`

Result:

The system recommended “An Effective Five-Step Process for Writing Captivating Headlines” by Nicole Bianchi.

Notes:

The answer was grounded in the article’s practical headline advice, including the one-sentence synopsis, specificity, curiosity, power words, reader benefit, multiple headline variants, and headline formulas. The system selected the best matching article rather than blindly using the first retrieved result.

#### 4. Additional stress test

Question:

`Find an article that discusses long-term effects of the pandemic, and summarize its central argument.`

Result:

The system selected “Your Brain On Coronavirus” by Simon Spichak and summarized its argument about long-term neurological and psychiatric effects of the COVID-19 pandemic, including brain health, mental health, stroke risk, and long-hauler symptoms.

Notes:

Although this article was not the top-ranked article group, it was the best semantic match for the user’s question. The model correctly selected it from the retrieved article groups and summarized only that article. This supports the decision to provide several article groups to the model instead of forcing it to use only the top-ranked group.

---

## Final Configuration Decision

I selected the 380-word chunk size with 10% overlap as the final configuration.

Reasons:

1. It produced focused retrieved passages.
2. It answered all required question types correctly.
3. It kept prompts smaller than the larger 550-word configuration.
4. It avoided unnecessary prompt length and reduced the chance of the model copying internal prompt labels.
5. It still provided enough context for precise retrieval, multi-title listing, central idea summaries, and recommendations.

Final settings:

* `chunk_size`: approximately 512 tokens
* `CHUNK_SIZE_WORDS`: 380 words
* `overlap_ratio`: 0.10
* `top_k`: 15
* `MAX_ARTICLES_IN_PROMPT`: 3
* `MAX_CHUNKS_PER_ARTICLE`: 4
