# Progress Log

## Day 1 — Basic RAG pipeline

Implemented:
- Word-based chunking with overlap
- SentenceTransformer embeddings
- In-memory vector DB
- Top-k retrieval using cosine similarity
- OpenAI response generation using retrieved context

Key lesson:
- Retrieval quality controls answer quality. Increasing k from 2 to 4 allowed the model to retrieve the chunk containing the final weighted mean estimator.
- Normalizing vectors before calculating the cosine similarity reduces the computation down to only the dot product.

## Day 2 — Metadata-aware chunk records and retrieval debugging

Implemented:

- Replaced plain chunk storage with metadata-aware records
- Each vector DB record now stores:
  - chunk text
  - embedding
  - source file
  - chunk index
- Updated the RAG pipeline so each indexed chunk keeps track of where it came from
- Updated retrieval so search returns full records instead of just chunk text
- Added logging for indexing and retrieval debugging
- Added source file and chunk index information to the retrieved context passed to the model

Example retrieval debug output with query:

```text
What is the final probability for Buffon's needle?
```

![Metadata-aware retrieval debug output](../assets/retrieval-debug-output.png)

1: The system first indexes each markdown file in the data/ directory.


2: The debug logs then show the top-k chunks retrieved for the query.


3: Each retrieved chunk includes useful metadata:
- `similarity`: the cosine similarity between the query embedding and the chunk embedding
- `Retrieved chunk`: The chunk returned from the search algorithm
- `source_file`: the markdown file that the chunk came from
- `chunk_index`: the position of the chunk within that source file


4: The final answer is generated from the retrieved context.
```math
P(crossing) = \frac{2L}{\pi t}
```
This confirms that the system retrieved the correct context and used it to answer the query.

Key lesson:

- Metadata makes retrieval much easier to inspect and debug.
- Seeing the `source_file` and `chunk_index` confirms whether the system is retrieving context from the correct document.
- For the query asking about Buffon's Needle, the highest-ranked retrieved chunks came from `04-buffons-needle.md`, which is the expected behaviour.
- Some lower-ranked chunks may come from unrelated files when `k` is larger, so choosing an appropriate number of retrieved chunks matters.
