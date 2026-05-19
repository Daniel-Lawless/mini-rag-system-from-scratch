## Knowledge Base

This project currently uses my own mathematical derivations as the retrieval corpus.

The original human-readable versions are maintained in my separate statistics repository:

[Statistics for AI](../Statistics-for-ai-)

For the RAG system, I keep simplified machine-readable versions of selected derivations in the `data/` folder so they can be chunked and embedded more cleanly.

## Roadmap

- [x] Build an in-memory vector database
- [x] Add cosine similarity search
- [x] Add chunking with overlap
- [x] Add SentenceTransformer embeddings
- [x] Connect retrieved chunks to OpenAI generation
- [ ] Add metadata storage for chunks
- [ ] Improve chunking with paragraph/sentence-aware splitting
- [ ] Vectorize search using NumPy matrix multiplication
