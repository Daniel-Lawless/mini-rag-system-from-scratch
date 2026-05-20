## Knowledge Base

This project currently uses my own mathematical statistics derivations as the retrieval corpus.

The original human-readable versions are maintained in my separate statistics repository:

[Statistics for AI](https://github.com/Daniel-Lawless/Statistics-for-ai)

For the RAG system, I keep simplified, machine-readable versions of selected derivations in the `data` folder so they can be chunked and embedded more cleanly.

Some of the original derivations use diagrams to explain geometric or abstract ideas, such as Buffon's Needle, the German Tank Problem, and the Lighthouse Problem. In the RAG versions, I keep links to the original images from the statistics repository, but I also add written explanations around each image.

This is important because a basic RAG pipeline does not understand the contents of an image directly. By explaining each diagram in text, the system can embed and retrieve the meaning of the image without needing image processing or OCR.

So, the images are still available for human readers, while the surrounding text makes the same ideas accessible to the retrieval system.

## Roadmap

- [x] Build an in-memory vector database
- [x] Add cosine similarity search
- [x] Add chunking with overlap
- [x] Add SentenceTransformer embeddings
- [x] Connect retrieved chunks to OpenAI generation
- [ ] Add metadata storage for chunks
- [ ] Improve chunking with paragraph/sentence-aware splitting
- [ ] Vectorize search using NumPy matrix multiplication
