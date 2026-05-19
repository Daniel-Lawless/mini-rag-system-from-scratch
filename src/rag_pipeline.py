from vector_db import VectorDB
from chunking import Chunking
from embeddings import Embeddings
from openai import OpenAI
from pathlib import Path

class RAGPipeline():

    # Initialise our vector db, our chunker, and our embedder.
    def __init__(self, vector_db: VectorDB, chunker: Chunking, embedder: Embeddings):
        self.vector_db = vector_db
        self.chunker = chunker
        self.embedder = embedder
    
    # Populates the vector database.
    def index_text(self, text: str) -> None:
        chunks = self.chunker.chunk_text(text)

        # Store each chunk and its embedding.
        for chunk in chunks:
            chunk_embedding = self.embedder.embed(chunk)
            self.vector_db.add_chunk(chunk, chunk_embedding)
    
    def response(self, query: str, num_retrieve: int) -> str:

        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Retrieve num_retrieve most similar chunks. 
        most_similar = self.vector_db.search(query_embedding, num_retrieve)

        # Just return the chunk text, not the similarity
        retrieved_texts = [chunk for _, chunk in most_similar]

        # Combined most similar chunks acts as model context.
        context = "\n\n --- \n\n".join(retrieved_texts)

        # Construct user query
        user_query = f"""
        Use the following retrieved context to answer the question. 

        question:
        {query}

        context:
        {context}

        Answer:
        """

        # Generate the response from the LLM
        client = OpenAI()
        response = client.responses.create(
            model = "gpt-5.4-nano",
            instructions = (
                    "You're a helpful assistant. Answer only using the provided context. "
                    "If the answer is not in the context, say you do not know."
            ),
            input=user_query
        )

        # Return the response from the LLM
        return response.output_text
    
if __name__ == "__main__":

    # initialise vector database, chunker, and embedder
    vector_db = VectorDB()
    chunker = Chunking()
    embedder = Embeddings()
    # pass them to our RAG pipeline
    rag_pipeline = RAGPipeline(vector_db, chunker, embedder)

    # Specify the path to the text we want to index in the vector DB.
    file = Path("../data/weighted_mean_derivation.md")
    text = file.read_text(encoding="utf-8")

    # Populate our vector database with this text
    rag_pipeline.index_text(text)

    # return the models response.
    answer = rag_pipeline.response("What is the final estimator for the weighted mean?", text, 4)
    print(answer)




