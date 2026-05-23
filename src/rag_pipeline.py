from vector_db import VectorDB
from chunking import Chunking
from embeddings import Embeddings
from openai import OpenAI
from pathlib import Path
import logging

class RAGPipeline():

    # Initialise our vector db, our chunker, and our embedder.
    def __init__(self, vector_db: VectorDB, chunker: Chunking, embedder: Embeddings):
        self.vector_db = vector_db
        self.chunker = chunker
        self.embedder = embedder
    
    # Populates the vector database.
    def index_text(self, text: str, source_file: str) -> None:

        # Split data into chunks
        chunks = self.chunker.chunk_text(text, chunk_size= 300, overlap=75)

        # Assign each chunk a chunk_id
        for chunk_index, chunk in enumerate(chunks):
            chunk_embedding = self.embedder.embed(chunk)
            self.vector_db.add_record(chunk = chunk,
                                      embedding = chunk_embedding,
                                      source_file = source_file,
                                      chunk_index = chunk_index
                                      )

    # Models response
    def response(self, query: str, num_retrieve: int) -> str:

        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Retrieve num_retrieve most similar chunks. 
        most_similar = self.vector_db.search(query_embedding, num_retrieve)

        retrieved_texts = []

        # Just return the chunk text, not the similarity
        for _, record in most_similar:
            source_file = record["metadata"]["source_file"]
            chunk_index = record["metadata"]["chunk_index"]
            chunk = record["chunk"]

            # To help with debugging:
            print()

            retrieved_texts.append(
                f"[Source file: {source_file}, chunk index: {chunk_index}]\n"
                f"{chunk}"
            )

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

    # Specify the path to the directory that contains the data we want to store in our vector DB.
    data_dir = Path("../data")

    # For each file in this directory, read its contents and index it. This populates our database.
    for file in sorted(data_dir.glob("*.md")):
        print(f"Indexing {file.name}...")
        text = file.read_text(encoding = "utf-8")
        rag_pipeline.index_text(text, source_file = file.name)

    # Once the DB is populated, return the models response.
    answer = rag_pipeline.response("What is the final probability for Buffons needle?", 10)
    print(answer)





