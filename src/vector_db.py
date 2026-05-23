import heapq
import numpy as np

class VectorDB:

    def __init__(self):
        # Store each chunk's normalised embedding and original text.
        self.records = []

    # Adds a record to the vector database
    def add_record(self, chunk: str, embedding: np.ndarray, source_file: str, chunk_index: int) -> None:

        record = {
            "chunk" : chunk,
            "embedding" : embedding,
            "metadata" : {
                "source_file" : source_file,
                "chunk_index" : chunk_index
            }
        }

        self.records.append(record)

    # Returns the top k most similar records in our vector database to the user query.
    def search(self, query: np.ndarray, k: int) -> list[tuple[float, dict]]:
        # Min-heap storing the current top-k most similar chunks.
        heap = []

        # Compare the query against every stored embedding.
        for index, record in enumerate(self.records):
            # Since both vectors are normalised, dot product = cosine similarity.
            similarity = query @ record["embedding"]

            # Include i as a tie-breaker if two similarities are equal.
            item = (similarity, index, record)

            # Fill the heap until it contains k chunks.
            if len(heap) < k:
                heapq.heappush(heap, item)

            # If this chunk is better than the worst current top-k result,
            # replace the worst result.
            elif similarity > heap[0][0]:
                heapq.heapreplace(heap, item)

        # Return chunks from most similar to least similar.
        return [(similarity, record) for similarity, index, record in sorted(heap, reverse=True)]
