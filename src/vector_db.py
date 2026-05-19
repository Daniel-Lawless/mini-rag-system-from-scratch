import heapq
import numpy as np

class VectorDB:

    def __init__(self):
        # Store each chunk's normalised embedding and original text.
        self.chunk_vectors = []
        self.chunks = []

    def add_chunk(self, chunk: str, vector: np.ndarray) -> None:

        self.chunk_vectors.append(vector)
        self.chunks.append(chunk)

    def search(self, query: np.ndarray, k: int) -> list[tuple[float, str]]:
        # Min-heap storing the current top-k most similar chunks.
        heap = []

        # Compare the query against every stored chunk vector.
        for i, (chunk, vector) in enumerate(zip(self.chunks, self.chunk_vectors)):
            # Since both vectors are normalised, dot product = cosine similarity.
            similarity = query @ vector

            # Include i as a tie-breaker if two similarities are equal.
            item = (similarity, i, chunk)

            # Fill the heap until it contains k chunks.
            if len(heap) < k:
                heapq.heappush(heap, item)

            # If this chunk is better than the worst current top-k result,
            # replace the worst result.
            elif similarity > heap[0][0]:
                heapq.heapreplace(heap, item)

        # Return chunks from most similar to least similar.
        return [(similarity, chunk) for similarity, i, chunk in sorted(heap, reverse=True)]