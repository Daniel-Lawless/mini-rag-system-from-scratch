class Chunking:

    def chunk_text(self, text: str, chunk_size: int = 20, overlap: int = 5) -> list[str]:

        # Overlap cannot be greater than or equal than chunk_size
        if overlap >= chunk_size:
            raise ValueError("Overlap cannot be larger than chunk size")

        words = text.split() # Splits our text into a list of words
        start = 0            # Starts at the first word
        chunks = []          # Stores each chunk

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end]) # ["Daniel", "is", "happy"] -> "Daniel is happy"
            chunks.append(chunk)

            # Overlap helps avoid missing context, especially when an important sentence,
            # definition, equation, or explanation is split between chunks.
            start += chunk_size - overlap 
        
        # Return the text split into chunks
        return chunks