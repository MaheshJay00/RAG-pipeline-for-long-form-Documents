import faiss
import numpy as np
import json

class VectorStore:
    def __init__(self, vector_dim=768, index_file="output/faiss_index.bin", embeddings_file="output/embeddings.json"):
        """
        Initializes the FAISS vector database and loads document texts.
        :param vector_dim: Embedding dimension (default: 768 for mpnet-base-v2)
        :param index_file: Path to FAISS index
        :param embeddings_file: Path to saved embeddings
        """
        self.vector_dim = vector_dim
        self.index_file = index_file
        self.embeddings_file = embeddings_file
        self.index = faiss.IndexFlatL2(vector_dim)  # Flat L2 index for similarity search
        self.texts = []  # Store document texts

        

        self.load_index()
        self.load_embeddings()
    
    def load_index(self):
        """Loads the FAISS index from a file."""
        try:
            self.index = faiss.read_index(self.index_file)
            print(f"✅ FAISS index loaded from {self.index_file}")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")

    def load_embeddings(self):
        """Loads embeddings and corresponding texts from a JSON file."""
        try:
            with open(self.embeddings_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            embeddings = np.array(data["embeddings"], dtype=np.float32)
            self.texts = data["texts"]  # ✅ Store document texts in self.texts

            # Load FAISS index
            self.index.add(embeddings)
            print(f"✅ Loaded {len(embeddings)} embeddings into FAISS.")

        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")

    def search(self, query_embedding, top_k=5):
        """
        Searches FAISS for the most similar document sections.
        :param query_embedding: The vectorized query
        :param top_k: Number of top results to retrieve
        :return: List of top matching texts
        """
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)

        # ✅ Ensure indices are valid before accessing self.texts
        results = [self.texts[idx] for idx in indices[0] if 0 <= idx < len(self.texts)]

        return results

# Example Usage
if __name__ == "__main__":
    vector_store = VectorStore()
    query_embedding = np.random.rand(768)  # Replace with real embedding
    results = vector_store.search(query_embedding, top_k=3)
    print("\n🔹 Top Matching Documents:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res.strip()}")
