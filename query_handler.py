import numpy as np
from utils.embeddings.embedder import TextEmbedder
from utils.embeddings.vector_store import VectorStore
import json
class QueryHandler:
    def __init__(self, vector_index_path="output/faiss_index.bin", embeddings_file="output/embeddings.json"):
        
        self.embedder = TextEmbedder()  # Load text embedder
        self.vector_store = VectorStore(index_file=vector_index_path)  # Load FAISS vector database
        self.vector_store.load_index()

        # Load document texts (corresponding to embeddings)
        with open(embeddings_file, "r", encoding="utf-8") as f:
            self.text_data = json.load(f)["texts"]

    def retrieve_documents(self, query, top_k=5):
       
        query_embedding = self.embedder.generate_embedding(query)
        matching_texts = self.vector_store.search(query_embedding, top_k=top_k)

        print(f"🔍 Query: {query}")
        print(f"📖 Retrieved Documents: {matching_texts}")

        return matching_texts

    def prepare_context(self, query, top_k=5):
        
        retrieved_docs = self.retrieve_documents(query, top_k=top_k)

        # Combine retrieved text sections into a context string
        context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        return prompt

# Example usage
if __name__ == "__main__":
    query_handler = QueryHandler()
    
    # Example user query
    query = "What is the customer id?"
    context = query_handler.prepare_context(query)

    print("🔹 Prepared Context for LLM:")
    print(context)
