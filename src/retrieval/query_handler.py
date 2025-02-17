import sys
import os
import numpy as np
import json
from embeddings.embedder import TextEmbedder
from embeddings.vector_storage import VectorStore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class QueryHandler:
    def __init__(self, vector_index_path = 'output/faiss_index.bin',embeddings_file = "output/embeddings.json"):
        
        self.embedder = TextEmbedder()
        self.vector_storage = VectorStore(index_file = vector_index_path)
        self.vector_store.load_index()

        with open(embeddings_file, 'r', encoding = 'utf-8') as f:
            self.text_data = json.load(f)['texts']

    def retrieve_documents(self, query, top_k =3):

        query_embedding = self.embedder.generate_embeddings(query)
        matching_texts = self.vector_storage.search(query_embedding, top_k = top_k)

        print(f"query : {query}")
        print(f"Retrieved docs:{matching_texts}")

    def prepare_context(self, query, top_k=3):

        retrieved_docs = self.retrieve_documents(query, top_k=top_k)
        context = "\n".join(retrieved_docs)

        prompt = f"Context:\n{context}\n\n Question:{query}\n\n Answer:"
 
        return prompt