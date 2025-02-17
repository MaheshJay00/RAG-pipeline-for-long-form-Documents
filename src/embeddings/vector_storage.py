import faiss 
import json
import numpy as np
from embeddings.embedder import TextEmbedder
class VectorStore:
    def __init__(self, vector_dim = 768, index_file = "output/faiss_index.bin"):
        
        self.vector_dim = vector_dim
        self.index_file = index_file

        self.index = faiss.IndexFlatL2(vector_dim)

    def load_embeddings(self,embedding_file):
        with open(embedding_file, 'r', endcoding = 'utf-8') as f:
            data = json.load(f)

        self.texts = data['texts']
        embeddings = np.array(data['embeddings'], dtype = np.float32)

        self.index.add(embeddings)

    def save_index(self):
        faiss.write_index(self.index,self.index_file)
        print(f"FAISS index saved to{self.index_file}")
    
    def load_index(self):

        self.index = faiss.read_index(self.index_file)
        print(f"FAISS loaded from{self.index_file}")

    def search(self,query_embedding,top_k =3):
        query_vector = np.array([query_embedding],dtype = np.float32)
        distances,indices = self.index.search(query_vector,top_k)

        results = [self.texts[idx] for idx in indices[0] if idx<len(self.texts)]
        return results
    
if __name__ == "main":
    vector_store = VectorStore()
    vector_store.load_embeddings('output/embeddings.json')

    vector_store.save_index()

    embedder = TextEmbedder()
    query_embedding = embedder.generate_embedding("Find contract termination clauses.")
    results = vector_store.search(query_embedding, top_k=3)

    print("Top matching documents:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res.strip()}")
