import torch
from sentence_transformers import SentenceTransformer
import json

class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
       
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)

    def generate_embedding(self, text):
        
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.cpu().detach().numpy()

    def process_document(self, input_file, output_file):
        
        with open(input_file, "r", encoding="utf-8") as f:
            text_lines = f.readlines()

        embeddings = [self.generate_embedding(line.strip()).tolist() for line in text_lines if line.strip()]

        # Save embeddings as JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"embeddings": embeddings, "texts": text_lines}, f, indent=4)

        print(f"Generated {len(embeddings)} embeddings and saved to {output_file}")

# Example usage
if __name__ == "__main__":
    embedder = TextEmbedder()
    
    # Generate embeddings for preprocessed text
    embedder.process_document("output/structured_text.txt", "output/embeddings.json")
