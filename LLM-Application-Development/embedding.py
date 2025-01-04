from typing import List
from sentence_transformers import SentenceTransformer


class LocalEmbedding:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model)

    def get_embedding_model(self):
        return self._model

    def embed_query(self, sentences) -> List:
        embeddings = self._model.encode(sentences)
        print("Total Embeddings -", len(embeddings), embeddings[0])
        return embeddings

