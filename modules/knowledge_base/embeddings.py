import os
import numpy as np
from typing import List, Union

class EmbeddingManager:
    """Manages dense vector embeddings using SentenceTransformers with TF-IDF fallback."""
    
    _instance = None
    _model = None
    _model_type = "none"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_fallback_if_needed: bool = True):
        self.model_name = model_name
        self.use_fallback_if_needed = use_fallback_if_needed
        self._init_model()

    def _init_model(self):
        if EmbeddingManager._model is not None:
            self.model = EmbeddingManager._model
            self.model_type = EmbeddingManager._model_type
            return

        try:
            # Try to load SentenceTransformer
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}...")
            # Load with local-first timeout handling
            model = SentenceTransformer(self.model_name)
            EmbeddingManager._model = model
            EmbeddingManager._model_type = "sentence-transformer"
            self.model = model
            self.model_type = "sentence-transformer"
            print("Successfully loaded SentenceTransformer model.")
        except Exception as e:
            print(f"Notice: SentenceTransformer initialization ({e}). Using Sklearn TF-IDF / Hashing embedding engine.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            EmbeddingManager._model = vectorizer
            EmbeddingManager._model_type = "tfidf"
            self.model = vectorizer
            self.model_type = "tfidf"

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encodes text or list of texts into normalized embedding vectors."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        if not text_list:
            return np.empty((0, 384), dtype=np.float32)

        if self.model_type == "sentence-transformer":
            try:
                embeddings = self.model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings[0] if is_single else embeddings
            except Exception as e:
                print(f"SentenceTransformer encoding error: {e}. Fallback to TF-IDF.")
                self.model_type = "tfidf"

        # TF-IDF fallback:
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = [t if t.strip() else "empty document" for t in text_list]
        try:
            # If not fitted or dimension mismatch, fit transform
            if not hasattr(self.model, "vocabulary_") or self.model.vocabulary_ is None:
                matrix = self.model.fit_transform(corpus).toarray()
            else:
                try:
                    matrix = self.model.transform(corpus).toarray()
                except Exception:
                    matrix = self.model.fit_transform(corpus).toarray()
        except Exception:
            vec = TfidfVectorizer(max_features=384)
            matrix = vec.fit_transform(corpus).toarray()
            self.model = vec

        # Pad or truncate to 384 dimensions for consistency
        target_dim = 384
        current_dim = matrix.shape[1]
        if current_dim < target_dim:
            padding = np.zeros((matrix.shape[0], target_dim - current_dim), dtype=np.float32)
            matrix = np.hstack([matrix, padding])
        elif current_dim > target_dim:
            matrix = matrix[:, :target_dim]

        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (matrix / norms).astype(np.float32)

        return normalized[0] if is_single else normalized

    @property
    def dimension(self) -> int:
        return 384
