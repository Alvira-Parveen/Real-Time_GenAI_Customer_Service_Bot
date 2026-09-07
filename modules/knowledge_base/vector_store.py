import os
import json
import numpy as np
import datetime
from typing import List, Dict, Any, Optional

class VectorStore:
    """Persistent vector store with FAISS backend and metadata management."""
    
    def __init__(self, persist_dir: str = "vectorstores/kb_index", embedding_dim: int = 384):
        self.persist_dir = persist_dir
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "last_updated": "Never",
            "indexed_sources": []
        }
        os.makedirs(self.persist_dir, exist_ok=True)
        self._init_backend()
        self.load()

    def _init_backend(self):
        """Initializes FAISS IndexFlatIP (Inner Product = Cosine Sim on normalized vectors)."""
        try:
            import faiss
            self.faiss = faiss
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.backend = "faiss"
        except Exception as e:
            print(f"FAISS init notice ({e}). Using NumPy normalized dot-product matrix.")
            self.backend = "numpy"
            self.vectors = np.empty((0, self.embedding_dim), dtype=np.float32)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adds text chunks with embeddings and metadata to the vector store."""
        if not chunks or len(chunks) == 0:
            return

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        if self.backend == "faiss":
            self.index.add(embeddings)
        else:
            if self.vectors.shape[0] == 0:
                self.vectors = embeddings
            else:
                self.vectors = np.vstack([self.vectors, embeddings])

        self.chunks_metadata.extend(chunks)
        self._update_stats()

    def search(self, query_embedding: np.ndarray, top_k: int = 3, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Searches top_k most similar chunks for a given query embedding."""
        if self.total_chunks == 0:
            return []

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        results = []
        k = min(top_k, self.total_chunks)

        if self.backend == "faiss":
            distances, indices = self.index.search(query_embedding, k)
            for score, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks_metadata) and score >= min_score:
                    chunk = dict(self.chunks_metadata[idx])
                    chunk["score"] = float(score)
                    results.append(chunk)
        else:
            scores = np.dot(self.vectors, query_embedding.T).squeeze()
            if np.isscalar(scores):
                scores = np.array([scores])
            top_indices = np.argsort(scores)[::-1][:k]
            for idx in top_indices:
                score = float(scores[idx])
                if score >= min_score:
                    chunk = dict(self.chunks_metadata[idx])
                    chunk["score"] = score
                    results.append(chunk)

        return results

    def clear(self):
        """Clears all indexed vectors and metadata."""
        if self.backend == "faiss":
            self.index = self.faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.vectors = np.empty((0, self.embedding_dim), dtype=np.float32)
        self.chunks_metadata = []
        self._update_stats()

    def _update_stats(self):
        sources = list(set(c.get("source", "unknown") for c in self.chunks_metadata))
        self.stats = {
            "total_documents": len(sources),
            "total_chunks": len(self.chunks_metadata),
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "indexed_sources": sorted(sources)
        }

    def save(self):
        """Persists vector store and metadata to disk."""
        meta_path = os.path.join(self.persist_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "stats": self.stats,
                "chunks": self.chunks_metadata
            }, f, indent=2)

        if self.backend == "faiss":
            index_path = os.path.join(self.persist_dir, "index.faiss")
            self.faiss.write_index(self.index, index_path)
        else:
            vec_path = os.path.join(self.persist_dir, "vectors.npy")
            np.save(vec_path, self.vectors)

    def load(self) -> bool:
        """Loads vector store and metadata from disk if available."""
        meta_path = os.path.join(self.persist_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.stats = data.get("stats", self.stats)
                self.chunks_metadata = data.get("chunks", [])

            if self.backend == "faiss":
                index_path = os.path.join(self.persist_dir, "index.faiss")
                if os.path.exists(index_path):
                    self.index = self.faiss.read_index(index_path)
            else:
                vec_path = os.path.join(self.persist_dir, "vectors.npy")
                if os.path.exists(vec_path):
                    self.vectors = np.load(vec_path)
            return True
        except Exception as e:
            print(f"Error loading vector store from {self.persist_dir}: {e}")
            return False

    @property
    def total_chunks(self) -> int:
        return len(self.chunks_metadata)

    @property
    def total_documents(self) -> int:
        return self.stats.get("total_documents", 0)

    @property
    def last_updated(self) -> str:
        return self.stats.get("last_updated", "Never")

    @property
    def indexed_sources(self) -> List[str]:
        return self.stats.get("indexed_sources", [])
