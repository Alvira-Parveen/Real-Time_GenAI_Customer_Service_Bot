import os
import json
from typing import List, Dict, Any
from .medquad_loader import MedQuADLoader
from ..knowledge_base.embeddings import EmbeddingManager
from ..knowledge_base.vector_store import VectorStore

class MedicalRetriever:
    """Indexes MedQuAD dataset and performs semantic medical Q&A retrieval."""

    def __init__(
        self,
        data_path: str = "data/medquad/medquad_qa.json",
        index_dir: str = "vectorstores/medquad_index"
    ):
        self.data_path = data_path
        self.index_dir = index_dir
        self.loader = MedQuADLoader(data_path)
        self.embeddings = EmbeddingManager()
        self.vector_store = VectorStore(persist_dir=index_dir, embedding_dim=self.embeddings.dimension)
        
        # Build index if empty
        if self.vector_store.total_chunks == 0:
            self.build_index()

    def build_index(self, force_reload: bool = False):
        """Builds or reloads the MedQuAD vector index."""
        if not force_reload and self.vector_store.total_chunks > 0:
            return

        qa_pairs = self.loader.load_qa_pairs()
        if not qa_pairs:
            print("No MedQuAD pairs found to index.")
            return

        print(f"Indexing {len(qa_pairs)} MedQuAD QA pairs...")
        chunks = []
        texts_to_embed = []
        
        for i, pair in enumerate(qa_pairs):
            chunk_dict = {
                "chunk_id": pair["id"] or f"medquad_{i}",
                "text": pair["searchable_text"],
                "source": f"MedQuAD ({pair['source']})",
                "focus": pair["focus"],
                "question": pair["question"],
                "answer": pair["answer"],
                "qtype": pair["qtype"]
            }
            chunks.append(chunk_dict)
            # Embed question + focus for higher semantic relevance
            texts_to_embed.append(f"{pair['focus']}: {pair['question']}")

        self.vector_store.clear()
        embeddings = self.embeddings.encode(texts_to_embed)
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save()
        print(f"Successfully indexed {self.vector_store.total_chunks} MedQuAD items.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top_k most relevant MedQuAD QA items."""
        query_vec = self.embeddings.encode(query)
        return self.vector_store.search(query_vec, top_k=top_k)
