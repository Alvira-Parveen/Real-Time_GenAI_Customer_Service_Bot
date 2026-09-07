import os
from typing import List, Dict, Any
from .arxiv_loader import ArxivLoader
from ..knowledge_base.embeddings import EmbeddingManager
from ..knowledge_base.vector_store import VectorStore

class ArxivPaperSearch:
    """Semantic vector retrieval engine for Computer Science arXiv papers."""

    def __init__(
        self,
        data_path: str = "data/arxiv/arxiv_cs_papers.json",
        index_dir: str = "vectorstores/arxiv_index"
    ):
        self.data_path = data_path
        self.index_dir = index_dir
        self.loader = ArxivLoader(data_path)
        self.embeddings = EmbeddingManager()
        self.vector_store = VectorStore(persist_dir=index_dir, embedding_dim=self.embeddings.dimension)

        if self.vector_store.total_chunks == 0:
            self.build_index()

    def build_index(self, force_reload: bool = False):
        """Indexes papers into the vector store."""
        if not force_reload and self.vector_store.total_chunks > 0:
            return

        papers = self.loader.load_papers()
        if not papers:
            print("No arXiv papers found to index.")
            return

        print(f"Indexing {len(papers)} Computer Science arXiv papers...")
        chunks = []
        texts_to_embed = []

        for p in papers:
            chunk_dict = {
                "chunk_id": p["id"],
                "text": p["searchable_text"],
                "source": f"arXiv:{p['id']}",
                "title": p["title"],
                "authors": p["authors_display"],
                "categories": p["categories"],
                "published": p["published"],
                "abstract": p["abstract"],
                "url": p["url"],
                "pdf_url": p["pdf_url"]
            }
            chunks.append(chunk_dict)
            texts_to_embed.append(f"{p['title']}. {p['abstract']}")

        self.vector_store.clear()
        embeddings = self.embeddings.encode(texts_to_embed)
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save()
        print(f"Successfully indexed {self.vector_store.total_chunks} arXiv papers.")

    def search_papers(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Finds most relevant research papers based on semantic similarity."""
        query_vec = self.embeddings.encode(query)
        return self.vector_store.search(query_vec, top_k=top_k)

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Returns all indexed papers for explorer view."""
        return self.loader.load_papers()
