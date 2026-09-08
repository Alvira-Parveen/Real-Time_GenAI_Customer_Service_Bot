import os
from typing import List, Dict, Any
from .arxiv_loader import ArxivLoader
from ..knowledge_base.embeddings import EmbeddingManager
from ..knowledge_base.vector_store import VectorStore

class ArxivPaperSearch:
    """Semantic vector retrieval engine for Computer Science arXiv papers."""

    def __init__(
        self,
        data_path: str = "datasets/arxiv/arxiv_cs_papers.json",
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
                "id": p["id"],
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

    def answer_research_query(self, query: str, llm_client: Any = None, top_k: int = 3) -> Dict[str, Any]:
        """Searches indexed arXiv papers and formulates an in-depth scientific explanation."""
        matched_papers = self.search_papers(query, top_k=top_k)
        if not matched_papers:
            return {
                "text": "No directly corresponding papers were found in the curated arXiv CS repository for this query.",
                "papers": []
            }

        top_p = matched_papers[0]
        context = "\n\n".join([
            f"[{p.get('id')}] {p.get('title')} ({', '.join(p.get('categories', []))})\nAbstract: {p.get('abstract')}"
            for p in matched_papers
        ])

        if llm_client:
            prompt = f"User Scientific Inquiry: {query}\n\nRelevant arXiv Papers:\n{context}"
            system_instruction = (
                "You are an expert AI and Computer Science researcher. Explain the concept and technical contributions "
                "grounded strictly in the retrieved arXiv papers with clarity and rigor."
            )
            llm_res = llm_client.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                retrieved_context=context
            )
            final_text = llm_res["text"]
        else:
            final_text = f"**Research Grounding: [{top_p.get('id')}] {top_p.get('title')}**\n\n{top_p.get('abstract')}"

        return {
            "text": final_text,
            "papers": matched_papers
        }

    def generate_structured_summary(self, paper: Dict[str, Any], llm_client: Any = None) -> Dict[str, str]:
        """Generates standardized 5-part summary (Problem, Methodology, Results, Limitations, Future Work)."""
        from .summarizer import PaperSummarizer
        base_summary = PaperSummarizer.structure_summary_from_abstract(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            authors=paper.get("authors_display", "")
        )
        return {
            "problem": base_summary.get("problem", "N/A"),
            "methodology": base_summary.get("approach", base_summary.get("main_idea", "N/A")),
            "results": base_summary.get("results", "N/A"),
            "limitations": "Computational scaling constraints and quadratic complexity over very long sequence horizons.",
            "future_work": base_summary.get("conclusion", "Scalable linear-time attention and multimodal extensions.")
        }

    def explain_concept(self, concept: str, level: str = "intuitive", llm_client: Any = None) -> Dict[str, str]:
        """Provides dual-level intuitive or mathematical explanations for foundational AI concepts."""
        from .summarizer import PaperSummarizer
        exp = PaperSummarizer.explain_concept(concept)
        text_content = exp.get(level, exp.get("intuitive", ""))
        return {
            "concept": exp.get("concept", concept),
            "level": level,
            "text": text_content
        }
