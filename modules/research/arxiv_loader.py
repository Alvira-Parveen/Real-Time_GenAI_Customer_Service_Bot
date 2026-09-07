import os
import json
from typing import List, Dict, Any

class ArxivLoader:
    """Loads and validates the Computer Science arXiv papers subset."""

    def __init__(self, data_path: str = "data/arxiv/arxiv_cs_papers.json"):
        self.data_path = data_path

    def load_papers(self) -> List[Dict[str, Any]]:
        """Loads papers and ensures complete metadata fields."""
        if not os.path.exists(self.data_path):
            print(f"arXiv dataset not found at {self.data_path}")
            return []

        with open(self.data_path, "r", encoding="utf-8") as f:
            papers = json.load(f)

        validated = []
        for p in papers:
            title = p.get("title", "").strip()
            abstract = p.get("abstract", "").strip()
            if title and abstract:
                authors = p.get("authors", [])
                authors_str = ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")
                categories = p.get("categories", ["cs.AI"])
                paper_id = p.get("id", "")
                
                validated.append({
                    "id": paper_id,
                    "title": title,
                    "authors": authors,
                    "authors_display": authors_str,
                    "categories": categories,
                    "published": p.get("published", "N/A"),
                    "abstract": abstract,
                    "url": p.get("url", f"https://arxiv.org/abs/{paper_id}"),
                    "pdf_url": p.get("pdf_url", f"https://arxiv.org/pdf/{paper_id}.pdf"),
                    "searchable_text": f"Title: {title}\nCategories: {', '.join(categories)}\nAbstract: {abstract}"
                })

        return validated
