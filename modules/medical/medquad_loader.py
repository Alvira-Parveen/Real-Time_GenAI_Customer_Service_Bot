import os
import json
from typing import List, Dict, Any

class MedQuADLoader:
    """Loads and preprocesses the official MedQuAD dataset."""

    def __init__(self, data_path: str = "datasets/medquad/medquad_qa.json"):
        self.data_path = data_path

    def load_qa_pairs(self) -> List[Dict[str, Any]]:
        """Loads and cleans MedQuAD question-answer pairs."""
        if not os.path.exists(self.data_path):
            print(f"MedQuAD data file {self.data_path} not found.")
            return []

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        cleaned_pairs = []
        for item in raw_data:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            focus = item.get("focus", "General Medicine").strip()
            if q and a:
                # Clean up XML artifacts or repetitive whitespaces
                a_clean = " ".join(a.split())
                cleaned_pairs.append({
                    "id": item.get("id", ""),
                    "source": item.get("source", "NIH"),
                    "focus": focus,
                    "qtype": item.get("qtype", "information"),
                    "question": q,
                    "answer": a_clean,
                    "searchable_text": f"Focus: {focus}\nQuestion: {q}\nAnswer: {a_clean}"
                })

        return cleaned_pairs
