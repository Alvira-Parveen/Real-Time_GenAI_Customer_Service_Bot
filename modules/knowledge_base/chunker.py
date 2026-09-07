from typing import List, Dict, Any

class TextChunker:
    """Splits documents into overlapping chunks with metadata preservation."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 75):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Splits a single text into overlapping chunks."""
        metadata = metadata or {}
        if not text or not text.strip():
            return []

        # Split into paragraphs first to respect logical blocks
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(current_chunk, chunk_index, metadata))
                    chunk_index += 1
                    # Keep overlap
                    overlap_chars = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = (overlap_chars + "\n\n" + para).strip() if overlap_chars else para
                else:
                    # Paragraph is longer than chunk_size, hard break
                    start = 0
                    while start < len(para):
                        end = start + self.chunk_size
                        sub_text = para[start:end]
                        chunks.append(self._create_chunk_dict(sub_text, chunk_index, metadata))
                        chunk_index += 1
                        start += (self.chunk_size - self.chunk_overlap)
                    current_chunk = ""

        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(current_chunk.strip(), chunk_index, metadata))

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunks a collection of document dicts."""
        all_chunks = []
        for doc in documents:
            meta = {
                "source": doc.get("source", "unknown"),
                "filepath": doc.get("filepath", ""),
                "file_hash": doc.get("file_hash", ""),
                "modified_time": doc.get("modified_time", 0)
            }
            doc_chunks = self.chunk_text(doc.get("content", ""), metadata=meta)
            all_chunks.extend(doc_chunks)
        return all_chunks

    def _create_chunk_dict(self, text: str, chunk_index: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chunk_id": f"{metadata.get('source', 'doc')}_chunk_{chunk_index}",
            "text": text,
            "chunk_index": chunk_index,
            "source": metadata.get("source", "unknown"),
            "filepath": metadata.get("filepath", ""),
            "file_hash": metadata.get("file_hash", ""),
            "char_len": len(text)
        }
