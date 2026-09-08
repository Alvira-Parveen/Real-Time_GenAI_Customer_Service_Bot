import os
import json
import time
from typing import Dict, Any, List, Tuple
from .loader import DocumentLoader
from .chunker import TextChunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStore

class KnowledgeBaseManager:
    """Orchestrates dynamic loading, change detection, chunking, and index updating."""
    
    def __init__(
        self,
        docs_dir: str = "datasets/customer_service",
        index_dir: str = "vectorstores/kb_index",
        state_file: str = "vectorstores/kb_index/kb_state.json"
    ):
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        self.state_file = state_file
        
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=75)
        self.embeddings = EmbeddingManager()
        self.vector_store = VectorStore(persist_dir=index_dir, embedding_dim=self.embeddings.dimension)
        
        self.doc_registry = self._load_state()
        self.last_periodic_check = 0
        self.last_check_timestamp = "Initial run"

    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        """Loads tracked document checksums and modification timestamps."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_state(self):
        """Saves current document registry state."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.doc_registry, f, indent=2)

    def scan_for_changes(self) -> Tuple[List[str], List[str], List[str]]:
        """Scans the documents directory and compares against the registry.
        Returns: (new_files, modified_files, deleted_files)
        """
        current_files = {}
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in DocumentLoader.SUPPORTED_EXTENSIONS:
                    filepath = os.path.join(root, file)
                    mtime = os.path.getmtime(filepath)
                    current_files[file] = {
                        "filepath": filepath,
                        "mtime": mtime
                    }

        new_files = []
        modified_files = []
        for filename, info in current_files.items():
            if filename not in self.doc_registry:
                new_files.append(filename)
            else:
                current_hash = DocumentLoader.compute_file_hash(info["filepath"])
                if current_hash != self.doc_registry[filename].get("hash"):
                    modified_files.append(filename)

        deleted_files = [
            filename for filename in self.doc_registry
            if filename not in current_files
        ]

        return new_files, modified_files, deleted_files

    def update_knowledge_base(self, force_reload: bool = False) -> Dict[str, Any]:
        """Runs the dynamic update pipeline:
        Detect Changes -> Extract -> Chunk -> Embed -> Update VectorStore -> Persist
        """
        new_files, modified_files, deleted_files = self.scan_for_changes()
        
        # If force_reload or changes detected or index is empty:
        has_changes = bool(new_files or modified_files or deleted_files)
        needs_update = force_reload or has_changes or (self.vector_store.total_chunks == 0)

        if not needs_update:
            return {
                "status": "up_to_date",
                "message": "Knowledge base is already up to date. No changes detected.",
                "new_files": [],
                "modified_files": [],
                "deleted_files": [],
                "total_documents": self.vector_store.total_documents,
                "total_chunks": self.vector_store.total_chunks,
                "last_updated": self.vector_store.last_updated
            }

        print(f"Updating Knowledge Base: new={new_files}, modified={modified_files}, deleted={deleted_files}")

        # Full or incremental re-index
        documents = self.loader.load_directory(self.docs_dir)
        all_chunks = self.chunker.chunk_documents(documents)

        # Clear and rebuild vector store
        self.vector_store.clear()
        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            chunk_vectors = self.embeddings.encode(texts)
            self.vector_store.add_chunks(all_chunks, chunk_vectors)
            self.vector_store.save()

        # Update registry state
        new_registry = {}
        for doc in documents:
            new_registry[doc["source"]] = {
                "filepath": doc["filepath"],
                "hash": doc["file_hash"],
                "mtime": doc["modified_time"],
                "char_count": doc["char_count"]
            }
        self.doc_registry = new_registry
        self._save_state()

        return {
            "status": "updated",
            "message": f"Knowledge base successfully updated with {len(documents)} documents and {len(all_chunks)} chunks.",
            "new_files": new_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
            "total_documents": self.vector_store.total_documents,
            "total_chunks": self.vector_store.total_chunks,
            "last_updated": self.vector_store.last_updated,
            "indexed_sources": self.vector_store.indexed_sources
        }

    def add_document_from_text(self, filename: str, content: str) -> Dict[str, Any]:
        """Saves a new text document and triggers an immediate update."""
        if not filename.endswith(('.txt', '.md')):
            filename += '.txt'
        filepath = os.path.join(self.docs_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return self.update_knowledge_base()

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Embeds user query and retrieves top_k relevant chunks."""
        query_vector = self.embeddings.encode(query_text)
        return self.vector_store.search(query_vector, top_k=top_k)

    def check_periodic_update(self, interval_seconds: int = 60) -> Dict[str, Any]:
        """Periodically checks for new/modified/deleted files without requiring manual intervention.
        Triggers update automatically if the configured interval has elapsed and changes are detected.
        """
        now = time.time()
        elapsed = now - getattr(self, "last_periodic_check", 0)
        if elapsed < interval_seconds and self.vector_store.total_chunks > 0:
            return {
                "status": "waiting_interval",
                "seconds_until_next_check": int(interval_seconds - elapsed),
                "last_checked": getattr(self, "last_check_timestamp", "Just now"),
                "periodic_triggered": False
            }

        self.last_periodic_check = now
        self.last_check_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        new_files, modified_files, deleted_files = self.scan_for_changes()
        if new_files or modified_files or deleted_files or (self.vector_store.total_chunks == 0):
            res = self.update_knowledge_base()
            res["periodic_triggered"] = True
            res["last_checked"] = self.last_check_timestamp
            return res
        
        return {
            "status": "up_to_date",
            "periodic_triggered": False,
            "message": "Periodic check: Knowledge base is up to date (no changes detected).",
            "last_checked": self.last_check_timestamp,
            "total_documents": self.vector_store.total_documents,
            "total_chunks": self.vector_store.total_chunks
        }

    def get_stats(self) -> Dict[str, Any]:
        """Returns knowledge base summary statistics for the Streamlit dashboard."""
        return {
            "total_documents": self.vector_store.total_documents,
            "total_chunks": self.vector_store.total_chunks,
            "last_updated": self.vector_store.last_updated,
            "last_check_timestamp": getattr(self, "last_check_timestamp", "Active"),
            "indexed_sources": self.vector_store.indexed_sources
        }

