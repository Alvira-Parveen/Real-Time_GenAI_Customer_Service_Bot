import os
import hashlib
from typing import List, Dict, Any
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

class DocumentLoader:
    """Loads text, markdown, and PDF documents with metadata and checksums."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.markdown', '.pdf', '.csv', '.json'}
    
    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """Computes SHA-256 hash of a file for change detection."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def load_file(cls, filepath: str) -> Dict[str, Any]:
        """Loads a single file and extracts its text and metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        file_stats = os.stat(filepath)
        file_hash = cls.compute_file_hash(filepath)
        filename = os.path.basename(filepath)
        
        content = ""
        if ext in {'.txt', '.md', '.markdown', '.csv', '.json'}:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif ext == '.pdf':
            if PdfReader is None:
                raise ImportError("pypdf is required to read PDF files. Install it with 'pip install pypdf'.")
            reader = PdfReader(filepath)
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(f"--- Page {i+1} ---\n{text}")
            content = "\n\n".join(pages_text)
            
        return {
            "source": filename,
            "filepath": os.path.abspath(filepath),
            "content": content,
            "char_count": len(content),
            "file_size": file_stats.st_size,
            "modified_time": file_stats.st_mtime,
            "file_hash": file_hash
        }

    @classmethod
    def load_directory(cls, dirpath: str) -> List[Dict[str, Any]]:
        """Scans directory and loads all supported documents."""
        documents = []
        if not os.path.exists(dirpath):
            return documents
            
        for root, _, files in os.walk(dirpath):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in cls.SUPPORTED_EXTENSIONS:
                    filepath = os.path.join(root, file)
                    try:
                        doc = cls.load_file(filepath)
                        documents.append(doc)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        return documents
