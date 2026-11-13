"""Document and PDF processing and chunking service.

Maintains backward-compatibility with existing PDFService callers while
delegating unified processing to DocumentProcessor.
"""

from typing import Any, Dict, List, Optional
from app.services.document_processor import DocumentProcessor


class PDFService(DocumentProcessor):
    """Backward-compatible PDFService delegating to DocumentProcessor."""

    def extract_chunks_from_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        category: str = "General",
    ) -> List[Dict[str, Any]]:
        """Extract chunks for PDF (and other formats) using unified DocumentProcessor."""
        chunks, _ = self.extract_chunks(
            file_bytes=file_bytes,
            filename=filename,
            category=category,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return chunks
