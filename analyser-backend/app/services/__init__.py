"""Analyser Backend Services Package.

Exports core pipeline services:
- PDFService: Ingestion and text chunking with page provenance.
- VectorStoreService: ChromaDB persistent vector storage with SentenceTransformers embeddings.
- RAGEngine: Grounded answer synthesis via Groq LLaMA-3.1 with strict campus citations.
"""

from app.services.pdf_service import PDFService
from app.services.vector_service import VectorStoreService, preload_models
from app.services.rag_engine import RAGEngine

__all__ = [
    "PDFService",
    "VectorStoreService",
    "preload_models",
    "RAGEngine",
]

