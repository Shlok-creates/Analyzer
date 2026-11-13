"""ChromaDB Vector Store pipeline and local embedding service.

Manages persistent storage, vector indexing with all-MiniLM-L6-v2, and similarity
retrieval for campus circular chunks. Supports startup model preloading for instant latency.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger("vector_service")

# Module-level shared instances to guarantee zero cold-start reloads
_GLOBAL_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_GLOBAL_CHROMA_CLIENT: Optional[chromadb.ClientAPI] = None
_GLOBAL_COLLECTION = None

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def get_resolved_chroma_dir(raw_dir: Optional[str] = None) -> str:
    """Ensure ChromaDB persistence directory is always absolute and points to analyser-backend/data/chroma."""
    path = Path(raw_dir or settings.CHROMA_PERSIST_DIR)
    if not path.is_absolute():
        path = (_BACKEND_DIR / path).resolve()
    return str(path)


def preload_models() -> None:
    """Preload SentenceTransformer and ChromaDB persistent client at application startup.

    Guarantees that user requests and page loads execute with instant subsecond latency.
    """
    global _GLOBAL_EMBEDDING_MODEL, _GLOBAL_CHROMA_CLIENT, _GLOBAL_COLLECTION
    logger.info("Preloading embedding model ('all-MiniLM-L6-v2') at startup...")
    if _GLOBAL_EMBEDDING_MODEL is None:
        _GLOBAL_EMBEDDING_MODEL = SentenceTransformer(VectorStoreService.EMBEDDING_MODEL_NAME)

    if _GLOBAL_CHROMA_CLIENT is None:
        persist_dir = get_resolved_chroma_dir()
        os.makedirs(persist_dir, exist_ok=True)
        _GLOBAL_CHROMA_CLIENT = chromadb.PersistentClient(path=persist_dir)
        _GLOBAL_COLLECTION = _GLOBAL_CHROMA_CLIENT.get_or_create_collection(
            name=VectorStoreService.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    logger.info("Vector model and ChromaDB client preloaded successfully.")


class VectorStoreService:
    """Service managing local embedding generation and ChromaDB vector operations."""

    COLLECTION_NAME = "campus_circulars"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        """Initialize the vector store service with persistent directory path.

        Args:
            persist_directory: Optional directory path overriding settings.CHROMA_PERSIST_DIR.
        """
        self.persist_directory = get_resolved_chroma_dir(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

    @property
    def client(self) -> chromadb.ClientAPI:
        """Shared persistent ChromaDB client."""
        global _GLOBAL_CHROMA_CLIENT
        if _GLOBAL_CHROMA_CLIENT is None:
            _GLOBAL_CHROMA_CLIENT = chromadb.PersistentClient(path=self.persist_directory)
        return _GLOBAL_CHROMA_CLIENT

    @property
    def collection(self):
        """Shared ChromaDB collection configured with cosine similarity."""
        global _GLOBAL_COLLECTION
        if _GLOBAL_COLLECTION is None:
            _GLOBAL_COLLECTION = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return _GLOBAL_COLLECTION

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Shared SentenceTransformer model ('all-MiniLM-L6-v2')."""
        global _GLOBAL_EMBEDDING_MODEL
        if _GLOBAL_EMBEDDING_MODEL is None:
            _GLOBAL_EMBEDDING_MODEL = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
        return _GLOBAL_EMBEDDING_MODEL

    def upsert_document_chunks(
        self,
        doc_id: int,
        filename: str,
        category: str,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> None:
        """Generate embeddings in batches of 32 and upsert circular chunks into ChromaDB in a single batch call.

        Args:
            doc_id: Primary key of the Document record in SQL database.
            filename: Source PDF filename.
            category: Document administrative category (e.g., Exams, Fees, Hostels).
            chunks: List of chunk dictionaries containing 'text', 'chunk_id', and 'page_number'.
            batch_size: Batch size for model.encode (default: 32).
        """
        if not chunks:
            return

        texts: List[str] = [chunk["text"] for chunk in chunks]

        # Fast batch embedding with batch_size=32 in a single optimized pass
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        ).tolist()

        ids: List[str] = [
            f"doc_{doc_id}_{chunk.get('chunk_id', idx)}"
            for idx, chunk in enumerate(chunks)
        ]

        metadatas: List[Dict[str, Any]] = [
            {
                "doc_id": doc_id,
                "circular_id": doc_id,
                "filename": filename,
                "category": category,
                "page_number": int(chunk.get("page_number", 1)),
                "chunk_id": str(chunk.get("chunk_id", "")),
            }
            for chunk in chunks
        ]

        # Insert all chunks and embeddings into ChromaDB in a single batch call
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query_similar_chunks(
        self,
        query: str,
        n_results: int = 4,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search against indexed circular chunks using cosine similarity.

        Args:
            query: Natural language query submitted by user.
            n_results: Number of nearest neighbor chunks to retrieve (default: 4).
            category_filter: Optional category string filter to restrict search space.

        Returns:
            List of matching chunk dictionaries containing id, text, metadata, and distance.
        """
        if not query or not query.strip():
            return []

        query_embedding = self.embedding_model.encode(
            [query.strip()],
            show_progress_bar=False,
        ).tolist()

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": n_results,
        }

        if category_filter:
            query_kwargs["where"] = {"category": category_filter}

        results = self.collection.query(**query_kwargs)

        formatted_chunks: List[Dict[str, Any]] = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            doc_list = results["documents"][0]
            meta_list = (
                results["metadatas"][0]
                if results.get("metadatas")
                else [{}] * len(doc_list)
            )
            id_list = results["ids"][0] if results.get("ids") else [""] * len(doc_list)
            dist_list = (
                results["distances"][0]
                if results.get("distances")
                else [None] * len(doc_list)
            )

            for idx, text in enumerate(doc_list):
                metadata = meta_list[idx] if idx < len(meta_list) else {}
                chunk_id = id_list[idx] if idx < len(id_list) else ""
                distance = dist_list[idx] if idx < len(dist_list) else None

                formatted_chunks.append(
                    {
                        "id": chunk_id,
                        "text": text,
                        "filename": metadata.get("filename", ""),
                        "page_number": metadata.get("page_number", 1),
                        "category": metadata.get("category", ""),
                        "doc_id": metadata.get("doc_id"),
                        "visibility": metadata.get("visibility", "public"),
                        "distance": distance,
                        "metadata": metadata,
                    }
                )

        return formatted_chunks

    def delete_document_chunks(self, doc_id: int, filename: Optional[str] = None) -> None:
        """Remove all indexed chunks belonging to a specific document across ID and filename keys.

        Args:
            doc_id: Primary key of the Document record whose chunks are to be purged.
            filename: Optional source filename to purge matching filename tags.
        """
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            logger.debug("ChromaDB delete by doc_id warning: %s", e)
        try:
            self.collection.delete(where={"circular_id": doc_id})
        except Exception as e:
            logger.debug("ChromaDB delete by circular_id warning: %s", e)
        if filename:
            try:
                self.collection.delete(where={"filename": filename})
            except Exception as e:
                logger.debug("ChromaDB delete by filename warning: %s", e)
