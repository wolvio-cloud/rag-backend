import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import Settings

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def store_chunks(
        self,
        document_id: str,
        file_name: str,
        chunks: list[str],
        embeddings: list[list[float]],
        page_numbers: list[Optional[int]],
        upload_date: str,
    ) -> None:
        if not chunks:
            logger.warning("No chunks to store for document %s", document_id)
            return

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for index, (chunk_text, page_number) in enumerate(zip(chunks, page_numbers)):
            chunk_id = f"chunk_{index + 1}"
            ids.append(f"{document_id}_{chunk_id}")
            metadatas.append(
                {
                    "document_id": document_id,
                    "file_name": file_name,
                    "page_number": page_number if page_number is not None else -1,
                    "chunk_id": chunk_id,
                    "upload_date": upload_date,
                }
            )

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Stored %s chunks for document %s", len(chunks), document_id)

    def delete_document_chunks(self, document_id: str) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as exc:
            logger.warning("Failed to delete Chroma chunks for %s: %s", document_id, exc)

    def search(self, query_embedding: list[float], top_k: int) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    @staticmethod
    def build_page_number(page_value: Any) -> Optional[int]:
        if page_value is None:
            return None
        if isinstance(page_value, int) and page_value > 0:
            return page_value
        return None
