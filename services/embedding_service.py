import logging
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self._model = None

    def chunk_text(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.generate_embeddings([query])[0]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.settings.embedding_model)
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def estimate_page_for_chunk(self, chunk_text: str, page_texts: list[tuple[int, str]]) -> Optional[int]:
        normalized_chunk = chunk_text.strip().lower()
        if not normalized_chunk:
            return None

        for page_number, page_text in page_texts:
            if normalized_chunk[:120] in page_text.lower():
                return page_number

        snippet = normalized_chunk[:80]
        for page_number, page_text in page_texts:
            if snippet in page_text.lower():
                return page_number

        return page_texts[0][0] if page_texts else None
