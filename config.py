import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_storage_bucket: str = "contracts"
    anthropic_api_key: str = ""
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "contracts"
    max_upload_size_mb: int = 100
    allowed_extensions: set[str] = {".pdf", ".jpg", ".jpeg", ".png"}
    cors_origins: str = "http://localhost:3000,http://localhost:5173,https://rag-contract-frontend-3437a857a700.herokuapp.com"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    ocr_min_text_length: int = 100
    rag_top_k: int = 5
    claude_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
