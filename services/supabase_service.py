import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from postgrest.exceptions import APIError
from supabase import Client, create_client

from config import Settings

logger = logging.getLogger(__name__)

SELECT_COLUMNS_FULL = (
    "id, file_name, file_url, file_type, uniform_pdf_url, uniform_pdf_name, status, created_at"
)
SELECT_COLUMNS_LEGACY = "id, file_name, file_url, file_type, status, created_at"
DETAIL_COLUMNS_FULL = (
    "id, file_name, file_url, file_type, uniform_pdf_url, uniform_pdf_name, extracted_text, status, created_at"
)
DETAIL_COLUMNS_LEGACY = "id, file_name, file_url, file_type, extracted_text, status, created_at"


class SupabaseService:
    def __init__(self, settings: Settings):
        if not settings.supabase_url or not settings.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")

        self.settings = settings
        self.client: Client = create_client(settings.supabase_url, settings.supabase_key)
        self.bucket = settings.supabase_storage_bucket
        self._uniform_pdf_columns: Optional[bool] = None

    def _has_uniform_pdf_columns(self) -> bool:
        if self._uniform_pdf_columns is not None:
            return self._uniform_pdf_columns

        try:
            (
                self.client.table("documents")
                .select("uniform_pdf_url")
                .limit(1)
                .execute()
            )
            self._uniform_pdf_columns = True
        except APIError as exc:
            if exc.code == "42703":
                logger.warning(
                    "uniform_pdf_url column missing. Run supabase/migration_uniform_pdf.sql in Supabase SQL Editor."
                )
                self._uniform_pdf_columns = False
            else:
                raise

        return self._uniform_pdf_columns

    @staticmethod
    def _with_uniform_defaults(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for document in documents:
            document.setdefault("uniform_pdf_url", None)
            document.setdefault("uniform_pdf_name", None)
        return documents

    def upload_file(self, file_bytes: bytes, file_name: str, content_type: str) -> tuple[str, str]:
        document_id = str(uuid4())
        storage_path = f"{document_id}/{file_name}"

        self.client.storage.from_(self.bucket).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "false"},
        )

        public_url = self.client.storage.from_(self.bucket).get_public_url(storage_path)
        return document_id, public_url

    def create_document(
        self,
        document_id: str,
        file_name: str,
        file_url: str,
        file_type: str,
        status: str = "uploaded",
    ) -> dict[str, Any]:
        payload = {
            "id": document_id,
            "file_name": file_name,
            "file_url": file_url,
            "file_type": file_type,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        result = self.client.table("documents").insert(payload).execute()
        return result.data[0]

    def update_document_status(self, document_id: str, status: str) -> None:
        self.client.table("documents").update({"status": status}).eq("id", document_id).execute()

    def upload_uniform_pdf(self, document_id: str, pdf_bytes: bytes, uniform_file_name: str) -> str:
        storage_path = f"{document_id}/uniform/{uniform_file_name}"

        self.client.storage.from_(self.bucket).upload(
            path=storage_path,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )

        return self.client.storage.from_(self.bucket).get_public_url(storage_path)

    def save_processing_result(
        self,
        document_id: str,
        extracted_text: str,
        uniform_pdf_url: str,
        uniform_pdf_name: str,
    ) -> None:
        payload: dict[str, Any] = {
            "extracted_text": extracted_text,
            "status": "completed",
        }

        if self._has_uniform_pdf_columns():
            payload["uniform_pdf_url"] = uniform_pdf_url
            payload["uniform_pdf_name"] = uniform_pdf_name

        self.client.table("documents").update(payload).eq("id", document_id).execute()

    def mark_failed(self, document_id: str) -> None:
        self.client.table("documents").update({"status": "failed"}).eq("id", document_id).execute()

    def get_all_documents(self) -> list[dict[str, Any]]:
        columns = SELECT_COLUMNS_FULL if self._has_uniform_pdf_columns() else SELECT_COLUMNS_LEGACY
        result = (
            self.client.table("documents")
            .select(columns)
            .order("created_at", desc=True)
            .execute()
        )
        return self._with_uniform_defaults(result.data or [])

    def get_document(self, document_id: str) -> Optional[dict[str, Any]]:
        columns = DETAIL_COLUMNS_FULL if self._has_uniform_pdf_columns() else DETAIL_COLUMNS_LEGACY
        result = (
            self.client.table("documents")
            .select(columns)
            .eq("id", document_id)
            .execute()
        )
        if not result.data:
            return None
        return self._with_uniform_defaults(result.data)[0]

    def download_file(self, document_id: str, file_name: str) -> bytes:
        storage_path = f"{document_id}/{file_name}"
        return self.client.storage.from_(self.bucket).download(storage_path)

    def delete_document(self, document_id: str) -> Optional[dict[str, Any]]:
        document = self.get_document(document_id)
        if not document:
            return None

        paths_to_remove = [f"{document_id}/{document['file_name']}"]
        if document.get("uniform_pdf_name"):
            paths_to_remove.append(f"{document_id}/uniform/{document['uniform_pdf_name']}")

        try:
            self.client.storage.from_(self.bucket).remove(paths_to_remove)
        except Exception as exc:
            logger.warning("Failed to delete storage files for %s: %s", document_id, exc)

        self.client.table("documents").delete().eq("id", document_id).execute()
        return document
