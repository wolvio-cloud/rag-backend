import logging
import mimetypes
import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import get_settings
from models.schemas import UploadResponse
from services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])


def _get_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    supabase_service = SupabaseService(settings)

    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    extension = _get_extension(file.filename)
    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(sorted(settings.allowed_extensions))}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
        )

    content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"

    try:
        document_id, public_url = supabase_service.upload_file(
            file_bytes=file_bytes,
            file_name=file.filename,
            content_type=content_type,
        )
        supabase_service.create_document(
            document_id=document_id,
            file_name=file.filename,
            file_url=public_url,
            file_type=extension,
            status="uploaded",
        )
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {exc}") from exc

    return UploadResponse(document_id=document_id)
