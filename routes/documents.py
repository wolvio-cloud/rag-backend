from fastapi import APIRouter, HTTPException

from config import get_settings
from models.schemas import DashboardStats, DocumentDetail, DocumentSummary
from services.supabase_service import SupabaseService
from services.vector_service import VectorService

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    settings = get_settings()
    supabase_service = SupabaseService(settings)
    documents = supabase_service.get_all_documents()
    return [DocumentSummary(**document) for document in documents]


@router.get("/documents/stats", response_model=DashboardStats)
async def get_dashboard_stats() -> DashboardStats:
    settings = get_settings()
    supabase_service = SupabaseService(settings)
    documents = supabase_service.get_all_documents()

    total = len(documents)
    processed = sum(1 for doc in documents if doc.get("status") == "completed")
    failed = sum(1 for doc in documents if doc.get("status") == "failed")

    return DashboardStats(
        total_documents=total,
        processed_documents=processed,
        failed_documents=failed,
    )


@router.get("/document/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str) -> DocumentDetail:
    settings = get_settings()
    supabase_service = SupabaseService(settings)
    document = supabase_service.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentDetail(**document)


@router.delete("/document/{document_id}")
async def delete_document(document_id: str) -> dict[str, str]:
    settings = get_settings()
    supabase_service = SupabaseService(settings)
    vector_service = VectorService(settings)

    deleted = supabase_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    vector_service.delete_document_chunks(document_id)
    return {"message": "Document deleted successfully."}
