import logging

from fastapi import APIRouter, HTTPException

from config import get_settings
from models.schemas import ProcessResponse
from services.embedding_service import EmbeddingService
from services.ocr_service import OCRService
from services.pdf_service import PDFService
from services.supabase_service import SupabaseService
from services.vector_service import VectorService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["process"])


@router.post("/process/{document_id}", response_model=ProcessResponse)
async def process_document(document_id: str) -> ProcessResponse:
    settings = get_settings()
    supabase_service = SupabaseService(settings)
    ocr_service = OCRService(settings)
    pdf_service = PDFService(settings)
    embedding_service = EmbeddingService(settings)
    vector_service = VectorService(settings)

    document = supabase_service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    supabase_service.update_document_status(document_id, "processing")

    try:
        original_bytes = supabase_service.download_file(document_id, document["file_name"])

        uniform_pdf_bytes, uniform_pdf_name = pdf_service.convert_to_uniform_pdf(
            file_bytes=original_bytes,
            file_type=document["file_type"],
            original_name=document["file_name"],
        )
        uniform_pdf_url = supabase_service.upload_uniform_pdf(
            document_id=document_id,
            pdf_bytes=uniform_pdf_bytes,
            uniform_file_name=uniform_pdf_name,
        )

        extracted_text, page_texts, ocr_pdf_bytes = ocr_service.extract_text(uniform_pdf_bytes, ".pdf")

        if ocr_pdf_bytes:
            uniform_pdf_url = supabase_service.upload_uniform_pdf(
                document_id=document_id,
                pdf_bytes=ocr_pdf_bytes,
                uniform_file_name=uniform_pdf_name,
            )

        if not extracted_text.strip():
            raise ValueError("No text could be extracted from the document.")

        supabase_service.save_processing_result(
            document_id=document_id,
            extracted_text=extracted_text,
            uniform_pdf_url=uniform_pdf_url,
            uniform_pdf_name=uniform_pdf_name,
        )

        chunks = embedding_service.chunk_text(extracted_text)
        embeddings = embedding_service.generate_embeddings(chunks)

        page_lookup = [(page.page_number, page.text) for page in page_texts]
        page_numbers = [
            embedding_service.estimate_page_for_chunk(chunk, page_lookup) for chunk in chunks
        ]

        vector_service.delete_document_chunks(document_id)
        vector_service.store_chunks(
            document_id=document_id,
            file_name=uniform_pdf_name,
            chunks=chunks,
            embeddings=embeddings,
            page_numbers=page_numbers,
            upload_date=document["created_at"],
        )

        return ProcessResponse(status="completed")
    except Exception as exc:
        logger.exception("Processing failed for document %s", document_id)
        supabase_service.mark_failed(document_id)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}") from exc
