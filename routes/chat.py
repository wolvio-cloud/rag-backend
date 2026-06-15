import logging

from fastapi import APIRouter, HTTPException

from config import get_settings
from models.schemas import ChatRequest, ChatResponse, ChatSource
from services.claude_service import ClaudeService
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    claude_service = ClaudeService(settings)
    embedding_service = EmbeddingService(settings)
    vector_service = VectorService(settings)

    # Dynamic K Adjustment based on query intent
    question_lower = request.question.lower()
    generic_keywords = ["summarize", "summary", "overview", "all documents", "high-level", "high level", "compare", "main topics"]
    
    is_generic = any(keyword in question_lower for keyword in generic_keywords)
    dynamic_k = 10 if is_generic else settings.rag_top_k

    query_embedding = embedding_service.embed_query(request.question)
    results = vector_service.search(query_embedding, dynamic_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return ChatResponse(
            answer="The requested information is not available in the uploaded documents.",
            sources=[],
        )

    context_blocks: list[str] = []
    sources: list[ChatSource] = []
    seen_sources: set[tuple[str, int | None]] = set()

    for chunk_text, metadata in zip(documents, metadatas):
        file_name = metadata.get("file_name", "unknown")
        page_number = vector_service.build_page_number(metadata.get("page_number"))
        context_blocks.append(
            f"Document: {file_name}\nPage: {page_number or 'N/A'}\nContent:\n{chunk_text}"
        )

        source_key = (file_name, page_number)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(ChatSource(document_name=file_name, page=page_number))

    answer, followup_questions = claude_service.generate_answer(request.question, context_blocks)
    return ChatResponse(answer=answer, sources=sources, followup_questions=followup_questions)
