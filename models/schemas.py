from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str


class ProcessResponse(BaseModel):
    status: str


class DocumentSummary(BaseModel):
    id: str
    file_name: str
    file_url: str
    file_type: str
    uniform_pdf_url: Optional[str] = None
    uniform_pdf_name: Optional[str] = None
    status: str
    created_at: datetime


class DocumentDetail(BaseModel):
    id: str
    file_name: str
    file_url: str
    file_type: str
    uniform_pdf_url: Optional[str] = None
    uniform_pdf_name: Optional[str] = None
    extracted_text: Optional[str] = None
    status: str
    created_at: datetime


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatSource(BaseModel):
    document_name: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    followup_questions: list[str] = []


class DashboardStats(BaseModel):
    total_documents: int
    processed_documents: int
    failed_documents: int
