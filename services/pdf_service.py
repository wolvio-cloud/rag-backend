import logging
import os
from io import BytesIO

import fitz

from config import Settings

logger = logging.getLogger(__name__)

# A4 size in points
A4_WIDTH = 595.28
A4_HEIGHT = 841.89
PAGE_MARGIN = 36


class PDFService:
    """Convert uploaded images and PDFs into a standardized PDF format."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def convert_to_uniform_pdf(self, file_bytes: bytes, file_type: str, original_name: str) -> tuple[bytes, str]:
        stem = os.path.splitext(original_name)[0] or "document"
        uniform_name = f"{stem}_uniform.pdf"

        if file_type == ".pdf":
            pdf_bytes = self._normalize_pdf(file_bytes)
        elif file_type in {".jpg", ".jpeg", ".png"}:
            pdf_bytes = self._image_to_pdf(file_bytes, file_type)
        else:
            raise ValueError(f"Unsupported file type for PDF conversion: {file_type}")

        logger.info("Created uniform PDF: %s (%s bytes)", uniform_name, len(pdf_bytes))
        return pdf_bytes, uniform_name

    def _normalize_pdf(self, file_bytes: bytes) -> bytes:
        """Re-export PDF with consistent structure for uniform storage."""
        source = fitz.open(stream=file_bytes, filetype="pdf")
        output = fitz.open()

        try:
            output.insert_pdf(source)
            buffer = BytesIO()
            output.save(
                buffer,
                garbage=4,
                deflate=True,
                clean=True,
                pretty=False,
            )
            return buffer.getvalue()
        finally:
            source.close()
            output.close()

    def _image_to_pdf(self, file_bytes: bytes, file_type: str) -> bytes:
        """Place image on a standard A4 PDF page."""
        output = fitz.open()

        try:
            page = output.new_page(width=A4_WIDTH, height=A4_HEIGHT)
            content_rect = fitz.Rect(
                PAGE_MARGIN,
                PAGE_MARGIN,
                A4_WIDTH - PAGE_MARGIN,
                A4_HEIGHT - PAGE_MARGIN,
            )
            page.insert_image(content_rect, stream=file_bytes, keep_proportion=True)

            buffer = BytesIO()
            output.save(
                buffer,
                garbage=4,
                deflate=True,
                clean=True,
            )
            return buffer.getvalue()
        finally:
            output.close()
