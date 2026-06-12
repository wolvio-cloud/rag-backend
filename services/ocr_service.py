import logging
import os
import tempfile
from dataclasses import dataclass

import easyocr
import fitz

from config import Settings

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    page_number: int
    text: str


class OCRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._reader: easyocr.Reader | None = None

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            logger.info("Initializing EasyOCR reader...")
            self._reader = easyocr.Reader(["en"], gpu=False)
        return self._reader

    def extract_text(self, file_bytes: bytes, file_type: str) -> tuple[str, list[PageText], bytes | None]:
        extension = file_type.lower()
        if extension == ".pdf":
            return self._extract_from_pdf(file_bytes)
        if extension in {".jpg", ".jpeg", ".png"}:
            return self._extract_from_image(file_bytes)
        raise ValueError(f"Unsupported file type for OCR: {file_type}")

    def _extract_from_pdf(self, file_bytes: bytes) -> tuple[str, list[PageText], bytes | None]:
        pages = self._extract_pdf_text_with_pymupdf(file_bytes)
        combined_text = "\n\n".join(page.text for page in pages if page.text.strip())

        if len(combined_text.strip()) >= self.settings.ocr_min_text_length:
            logger.info("PDF text extracted successfully using PyMuPDF.")
            return combined_text, pages, None

        logger.info("PDF appears scanned. Falling back to EasyOCR.")
        return self._ocr_pdf_pages(file_bytes)

    def _extract_pdf_text_with_pymupdf(self, file_bytes: bytes) -> list[PageText]:
        pages: list[PageText] = []
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                pages.append(PageText(page_number=index, text=text))
        return pages

    def _ocr_pdf_pages(self, file_bytes: bytes) -> tuple[str, list[PageText], bytes]:
        pages: list[PageText] = []
        document = fitz.open(stream=file_bytes, filetype="pdf")
        dpi = 150
        scale = 72.0 / dpi

        for index, page in enumerate(document, start=1):
            pix = page.get_pixmap(dpi=dpi)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
            pix.save(temp_path)

            try:
                results = self.reader.readtext(temp_path, detail=1, paragraph=False)
                page_text_lines = []
                for bbox, text, prob in results:
                    text_str = text.strip()
                    if not text_str:
                        continue
                    page_text_lines.append(text_str)
                    
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    
                    x0, y0 = min(x_coords) * scale, min(y_coords) * scale
                    x1, y1 = max(x_coords) * scale, max(y_coords) * scale
                    
                    # Ensure minimum size to prevent "empty box" error
                    if x1 - x0 < 1.0:
                        x1 = x0 + 1.0
                    if y1 - y0 < 1.0:
                        y1 = y0 + 1.0
                        
                    rect = fitz.Rect(x0, y0, x1, y1)
                    if rect.is_valid and not rect.is_empty:
                        try:
                            page.insert_textbox(rect, text_str, render_mode=3, align=0)
                        except Exception as e:
                            logger.warning(f"Skipped inserting text '{text_str}' due to error: {e}")
                    
                page_text = "\n".join(page_text_lines)
                pages.append(PageText(page_number=index, text=page_text))
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        combined_text = "\n\n".join(page.text for page in pages if page.text.strip())
        out_pdf_bytes = document.write()
        document.close()
        return combined_text, pages, out_pdf_bytes

    def _extract_from_image(self, file_bytes: bytes) -> tuple[str, list[PageText], bytes | None]:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            text = self._run_easyocr_on_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return text, [PageText(page_number=1, text=text)], None

    def _run_easyocr_on_file(self, file_path: str) -> str:
        results = self.reader.readtext(file_path, detail=0, paragraph=True)
        return "\n".join(result.strip() for result in results if result and result.strip())
