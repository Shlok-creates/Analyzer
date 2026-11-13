"""Unified Document Processor for Campus Circulars.

Supports extraction and chunking for all common college document formats:
- PDF (.pdf) with automatic Scanned/Image-based fallback using Groq Vision API
- Microsoft Word (.docx) with paragraph and table extraction
- Microsoft PowerPoint (.pptx) with slide-by-slide shape and table extraction
- Spreadsheets (.xlsx, .xls, .csv) with sheet-by-sheet structured table extraction
- Images (.png, .jpg, .jpeg, .webp) with Groq Vision OCR transcription

Generates overlapping chunks tagged with provenance metadata:
[Document: <filename> | Category: <category> | Section/Page: <label>]
"""

import base64
import csv
import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
from PIL import Image
import pypdf
import pypdfium2 as pdfium
from docx import Document as DocxDocument
from pptx import Presentation
from groq import Groq

from app.config import settings

logger = logging.getLogger("document_processor")

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

VISION_MODELS = [
    "qwen/qwen3.8-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
]


class DocumentProcessor:
    """Unified service for extracting structured text and creating searchable chunks across formats."""

    def __init__(self, groq_api_key: Optional[str] = None) -> None:
        self.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        self._groq_client: Optional[Groq] = None

    @property
    def groq_client(self) -> Optional[Groq]:
        if self._groq_client is None and self.groq_api_key:
            try:
                self._groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning("Could not initialize Groq client for vision extraction: %s", e)
        return self._groq_client

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace, remove excess line breaks, and strip text."""
        if not text:
            return ""
        cleaned = re.sub(r"[\r\t]+", " ", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
        return cleaned.strip()

    def transcribe_image_with_groq(
        self,
        image_bytes: bytes,
        filename: str,
        section_label: str = "Image",
    ) -> str:
        """Transcribe text, tables, dates, and seating charts from an image using Groq Vision API."""
        client = self.groq_client
        if not client:
            logger.warning("Groq client unavailable for image transcription of '%s'", filename)
            return ""

        try:
            # Ensure image is valid and within reasonable dimension limits
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.mode in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")

            # Resize if dimensions exceed 2048px on longest side
            max_dim = max(pil_img.size)
            if max_dim > 2048:
                scale = 2048 / max_dim
                new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

            # Ensure minimum 32x32 pixels as required by Groq Vision
            if pil_img.size[0] < 32 or pil_img.size[1] < 32:
                pil_img = pil_img.resize((max(32, pil_img.size[0]), max(32, pil_img.size[1])))

            out_buf = io.BytesIO()
            pil_img.save(out_buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(out_buf.getvalue()).decode("utf-8")

            prompt = (
                f"You are an expert document OCR transcriber. Transcribe the contents of this college circular/document "
                f"('{filename}', {section_label}) with complete accuracy. "
                f"Extract ALL notice titles, headings, dates, deadlines, timings, subjects, classroom/room numbers, "
                f"and numeric student roll number ranges (e.g. 11262001 to 11262100). "
                f"Format tables clearly using row-by-row '|' separators. Do not summarize or omit any row or roll range."
            )

            for model_name in VISION_MODELS:
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                                },
                            ],
                        }],
                        max_tokens=2048,
                    )
                    transcription = resp.choices[0].message.content or ""
                    if transcription.strip():
                        logger.info(
                            "Successfully transcribed %s (%s) with Groq model '%s' (%d chars).",
                            filename,
                            section_label,
                            model_name,
                            len(transcription),
                        )
                        return transcription.strip()
                except Exception as model_err:
                    logger.warning(
                        "Vision model '%s' failed for %s (%s): %s",
                        model_name,
                        filename,
                        section_label,
                        model_err,
                    )
                    continue

        except Exception as exc:
            logger.error("Error processing image transcription for '%s': %s", filename, exc, exc_info=True)

        return ""

    def _process_pdf(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> List[Tuple[int, str, str]]:
        """Extract pages from PDF. Automatically renders & OCRs pages with < 30 characters."""
        sections: List[Tuple[int, str, str]] = []  # (page_number, section_label, text)
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)

        pdfium_doc = None
        try:
            pdfium_doc = pdfium.PdfDocument(file_bytes)
        except Exception as pe:
            logger.warning("Could not open PDF with pypdfium2: %s", pe)

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            section_label = f"Page {page_num}"
            raw_text = page.extract_text() or ""
            cleaned = self.clean_text(raw_text)

            # Check for scanned or image-based PDF page (< 30 chars)
            if len(cleaned) < 30 and pdfium_doc is not None and page_idx < len(pdfium_doc):
                logger.info(
                    "Page %d of '%s' has only %d text chars. Triggering Groq Vision rendering...",
                    page_num,
                    filename,
                    len(cleaned),
                )
                try:
                    page_render = pdfium_doc[page_idx].render(scale=2.0).to_pil()
                    img_buf = io.BytesIO()
                    page_render.save(img_buf, format="PNG")
                    transcribed = self.transcribe_image_with_groq(
                        image_bytes=img_buf.getvalue(),
                        filename=filename,
                        section_label=section_label,
                    )
                    if transcribed:
                        cleaned = self.clean_text(transcribed)
                except Exception as render_err:
                    logger.warning("Failed to render/transcribe PDF page %d: %s", page_num, render_err)

            if cleaned:
                sections.append((page_num, section_label, cleaned))

        return sections

    def _process_docx(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> List[Tuple[int, str, str]]:
        """Extract paragraphs and tables from Word (.docx) document."""
        doc = DocxDocument(io.BytesIO(file_bytes))
        parts: List[str] = []

        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                parts.append(txt)

        for table in doc.tables:
            table_lines: List[str] = []
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                if any(cells):
                    table_lines.append(" | ".join(cells))
            if table_lines:
                parts.append("\n".join(table_lines))

        full_text = "\n\n".join(parts)
        cleaned = self.clean_text(full_text)
        if not cleaned:
            return []

        # If length is long, split into ~1200 char logical sections
        chunk_target = 1200
        sections: List[Tuple[int, str, str]] = []
        if len(cleaned) <= chunk_target:
            sections.append((1, "Section 1", cleaned))
        else:
            paragraphs = cleaned.split("\n\n")
            current_sec: List[str] = []
            current_len = 0
            sec_idx = 1
            for para in paragraphs:
                if current_len + len(para) > chunk_target and current_sec:
                    sections.append((sec_idx, f"Section {sec_idx}", "\n\n".join(current_sec)))
                    sec_idx += 1
                    current_sec = [para]
                    current_len = len(para)
                else:
                    current_sec.append(para)
                    current_len += len(para)
            if current_sec:
                sections.append((sec_idx, f"Section {sec_idx}", "\n\n".join(current_sec)))

        return sections

    def _process_pptx(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> List[Tuple[int, str, str]]:
        """Extract text slide-by-slide from PowerPoint (.pptx)."""
        prs = Presentation(io.BytesIO(file_bytes))
        sections: List[Tuple[int, str, str]] = []

        for slide_idx, slide in enumerate(prs.slides):
            slide_num = slide_idx + 1
            slide_parts: List[str] = []

            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_parts.append(text)
                elif shape.has_table:
                    for row in shape.table.rows:
                        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                        if any(cells):
                            slide_parts.append(" | ".join(cells))

            slide_text = self.clean_text("\n".join(slide_parts))
            if slide_text:
                sections.append((slide_num, f"Slide {slide_num}", slide_text))

        return sections

    def _process_spreadsheet(
        self,
        file_bytes: bytes,
        filename: str,
        ext: str,
    ) -> List[Tuple[int, str, str]]:
        """Extract tabular sheets from Excel (.xlsx, .xls) and CSV (.csv)."""
        sections: List[Tuple[int, str, str]] = []

        if ext == ".csv":
            text = ""
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    text = file_bytes.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue

            reader = csv.reader(io.StringIO(text))
            rows_str = []
            for row in reader:
                if any(c.strip() for c in row):
                    rows_str.append(" | ".join(c.strip() for c in row))
            csv_content = self.clean_text("\n".join(rows_str))
            if csv_content:
                sections.append((1, "Sheet: CSV Data", csv_content))
            return sections

        if ext == ".xlsx":
            try:
                wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
                for sheet_idx, sheet_name in enumerate(wb.sheetnames):
                    ws = wb[sheet_name]
                    rows_str = []
                    for row in ws.iter_rows(values_only=True):
                        cells = [str(c).strip() if c is not None else "" for c in row]
                        if any(cells):
                            rows_str.append(" | ".join(cells))
                    sheet_content = self.clean_text("\n".join(rows_str))
                    if sheet_content:
                        sections.append((sheet_idx + 1, f"Sheet: {sheet_name}", sheet_content))
            except Exception as e:
                logger.warning("openpyxl failed to read '%s': %s", filename, e)

        elif ext == ".xls":
            try:
                import xlrd
                wb = xlrd.open_workbook(file_contents=file_bytes)
                for sheet_idx in range(wb.nsheets):
                    sheet = wb.sheet_by_index(sheet_idx)
                    rows_str = []
                    for row_idx in range(sheet.nrows):
                        row_vals = sheet.row_values(row_idx)
                        cells = [str(c).strip() if c is not None else "" for c in row_vals]
                        if any(cells):
                            rows_str.append(" | ".join(cells))
                    sheet_content = self.clean_text("\n".join(rows_str))
                    if sheet_content:
                        sections.append((sheet_idx + 1, f"Sheet: {sheet.name}", sheet_content))
            except Exception as e:
                logger.warning("xlrd failed to read '%s': %s", filename, e)

        return sections

    def _process_image(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> List[Tuple[int, str, str]]:
        """Transcribe text, tables, and dates from Image (.png, .jpg, .jpeg, .webp)."""
        transcribed = self.transcribe_image_with_groq(
            image_bytes=file_bytes,
            filename=filename,
            section_label="Image Notice",
        )
        cleaned = self.clean_text(transcribed)
        if cleaned:
            return [(1, "Image Notice", cleaned)]
        return []

    def extract_chunks(
        self,
        file_bytes: bytes,
        filename: str,
        category: str = "General",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Extract and partition text into overlapping chunks tagged with provenance metadata.

        Returns:
            Tuple of:
                - List of chunk dicts (chunk_id, text, page_number, filename, category, section_label)
                - Total count of pages/slides/sheets
        """
        ext = Path(filename).suffix.lower()
        sections: List[Tuple[int, str, str]] = []

        if ext == ".pdf":
            sections = self._process_pdf(file_bytes, filename)
        elif ext == ".docx":
            sections = self._process_docx(file_bytes, filename)
        elif ext == ".pptx":
            sections = self._process_pptx(file_bytes, filename)
        elif ext in (".xlsx", ".xls", ".csv"):
            sections = self._process_spreadsheet(file_bytes, filename, ext)
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            sections = self._process_image(file_bytes, filename)
        else:
            logger.warning("Unsupported file extension '%s' for '%s'", ext, filename)
            return [], 0

        total_pages = max(1, len(sections))
        chunks: List[Dict[str, Any]] = []
        global_idx = 0
        stride = max(1, chunk_size - chunk_overlap)

        for page_num, section_label, text in sections:
            tag = f"[Document: {filename} | Category: {category} | Section/Page: {section_label}]\n"
            text_len = len(text)

            if text_len <= chunk_size:
                chunks.append({
                    "chunk_id": f"{filename}_s{page_num}_c{global_idx}",
                    "text": tag + text,
                    "page_number": page_num,
                    "filename": filename,
                    "category": category,
                    "section_label": section_label,
                })
                global_idx += 1
            else:
                start = 0
                while start < text_len:
                    end = min(start + chunk_size, text_len)
                    chunk_body = text[start:end].strip()
                    if chunk_body:
                        chunks.append({
                            "chunk_id": f"{filename}_s{page_num}_c{global_idx}",
                            "text": tag + chunk_body,
                            "page_number": page_num,
                            "filename": filename,
                            "category": category,
                            "section_label": section_label,
                        })
                        global_idx += 1
                    if end >= text_len:
                        break
                    start += stride

        return chunks, total_pages

