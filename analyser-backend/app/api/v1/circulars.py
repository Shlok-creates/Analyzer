"""Campus Circular upload and background ingestion pipeline."""

import io
import logging
import mimetypes
from pathlib import Path
from typing import List, Optional
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
import pypdf
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_role, get_current_user, oauth2_scheme_optional
from app.core.security import decode_access_token
from app.database import SessionLocal
from app.models import Document, User, KeyDate
from app.schemas import DocumentOut, DocumentUploadResponse, KeyDateCreate, KeyDateOut
from app.services import PDFService, VectorStoreService
from app.services.document_processor import DocumentProcessor, ALLOWED_EXTENSIONS

logger = logging.getLogger("circulars_ingestion")
router = APIRouter(prefix="/circulars", tags=["Circulars & Document Ingestion"])

# Storage directory for uploaded circular PDFs
CIRCULARS_UPLOAD_DIR = Path("uploads/circulars")
CIRCULARS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def process_circular_pdf(
    doc_id: int,
    filename: str,
    category: str,
    file_bytes: bytes,
) -> None:
    """Fast worker task for atomic notice extraction, vectorization, and DB update."""
    db: Session = SessionLocal()
    try:
        logger.info("Processing circular doc_id=%d ('%s')...", doc_id, filename)
        pdf_service = PDFService()
        vector_service = VectorStoreService()

        # Determine total page count
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)

        # a) Extract clean text chunks (chunk_size=800, overlap=100)
        chunks = pdf_service.extract_chunks_from_pdf(
            file_bytes=file_bytes,
            filename=filename,
            chunk_size=800,
            chunk_overlap=100,
        )

        # b) Batch embedding (batch_size=32) and single-batch ChromaDB insertion
        vector_service.upsert_document_chunks(
            doc_id=doc_id,
            filename=filename,
            category=category,
            chunks=chunks,
            batch_size=32,
        )

        # c) Atomically update document status in PostgreSQL to 'active'
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "active"
            doc.total_pages = total_pages
            db.commit()
            logger.info(
                "Successfully activated circular doc_id=%d with %d chunks across %d pages.",
                doc_id,
                len(chunks),
                total_pages,
            )
    except Exception as exc:
        db.rollback()
        logger.error(
            "Unhandled failure processing circular doc_id=%d: %s",
            doc_id,
            exc,
            exc_info=True,
        )
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = "failed"
                db.commit()
        except Exception as rollback_err:
            logger.error("Failed to mark document as failed: %s", rollback_err)
    finally:
        db.close()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload campus circular document for ingestion (Faculty/SuperAdmin)",
)
async def upload_circular(
    file: UploadFile = File(..., description="Circular document file to upload (PDF, Word, PPTX, Excel, CSV, Image)"),
    category: str = Form(..., description="Circular category (e.g. Exams, Fees, Hostels)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["faculty", "superadmin"])),
):
    """Upload and index a campus circular document across supported formats.

    Supports: PDF, DOCX, PPTX, XLSX, XLS, CSV, PNG, JPG, JPEG, WEBP.
    Extracts text/sections, performs batch embedding, inserts in ChromaDB, and marks active.
    """
    filename = file.filename or "untitled.pdf"
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format '{file_ext}'. Allowed formats: {allowed_list}",
        )

    file_bytes = await file.read()
    if not file_bytes or len(file_bytes) < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or corrupted.",
        )

    raw_cat = (category or "").strip()
    CATEGORY_MAPPING = {
        "fees & aid": "Fees & Aid",
        "fees": "Fees & Aid",
        "examinations": "Examinations",
        "exams": "Examinations",
        "admissions": "Admissions",
        "admission": "Admissions",
        "student life": "Student Life",
        "life": "Student Life",
        "general": "General",
    }
    clean_category = CATEGORY_MAPPING.get(raw_cat.lower(), raw_cat if raw_cat else "General")

    doc_processor = DocumentProcessor()
    vector_service = VectorStoreService()

    try:
        # 1. Extract clean chunks & count total pages/slides/sheets
        chunks, total_pages = doc_processor.extract_chunks(
            file_bytes=file_bytes,
            filename=filename,
            category=clean_category,
            chunk_size=800,
            chunk_overlap=100,
        )

        # 2. Create document record in database marked active immediately
        doc = Document(
            filename=filename,
            category=clean_category,
            status="active",
            is_active=True,
            total_pages=total_pages,
            uploaded_by=current_user.id,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # 3. Save physical file to disk for browser reading / downloading
        saved_file_path = CIRCULARS_UPLOAD_DIR / f"{doc.id}_{filename}"
        saved_file_path.write_bytes(file_bytes)

        # 4. Fast batch embedding and single-batch ChromaDB insertion
        if chunks:
            vector_service.upsert_document_chunks(
                doc_id=doc.id,
                filename=doc.filename,
                category=doc.category,
                chunks=chunks,
                batch_size=32,
            )

        logger.info(
            "Circular document ingestion completed for doc_id=%d ('%s'): %d chunks indexed across %d pages/sections.",
            doc.id,
            filename,
            len(chunks),
            total_pages,
        )

        return DocumentUploadResponse(
            message="Uploaded",
            document_id=doc.id,
            status="active",
        )

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Fast ingestion failed for '%s': %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index circular: {str(exc)}",
        )


@router.get(
    "",
    response_model=List[DocumentOut],
    summary="List all uploaded circulars (Accessible by Both Students and Faculty)",
)
def list_circulars(
    category: Optional[str] = None,
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List official campus circulars for student/faculty PDF library."""
    query = db.query(Document).filter(or_(Document.is_active == True, Document.is_active.is_(None)))

    if category:
        cat_clean = category.strip().lower()
        if cat_clean in ("fees", "fees & aid"):
            query = query.filter(func.lower(Document.category).in_(["fees & aid", "fees"]))
        elif cat_clean in ("exams", "examinations"):
            query = query.filter(func.lower(Document.category).in_(["examinations", "exams"]))
        elif cat_clean in ("admissions", "admission"):
            query = query.filter(func.lower(Document.category).in_(["admissions", "admission"]))
        elif cat_clean in ("student life", "life"):
            query = query.filter(func.lower(Document.category).in_(["student life", "life"]))
        else:
            query = query.filter(func.lower(Document.category) == cat_clean)
    if status_filter:
        query = query.filter(Document.status == status_filter.strip())

    documents = (
        query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    )
    results = []
    for doc in documents:
        item = DocumentOut.model_validate(doc)
        item.file_url = f"/api/v1/circulars/{doc.id}/file"
        results.append(item)
    return results


@router.get(
    "/{doc_id}/file",
    summary="Serve the actual PDF document file for in-browser reading",
)
def get_circular_file(
    doc_id: int,
    token: Optional[str] = Query(None),
    header_token: Optional[str] = Depends(oauth2_scheme_optional),
    db: Session = Depends(get_db),
):
    """Serve the raw PDF file directly so users can click 'View / Read PDF' and view it inline."""
    auth_token = token or header_token
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to view campus circular files.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_access_token(auth_token)
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = db.query(User).filter(User.username == username).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or deactivated.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc or doc.is_active is False:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circular document with ID {doc_id} not found or has been removed.",
        )

    file_path = CIRCULARS_UPLOAD_DIR / f"{doc.id}_{doc.filename}"
    if not file_path.exists():
        file_path = CIRCULARS_UPLOAD_DIR / doc.filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Circular document file not found on server storage.",
        )

    # Determine accurate MIME type and disposition based on file extension
    content_type, _ = mimetypes.guess_type(doc.filename)
    if not content_type:
        ext = Path(doc.filename).suffix.lower()
        ext_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".csv": "text/csv; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        content_type = ext_map.get(ext, "application/octet-stream")

    is_inline_viewable = (
        content_type.startswith("image/")
        or content_type == "application/pdf"
        or content_type.startswith("text/")
    )
    disposition = (
        f"inline; filename=\"{doc.filename}\""
        if is_inline_viewable
        else f"attachment; filename=\"{doc.filename}\""
    )

    return FileResponse(
        str(file_path),
        media_type=content_type,
        filename=doc.filename,
        headers={"Content-Disposition": disposition},
    )


@router.get(
    "/{doc_id}",
    response_model=DocumentOut,
    summary="Get status and details of a specific circular",
)
def get_circular_status(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve metadata and ingestion status for a circular document."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc or doc.is_active is False:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circular document with ID {doc_id} not found or has been removed.",
        )
    item = DocumentOut.model_validate(doc)
    item.file_url = f"/api/v1/circulars/{doc.id}/file"
    return item


@router.get(
    "/dates/list",
    response_model=List[KeyDateOut],
    summary="List all official campus key dates",
)
def list_key_dates(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve all active campus calendar key dates and deadlines."""
    return db.query(KeyDate).order_by(KeyDate.created_at.desc()).all()


@router.post(
    "/dates/create",
    response_model=KeyDateOut,
    status_code=status.HTTP_201_CREATED,
    summary="Publish a new campus key date (Faculty & Admin only)",
)
def create_key_date(
    date_in: KeyDateCreate,
    db: Session = Depends(get_db),
    current_faculty: User = Depends(require_role(["faculty", "admin", "superadmin"])),
):
    """Publish a new academic calendar key date."""
    new_date = KeyDate(
        title=date_in.title.strip(),
        date_str=date_in.date_str.strip(),
    )
    db.add(new_date)
    db.commit()
    db.refresh(new_date)
    return KeyDateOut.model_validate(new_date)


@router.delete(
    "/{doc_id}",
    summary="Soft-delete a circular and remove from search index (Faculty & Admin only)",
)
def delete_circular(
    doc_id: int,
    db: Session = Depends(get_db),
    current_faculty: User = Depends(require_role(["faculty", "admin", "superadmin"])),
):
    """Soft-delete an official circular from search views while keeping it preserved in database."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circular document #{doc_id} not found.",
        )

    # 1. Soft-delete in database (preserve row and physical files)
    doc.is_active = False
    db.commit()

    # 2. Explicitly purge vector chunks from ChromaDB so it immediately disappears from search & AI recall
    try:
        vector_service = VectorStoreService()
        vector_service.delete_document_chunks(doc_id=doc.id, filename=doc.filename)
        logger.info("Successfully purged vector embeddings for soft-deleted doc_id=%d ('%s')", doc.id, doc.filename)
    except Exception as exc:
        logger.warning("Could not delete vector embeddings for doc_id=%d: %s", doc_id, exc)

    return {"message": f"Circular '{doc.filename}' removed from active view.", "doc_id": doc_id}


@router.delete(
    "/dates/{date_id}",
    summary="Delete an academic key date (Faculty & Admin only)",
)
def delete_key_date(
    date_id: int,
    db: Session = Depends(get_db),
    current_faculty: User = Depends(require_role(["faculty", "admin", "superadmin"])),
):
    """Remove an announced academic calendar key date."""
    kd = db.query(KeyDate).filter(KeyDate.id == date_id).first()
    if not kd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key date #{date_id} not found.",
        )

    db.delete(kd)
    db.commit()
    return {"message": "Key date removed successfully.", "date_id": date_id}

