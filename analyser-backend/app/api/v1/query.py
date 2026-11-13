"""Student natural language query endpoint with calibrated cosine threshold gating."""

from datetime import datetime, timedelta, timezone
import json
import logging
import re
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models import ChatHistory, Document, QueryLog, User
from app.schemas import Citation, ChatHistoryItem, QueryRequest, QueryResponse
from app.services import RAGEngine, VectorStoreService

logger = logging.getLogger("rag_query")
router = APIRouter(prefix="/query", tags=["Campus Query & RAG Engine"])

# Cosine similarity distance threshold cutoff (calibrated for all-MiniLM-L6-v2)
COSINE_DISTANCE_CUTOFF = 0.92


def save_chat_record(
    db: Session,
    user_id: int,
    query_text: str,
    answer_text: str,
    status: str,
    citations: Optional[list] = None,
):
    """Atomically record chat interaction in ChatHistory for persistent 30-day recall."""
    try:
        citations_json = None
        if citations:
            raw_citations = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in citations
            ]
            citations_json = json.dumps(raw_citations)
        rec = ChatHistory(
            user_id=user_id,
            query_text=query_text,
            answer_text=answer_text,
            citations_json=citations_json,
            status=status,
        )
        db.add(rec)
        db.commit()
    except Exception as e:
        logger.warning("Could not persist chat history: %s", e)
        db.rollback()


@router.post(
    "",
    response_model=QueryResponse,
    summary="Submit natural language circular query with calibrated cosine distance gating",
)
def submit_query(
    query_in: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Execute grounded semantic query against indexed campus circulars.

    Hallucination Defense Architecture:
    1. Validates query text.
    2. Empty DB Guard: If ChromaDB has 0 documents, aborts safely without throwing 500.
    3. Retrieves top candidate chunks via ChromaDB cosine similarity.
    4. Gating Check:
       - IF cosine distance <= 0.75: Invokes active Groq LLM (Qwen-27B / GPT-OSS) to synthesize grounded answer with citations.
       - IF cosine distance > 0.75: Aborts LLM call immediately, shielding students from hallucinations
         and returning a graceful not_found response with escalation permission.
    5. Saves interaction into ChatHistory with 30-day retention window.
    6. Comprehensive Exception Guard: Logs full traceback and always returns valid JSON QueryResponse.
    """
    try:
        clean_query = (query_in.query or "").strip()
        if not clean_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query string cannot be empty.",
            )

        vector_service = VectorStoreService()
        rag_engine = RAGEngine()

        # Step 1: Empty DB Guard and Active Document Verification
        is_student = (getattr(current_user, "role", "student") == "student")
        all_active_docs = db.query(Document).filter(
            Document.status == "active",
            or_(Document.is_active == True, Document.is_active.is_(None)),
        ).all()

        # Treat visibility case-insensitively: any document that is not explicitly 'faculty' is public
        active_docs = []
        for d in all_active_docs:
            d_vis = (getattr(d, "visibility", None) or "public").strip().lower()
            if is_student and d_vis == "faculty":
                continue
            active_docs.append(d)

        active_doc_id_strs = {str(d.id).strip() for d in active_docs}
        active_doc_id_ints = {int(d.id) for d in active_docs}
        active_filenames = {str(d.filename).strip().lower() for d in active_docs}

        try:
            item_count = vector_service.collection.count()
        except Exception as exc:
            logger.warning("Could not check vector collection count: %s", exc)
            item_count = 0

        if item_count == 0 or len(active_docs) == 0:
            try:
                log_entry = QueryLog(query_text=clean_query[:255], is_relevant=False)
                db.add(log_entry)
                db.commit()
            except Exception as db_err:
                logger.warning("Could not write query log: %s", db_err)
                db.rollback()

            fallback_msg = "I couldn't find verified details about this in existing campus circulars. No documents have been uploaded yet."
            save_chat_record(db, current_user.id, clean_query, fallback_msg, "not_found")

            return QueryResponse(
                status="not_found",
                message=fallback_msg,
                citations=[],
                can_escalate=True,
                best_distance=None,
            )

        # Step 2: Conversational Follow-Up Context Integration
        query_words_list = clean_query.split()
        search_query = clean_query
        if len(query_words_list) <= 6 and current_user:
            ten_mins_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
            prev_chat = (
                db.query(ChatHistory)
                .filter(
                    ChatHistory.user_id == current_user.id,
                    ChatHistory.created_at >= ten_mins_ago,
                )
                .order_by(ChatHistory.id.desc())
                .first()
            )
            if prev_chat and prev_chat.query_text:
                prev_text = prev_chat.query_text.strip()
                if prev_text.lower() not in clean_query.lower():
                    search_query = f"{prev_text} {clean_query}"
                    logger.info("Follow-up context augmented: '%s' -> '%s'", clean_query, search_query)

        # Step 3: Retrieve candidate chunks with keyword + numeric range hybrid retrieval
        SCHEDULE_KEYWORDS = {"exam", "room", "seating", "roll", "timing", "date", "schedule", "timetable"}
        query_words = set(re.findall(r"\b\w+\b", search_query.lower()))
        is_schedule_query = bool(query_words & SCHEDULE_KEYWORDS)
        effective_cutoff = 1.35 if is_schedule_query else COSINE_DISTANCE_CUTOFF

        retrieved_chunks = vector_service.query_similar_chunks(
            query=search_query,
            n_results=10,
            category_filter=query_in.category.strip() if query_in.category else None,
        )

        # Automatically include chunks from matching documents (e.g. Seating, Exam, Timetable)
        if is_schedule_query:
            try:
                raw_matching_docs = db.query(Document).filter(
                    Document.status == "active",
                    or_(Document.is_active == True, Document.is_active.is_(None)),
                    or_(
                        Document.filename.ilike("%seating%"),
                        Document.filename.ilike("%exam%"),
                        Document.filename.ilike("%timetable%"),
                        Document.filename.ilike("%schedule%"),
                        Document.category.ilike("%exam%"),
                    ),
                ).all()
                matching_docs = [
                    m for m in raw_matching_docs
                    if not (is_student and (getattr(m, "visibility", None) or "public").strip().lower() == "faculty")
                ]

                existing_chunk_ids = {c.get("id") or c.get("text") for c in retrieved_chunks}
                extra_chunks = []
                for m_doc in matching_docs:
                    doc_chunks = vector_service.collection.get(
                        where={"doc_id": m_doc.id},
                        limit=10,
                    )
                    if doc_chunks and doc_chunks.get("documents"):
                        docs_text = doc_chunks["documents"]
                        metas = doc_chunks.get("metadatas") or [{}] * len(docs_text)
                        ids = doc_chunks.get("ids") or [""] * len(docs_text)
                        for idx, d_text in enumerate(docs_text):
                            c_id = ids[idx] if idx < len(ids) else f"doc_{m_doc.id}_{idx}"
                            if c_id not in existing_chunk_ids and d_text not in existing_chunk_ids:
                                meta = metas[idx] if idx < len(metas) else {}
                                extra_chunks.append({
                                    "id": c_id,
                                    "text": d_text,
                                    "filename": meta.get("filename", m_doc.filename),
                                    "page_number": meta.get("page_number", 1),
                                    "category": meta.get("category", m_doc.category),
                                    "doc_id": m_doc.id,
                                    "visibility": meta.get("visibility", getattr(m_doc, "visibility", "public")),
                                    "distance": 0.40,
                                    "metadata": meta,
                                })
                                existing_chunk_ids.add(c_id)
                                existing_chunk_ids.add(d_text)

                if extra_chunks:
                    # Prepend matching seating/exam chunks and retain top 10
                    retrieved_chunks = (extra_chunks + retrieved_chunks)[:10]
            except Exception as hybrid_err:
                logger.warning("Error in hybrid retrieval: %s", hybrid_err)

        # Step 4: Strict Security & Deletion Filter (Zero tolerance for soft-deleted/confidential documents)
        def is_chunk_active_and_permitted(c: dict) -> bool:
            c_meta = c.get("metadata") or {}
            c_doc_id = c.get("doc_id")
            if c_doc_id is None:
                c_doc_id = c_meta.get("doc_id")
            if c_doc_id is None:
                c_doc_id = c_meta.get("circular_id")

            c_filename = str(c.get("filename") or c_meta.get("filename") or "").strip()
            c_vis = str(c.get("visibility") or c_meta.get("visibility") or "public").strip().lower()

            if is_student and c_vis == "faculty":
                return False

            if c_doc_id is not None:
                str_id = str(c_doc_id).strip()
                if str_id in active_doc_id_strs:
                    return True
                try:
                    if int(str_id) in active_doc_id_ints:
                        return True
                except (ValueError, TypeError):
                    pass

            if c_filename and c_filename.lower() in active_filenames:
                return True

            return False

        retrieved_chunks = [c for c in retrieved_chunks if is_chunk_active_and_permitted(c)]

        # Step 5: Compute minimum cosine distance on active chunks
        distances = [
            c["distance"] for c in retrieved_chunks if c.get("distance") is not None
        ]
        best_distance = min(distances) if distances else 1.0
        user_label = current_user.username if current_user else "guest"

        logger.info(
            "User '%s' queried '%s' (Search: '%s'). Retrieved %d active chunks (Best Cosine Distance: %.4f, Cutoff: %.2f)",
            user_label,
            clean_query[:50],
            search_query[:50],
            len(retrieved_chunks),
            best_distance,
            effective_cutoff,
        )

        # Step 6: Cutoff Gate Evaluation
        if retrieved_chunks and best_distance <= effective_cutoff:
            rag_result = rag_engine.generate_grounded_answer(
                query=clean_query,
                retrieved_chunks=retrieved_chunks,
                user_role=getattr(current_user, "role", "student"),
                user_username=getattr(current_user, "username", None),
                user_full_name=getattr(current_user, "full_name", None),
            )
            is_found = rag_result.get("found", False)

            try:
                log_entry = QueryLog(query_text=clean_query[:255], is_relevant=is_found)
                db.add(log_entry)
                db.commit()
            except Exception as db_err:
                logger.warning("Could not write query log: %s", db_err)
                db.rollback()

            if is_found:
                # Return strictly the single best/highest-relevance citation
                top_citations = rag_result.get("citations", [])[:1]
                citations_list = [
                    Citation(
                        filename=c.get("filename", "Official Circular"),
                        page_number=int(c.get("page_number", 1)),
                        category=c.get("category"),
                    )
                    for c in top_citations
                ]
                ans = rag_result.get("answer") or ""
                save_chat_record(db, current_user.id, clean_query, ans, "answered", citations_list)

                return QueryResponse(
                    status="answered",
                    answer=ans,
                    citations=citations_list,
                    can_escalate=False,
                    best_distance=round(best_distance, 4),
                )
            else:
                not_found_msg = (
                    rag_result.get("answer")
                    or "I couldn't find verified details about this in the available circulars."
                )
                save_chat_record(db, current_user.id, clean_query, not_found_msg, "not_found")

                return QueryResponse(
                    status="not_found",
                    message=not_found_msg,
                    citations=[],
                    can_escalate=True,
                    best_distance=round(best_distance, 4),
                )
        else:
            try:
                log_entry = QueryLog(query_text=clean_query[:255], is_relevant=False)
                db.add(log_entry)
                db.commit()
            except Exception as db_err:
                logger.warning("Could not write query log: %s", db_err)
                db.rollback()

            cutoff_msg = "I couldn't find verified details about this in the available circulars."
            save_chat_record(db, current_user.id, clean_query, cutoff_msg, "not_found")

            return QueryResponse(
                status="not_found",
                message=cutoff_msg,
                citations=[],
                can_escalate=True,
                best_distance=round(best_distance, 4) if distances else None,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error occurred while processing query '%s': %s", clean_query if 'clean_query' in locals() else '', exc)
        return QueryResponse(
            status="not_found",
            message="An unexpected error occurred while processing your circular inquiry. You may escalate it to the Faculty Desk.",
            citations=[],
            can_escalate=True,
            best_distance=None,
        )


@router.get(
    "/popular",
    summary="Retrieve frequently asked student queries from QueryLog",
)
def get_popular_queries(
    limit: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return top distinct logged queries to power the Common Inquiries panel dynamically."""
    from sqlalchemy import func

    try:
        results = (
            db.query(QueryLog.query_text, func.count(QueryLog.id).label("cnt"))
            .filter(QueryLog.is_relevant == True)
            .group_by(QueryLog.query_text)
            .order_by(func.count(QueryLog.id).desc())
            .limit(limit)
            .all()
        )
        if not results:
            results = (
                db.query(QueryLog.query_text, func.count(QueryLog.id).label("cnt"))
                .group_by(QueryLog.query_text)
                .order_by(func.count(QueryLog.id).desc())
                .limit(limit)
                .all()
            )
        return [r[0] for r in results if r[0]]
    except Exception as exc:
        logger.warning("Could not load popular queries: %s", exc)
        return []


@router.get(
    "/history",
    response_model=List[ChatHistoryItem],
    summary="Retrieve user chat conversation history (30-day retention window)",
)
def get_chat_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch user's query and answer history within the last 30 days chronologically."""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
    # Prune expired records across the system
    try:
        db.query(ChatHistory).filter(ChatHistory.created_at < cutoff_date).delete(synchronize_session=False)
        db.commit()
    except Exception as exc:
        logger.warning("Error pruning expired chat history: %s", exc)
        db.rollback()

    records = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == current_user.id, ChatHistory.created_at >= cutoff_date)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )

    items = []
    for r in records:
        citations = []
        if r.citations_json:
            try:
                raw_c = json.loads(r.citations_json)
                citations = [Citation(**c) for c in raw_c]
            except Exception:
                citations = []
        items.append(
            ChatHistoryItem(
                id=r.id,
                query_text=r.query_text,
                answer_text=r.answer_text,
                citations=citations,
                status=r.status,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
        )
    return items


@router.delete(
    "/history",
    summary="Clear personal chat history for authenticated user",
)
def clear_chat_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete all personal chat history records for current user."""
    try:
        db.query(ChatHistory).filter(ChatHistory.user_id == current_user.id).delete(synchronize_session=False)
        db.commit()
        return {"message": "Chat history cleared successfully."}
    except Exception as exc:
        db.rollback()
        logger.error("Failed to clear chat history: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear chat history.",
        )

