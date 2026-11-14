"""Grounded synthesis and RAG generation service using Groq API.

Synthesizes authoritative answers strictly grounded in official campus circulars
using active Groq models (qwen/qwen3.8-27b, gpt-oss-120b) with citation tracking
and graceful fallback handling.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from groq import Groq

from app.config import settings

logger = logging.getLogger("rag_engine")

FALLBACK_MESSAGE = (
    "I couldn't find verified details about this in the available campus circulars."
)

SYSTEM_PROMPT = """You are "Analyzer", an intelligent, authoritative, and helpful campus personal assistant for university students.
Your primary role is to provide clear, thorough, and highly accurate answers derived directly from the provided official campus circular excerpts.

CORE REASONING & PRESENTATION GUIDELINES:
1. First-Person Singular Persona: Always speak naturally in the first-person singular ("I", "my") as the student's personal assistant (e.g., "Based on the official circular, I found that...", "Here is what the document outlines:"). NEVER refer to yourself in the plural ("we", "our team", "our system").
2. Source of Truth & Zero Hallucination: The uploaded circular excerpts are your authoritative source of truth. Never invent fake dates, fees, penalties, or rules.
3. Bold Key Information: Wrap all critical facts—such as **dates, deadlines, fee amounts, room numbers, roll number ranges, student names, courses, timings, subjects, and eligibility percentages**—in Markdown bold (`**...**`).
4. Single Authoritative Citation Tag:
   At the very end of your response on a new line, explicitly indicate which single chunk directly provided the verified answer using the exact tag:
   [USE_CHUNK: <index>]
   For example, if CHUNK 1 contained the information, conclude with:
   [USE_CHUNK: 1]
   Do NOT include any inline citations or [Notice: ...] tags in the body of your text. Write a clean, natural response.
5. Seating Arrangements, Exam Rooms & Roll Number Range Reasoning:
   When asked about seating arrangements, room allocations, timetables, or exam schedules for a specific student roll number (e.g., "what is the room for roll 11262064?"):
   - Inspect all tables, rows, and excerpts for numeric roll number ranges (e.g., 11262001 - 11262100).
   - Evaluate whether the target roll number falls between the start and end of any range (e.g., 11262001 <= 11262064 <= 11262100).
   - Extract the exact **Room Number**, **Classroom/Building**, **Exam Date**, **Subject**, and **Timing**.
   - Answer directly and authoritatively in natural bolded prose:
     "The roll number **11262064** falls within the range **11262001 to 11262100**, assigned to **Room 204** for **Midterm Exam** on **25 November 2026 at 10:00 AM**."
6. Adaptive Synthesis & Broad Queries: If the user asks a broad or general question about a circular (e.g., "provide a brief", "overview", "objectives", "summary", "explain this PDF", "what is this notice about?"), use your full reasoning capability to synthesize and structure the whole document's purpose, key highlights, crucial dates, and takeaways into clean, well-organized sections or bullet points with bolded key details.
7. Calculations, Comparisons & Rule Reasoning: If the user asks for calculations (e.g., fee totals, credit sums, grading thresholds), comparisons between rules, or specific procedural steps, reason step-by-step and answer naturally and thoroughly with bolded numbers.
8. Strict Grounding & Topic Relevance: Answer ONLY using facts directly relevant to the user's topic from the provided excerpts. Do NOT pull unrelated notes or generic policies if the excerpts do not actually answer the user's specific question—if no relevant active circular matches, return the fallback not-found status.
9. Fallback Rule: If the provided excerpts do not contain sufficient or relevant information to answer or address the user's inquiry, you MUST respond with:
"I couldn't find verified details about this in the available campus circulars."
Do not invent unverified outside policies."""

# Active and supported Groq model list in order of preference
ACTIVE_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


class RAGEngine:
    """Service providing grounded synthesis over circular chunks using Groq LLM API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
    ) -> None:
        """Initialize RAG Engine with Groq credentials and model name.

        Args:
            api_key: Optional Groq API key overriding settings.GROQ_API_KEY.
            model: Model identifier on Groq (default: 'openai/gpt-oss-120b').
        """
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        """Lazy-loaded Groq client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY is not configured in settings or environment."
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synthesize a grounded answer from retrieved circular chunks with strict fallbacks and citations.

        Args:
            query: The student's natural language question.
            retrieved_chunks: List of relevant chunks retrieved from the vector store.

        Returns:
            Dictionary containing:
                - answer (str): Synthesized response or standard fallback notice.
                - found (bool): True if answer was confirmed in circulars, False if missing.
                - citations (list): Structured list containing ONLY the chosen citation.
                - model (str): Name of the LLM model utilized.
        """
        # If no chunks were retrieved, immediately trigger standard fallback
        if not retrieved_chunks:
            return {
                "answer": FALLBACK_MESSAGE,
                "found": False,
                "citations": [],
                "model": self.model,
            }

        context_blocks: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks):
            filename = (
                chunk.get("filename")
                or chunk.get("metadata", {}).get("filename", "Unknown")
            )
            page_number = (
                chunk.get("page_number")
                or chunk.get("metadata", {}).get("page_number", 1)
            )
            category = (
                chunk.get("category")
                or chunk.get("metadata", {}).get("category", "General")
            )
            text = chunk.get("text", "").strip()

            context_blocks.append(
                f"[CHUNK {idx} | File: {filename} | Page: {page_number} | Category: {category}]\n{text}"
            )

        context_str = "\n\n".join(context_blocks)
        user_prompt = (
            f"OFFICIAL CAMPUS CIRCULAR CONTEXT:\n{context_str}\n\n"
            f"STUDENT QUERY:\n{query.strip()}\n\n"
            "Provide a clear, helpful answer strictly grounded in the excerpts above. "
            "Bold all key dates, deadlines, room numbers, roll ranges, amounts, names, courses, and timings with **bold**. "
            "Select ONLY the single chunk that actually contains the answer and end your response with [USE_CHUNK: <index>] on its own line. "
            "Do NOT include any inline citations in your text body."
        )

        candidate_models = [self.model] + [m for m in ACTIVE_MODELS if m != self.model]
        raw_answer: Optional[str] = None
        used_model = self.model

        for candidate in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
                raw_answer = response.choices[0].message.content.strip()
                used_model = candidate
                break
            except Exception as exc:
                logger.warning("Groq model '%s' failed: %s. Trying next candidate...", candidate, exc)

        if raw_answer is None:
            logger.error("All Groq model attempts failed for query: '%s'", query[:50])
            return {
                "answer": FALLBACK_MESSAGE,
                "found": False,
                "citations": [],
                "model": "error-fallback",
            }

        # Check if the model triggered the fallback notice
        is_fallback = (
            FALLBACK_MESSAGE.lower() in raw_answer.lower()
            or "information not found in existing campus circulars" in raw_answer.lower()
            or "couldn't find verified details" in raw_answer.lower()
            or "could not find verified details" in raw_answer.lower()
        )

        if is_fallback:
            return {
                "answer": FALLBACK_MESSAGE,
                "found": False,
                "citations": [],
                "model": used_model,
            }

        # Parse [USE_CHUNK: <index>] from LLM response
        chunk_match = re.search(r"\[USE_CHUNK:\s*(\d+)\]", raw_answer, flags=re.IGNORECASE)
        selected_chunk_idx: Optional[int] = None
        if chunk_match:
            try:
                c_idx = int(chunk_match.group(1))
                if 0 <= c_idx < len(retrieved_chunks):
                    selected_chunk_idx = c_idx
            except Exception:
                pass

        # Strip [USE_CHUNK: ...] and any accidental bracketed inline citations from the answer text
        clean_answer = re.sub(r"\[USE_CHUNK:\s*\d+\]", "", raw_answer, flags=re.IGNORECASE)
        clean_answer = re.sub(r"\[(?:Notice|Document|Page|CHUNK):[^\]]+\]", "", clean_answer, flags=re.IGNORECASE)
        clean_answer = re.sub(r"[ ]{2,}", " ", clean_answer).strip()

        # Map directly to the exact chunk selected by LLM (falling back to chunk 0 only if missing)
        chosen_chunk = retrieved_chunks[selected_chunk_idx if selected_chunk_idx is not None else 0]
        chosen_filename = (
            chosen_chunk.get("filename")
            or chosen_chunk.get("metadata", {}).get("filename", "Official Circular")
        )
        chosen_page = (
            chosen_chunk.get("page_number")
            or chosen_chunk.get("metadata", {}).get("page_number", 1)
        )
        chosen_cat = (
            chosen_chunk.get("category")
            or chosen_chunk.get("metadata", {}).get("category", "General")
        )

        citation = {
            "filename": chosen_filename,
            "page_number": int(chosen_page),
            "category": chosen_cat,
        }

        return {
            "answer": clean_answer,
            "found": True,
            "citations": [citation],
            "model": used_model,
        }
