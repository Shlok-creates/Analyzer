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
3. Bold Key Information: Wrap all critical facts—such as **dates, deadlines, fee amounts, room numbers, roll number ranges, timings, subjects, and eligibility percentages**—in Markdown bold (`**...**`).
4. NO Inline Citations: Do NOT include any `[Notice: ...]` or bracketed citations inside the body of your answer text. Write a clean, natural response. The single top citation will be attached automatically at the bottom.
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
        user_role: str = "student",
        user_username: Optional[str] = None,
        user_full_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize a grounded answer from retrieved circular chunks with strict fallbacks and citations.

        Args:
            query: The student's natural language question.
            retrieved_chunks: List of relevant chunks retrieved from the vector store.
            user_role: Role of the active user ('student', 'faculty', 'admin', 'superadmin').
            user_username: Username / roll number / employee ID.
            user_full_name: Full name of the user if present.

        Returns:
            Dictionary containing:
                - answer (str): Synthesized response or standard fallback notice.
                - found (bool): True if answer was confirmed in circulars, False if missing.
                - citations (list): Structured list of referenced circular filenames and pages.
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

        # Dynamic System Prompt with Role-Based Privacy Guard
        effective_system_prompt = SYSTEM_PROMPT
        user_privacy_note = ""
        if user_role == "student":
            resolved_name = (user_full_name or "").strip() or (user_username or "Student")
            resolved_id = (user_username or "Unknown").strip()
            privacy_mandate = (
                f"### TOP-PRIORITY PRIVACY MANDATE ###\n"
                f"The active user is a STUDENT named '{resolved_name}' (ID/Roll No: '{resolved_id}').\n"
                f"NEVER reveal personal, fee, grade, or admission status details of ANY OTHER student.\n"
                f"If asked about another student's personal record, refuse with:\n"
                f"'For privacy reasons, students can only view their own personal records or general campus announcements.'\n"
                f"Only share personal records matching '{resolved_name}' or '{resolved_id}', or general campus notices.\n\n"
            )
            effective_system_prompt = privacy_mandate + SYSTEM_PROMPT
            user_privacy_note = (
                f"CRITICAL PRIVACY DIRECTIVE:\n"
                f"The requesting user is student '{resolved_name}' (ID: '{resolved_id}'). "
                f"You MUST NEVER reveal individual fee balances, grades, or personal records of any other specific student. "
                f"If the query asks for another student's personal record, refuse with: "
                f"'For privacy reasons, students can only view their own personal records or general campus announcements.' "
                f"Otherwise, for general campus questions, hackathon rules, fee schedules, exam dates, syllabus, and announcements, answer normally and thoroughly.\n\n"
            )
        else:
            staff_name = (user_full_name or user_username or "Faculty/Admin").strip()
            effective_system_prompt += (
                f"\n\nROLE CONTEXT: The user is a university staff member ({user_role.upper()}: {staff_name}). "
                "Full operational administrative access is granted for official queries."
            )

        context_blocks: List[str] = []
        citations: List[Dict[str, Any]] = []
        seen_citations = set()

        for idx, chunk in enumerate(retrieved_chunks, 1):
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

            cit_key = (filename, page_number)
            if cit_key not in seen_citations:
                seen_citations.add(cit_key)
                citations.append(
                    {
                        "filename": filename,
                        "page_number": page_number,
                        "category": category,
                    }
                )

            context_blocks.append(
                f"--- EXCERPT {idx} | Circular: {filename} | Page: {page_number} | Category: {category} ---\n{text}"
            )

        context_str = "\n\n".join(context_blocks)
        user_prompt = (
            f"OFFICIAL CAMPUS CIRCULAR CONTEXT:\n{context_str}\n\n"
            f"STUDENT QUERY:\n{query.strip()}\n\n"
            f"{user_privacy_note}"
            "Provide a clear, helpful answer strictly grounded in the excerpts above. "
            "Bold all key dates, deadlines, room numbers, roll ranges, amounts, and timings with **bold**. "
            "Do NOT include any inline citations or [Notice: ...] tags in your text."
        )

        candidate_models = [self.model] + [m for m in ACTIVE_MODELS if m != self.model]
        raw_answer: Optional[str] = None
        used_model = self.model

        for candidate in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": effective_system_prompt},
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

        # Check if privacy refusal was returned
        is_privacy_refusal = (
            "for privacy reasons, students can only view their own personal records" in raw_answer.lower()
        )
        if is_privacy_refusal:
            return {
                "answer": raw_answer,
                "found": True,
                "citations": [],
                "model": used_model,
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

        # Strip any accidental inline citations from the answer text
        clean_answer = re.sub(r"\[(?:Notice|Document|Page):[^\]]+\]", "", raw_answer, flags=re.IGNORECASE)
        clean_answer = re.sub(r"[ ]{2,}", " ", clean_answer).strip()

        return {
            "answer": clean_answer,
            "found": True,
            "citations": citations[:1],  # Return ONLY the single highest-relevance citation
            "model": used_model,
        }
