"""Pydantic request and response schemas for Analyser API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Authentication & User Schemas
# ---------------------------------------------------------
class LoginRequest(BaseModel):
    """User credentials schema for authentication."""

    username: str = Field(..., min_length=1, description="Student Roll Number or Admin Handle")
    password: str = Field(..., min_length=1, description="Account password")


class UserOut(BaseModel):
    """Public user profile response representation."""

    id: int
    username: str
    full_name: Optional[str] = None
    role: str
    department: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """OAuth2 Bearer token response payload."""

    access_token: str
    token_type: str = "bearer"
    user: UserOut


class UserCreate(BaseModel):
    """Schema for SuperAdmin provisioning of new accounts."""

    username: str = Field(..., min_length=3, max_length=100, description="Roll No or Username")
    password: str = Field(..., min_length=4, max_length=128, description="Initial password")
    full_name: Optional[str] = Field(default=None, max_length=255, description="Full Name")
    role: str = Field(
        default="faculty",
        description="Assigned role: student, faculty, admin, superadmin",
    )
    department: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Department affiliation for faculty/admins",
    )


class UserRegister(BaseModel):
    """Schema for open registration of Students and Teachers."""

    username: Optional[str] = Field(default=None, max_length=100, description="Roll Number, Employee ID or Username")
    user_id: Optional[str] = Field(default=None, max_length=100, description="User / Roll / Employee ID alias")
    roll_number: Optional[str] = Field(default=None, max_length=100, description="Roll Number alias")
    password: str = Field(..., min_length=4, max_length=128, description="Account password")
    role: Optional[str] = Field(
        default="student",
        description="Target role: student, faculty, teacher, admin",
    )
    name: Optional[str] = Field(default=None, max_length=255, description="Full Name alias")
    full_name: Optional[str] = Field(default=None, max_length=255, description="Full Name")
    department: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Department affiliation (e.g. Computer Science, Accounts)",
    )
    faculty_code: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Faculty authorization secret",
    )

    def get_resolved_username(self) -> str:
        raw = self.username or self.user_id or self.roll_number or ""
        return raw.strip()

    def get_resolved_name(self) -> Optional[str]:
        raw = self.full_name or self.name or ""
        trimmed = raw.strip()
        return trimmed if trimmed else None

    def get_normalized_role(self) -> str:
        raw = (self.role or "student").lower().strip()
        if raw in ("faculty", "teacher", "prof", "professor", "staff"):
            return "faculty"
        if raw in ("admin", "superadmin", "super_admin"):
            return "admin"
        return "student"


# ---------------------------------------------------------
# Circular & Ingestion Schemas
# ---------------------------------------------------------
class DocumentOut(BaseModel):
    """Document record response schema."""

    id: int
    filename: str
    category: str
    status: str
    total_pages: int
    visibility: str = "public"
    is_active: bool = True
    uploaded_by: Optional[int] = None
    created_at: datetime
    file_url: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful circular upload dispatch."""

    message: str
    document_id: int
    status: str


# ---------------------------------------------------------
# RAG Query Schemas
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    """Natural language campus query input schema."""

    query: str = Field(..., min_length=1, max_length=500, description="Natural language student query")
    category: Optional[str] = Field(None, description="Optional category filter (e.g. Exams, Fees)")


class Citation(BaseModel):
    """Bibliographic citation referencing source circular and page."""

    filename: str
    page_number: int
    category: Optional[str] = None


class QueryResponse(BaseModel):
    """Grounded answer or negative response with citations."""

    status: str = Field(..., description="Query result status: 'answered' or 'not_found'")
    answer: Optional[str] = Field(None, description="Grounded synthesized answer")
    message: Optional[str] = Field(None, description="Explanation or fallback message")
    citations: List[Citation] = Field(default_factory=list, description="Source circular citations")
    can_escalate: bool = Field(default=False, description="Whether query can be escalated to faculty")
    best_distance: Optional[float] = Field(None, description="Best cosine distance score")


# ---------------------------------------------------------
# Ticket Management Schemas
# ---------------------------------------------------------
class TicketCreateRequest(BaseModel):
    """Student ticket escalation request."""

    query_text: str = Field(..., min_length=5, max_length=2000, description="Unresolved query to escalate (up to 200 words)")


class TicketOut(BaseModel):
    """Full ticket representation for faculty review."""

    id: int
    student_id: int
    student_username: Optional[str] = None
    query_text: str
    status: str
    faculty_reply: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TicketResolveRequest(BaseModel):
    """Faculty resolution input payload."""

    resolution_notes: str = Field(..., min_length=2, description="Faculty reply or resolution guidance")
    status: Optional[str] = Field(default="resolved", description="Target status: resolved or rejected")


class TicketResolveResponse(BaseModel):
    """Confirmation payload for ticket resolution."""

    status: str
    ticket_id: int
    ticket_status: str
    resolved_at: datetime


# ---------------------------------------------------------
# Campus Key Dates Schemas
# ---------------------------------------------------------
class KeyDateCreate(BaseModel):
    """Schema for faculty posting campus calendar key dates."""

    title: str = Field(..., min_length=2, max_length=255)
    date_str: str = Field(..., min_length=1, max_length=50)


class KeyDateOut(BaseModel):
    """Public key date representation."""

    id: int
    title: str
    date_str: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------
# Chat History Schemas
# ---------------------------------------------------------
class ChatHistoryItem(BaseModel):
    """Historical chat query and answer representation."""

    id: int
    query_text: str
    answer_text: str
    citations: List[Citation] = Field(default_factory=list)
    status: str
    created_at: str


