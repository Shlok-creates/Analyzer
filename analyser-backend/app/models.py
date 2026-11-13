"""Relational database models for the Analyser campus circular query engine.

Defines schemas for Users, Documents, Tickets, and QueryLogs with RBAC,
referential integrity, constraints, and audit timestamps.
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    """User account entity representing Students, Faculty/Department Admins, and SuperAdmins."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="Student Roll No, Employee ID, or Admin Handle",
    )
    full_name = Column(
        String(255),
        nullable=True,
        comment="Student or Faculty Member's Full Name",
    )
    hashed_password = Column(String(255), nullable=False)
    role = Column(
        String(50),
        nullable=False,
        default="student",
        comment="Allowed: student, faculty, admin, superadmin, super_admin",
    )
    department = Column(
        String(255),
        nullable=True,
        comment="Assigned administrative or academic department (e.g., Computer Science, Accounts)",
    )
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    tickets = relationship(
        "Ticket",
        back_populates="student",
        cascade="all, delete-orphan",
    )
    documents = relationship(
        "Document",
        back_populates="uploader",
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('student', 'faculty', 'admin', 'superadmin', 'super_admin')",
            name="check_valid_user_role",
        ),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Document(Base):
    """Campus circular/official document entity indexed for query retrieval."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    category = Column(
        String(50),
        nullable=False,
        comment="Circular category (e.g., Exams, Fees, Timetables, Hostels)",
    )
    status = Column(
        String(20),
        default="processing",
        nullable=False,
        comment="Processing state: processing, active, failed",
    )
    total_pages = Column(Integer, default=0, nullable=False)
    visibility = Column(
        String(20),
        default="public",
        server_default="public",
        nullable=False,
        comment="Visibility level: 'public' (all students & faculty) or 'faculty' (confidential faculty only)",
    )
    is_active = Column(
        Boolean,
        default=True,
        server_default="true",
        nullable=False,
        comment="Soft-deletion flag: True if visible/active, False if soft-deleted",
    )
    uploaded_by = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    uploader = relationship("User", back_populates="documents")

    __table_args__ = (
        CheckConstraint(
            "status IN ('processing', 'active', 'failed')",
            name="check_valid_document_status",
        ),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', category='{self.category}', status='{self.status}')>"


class Ticket(Base):
    """Student query escalations routed to department faculties for resolution."""

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    query_text = Column(Text, nullable=False)
    status = Column(
        String(20),
        default="pending",
        nullable=False,
        comment="Ticket resolution status: pending, resolved",
    )
    faculty_reply = Column(Text, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    student = relationship("User", back_populates="tickets")

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'resolved', 'rejected')",
            name="check_valid_ticket_status",
        ),
    )

    @property
    def resolution_notes(self) -> Optional[str]:
        """Convenience property alias for faculty_reply."""
        return self.faculty_reply

    @resolution_notes.setter
    def resolution_notes(self, value: Optional[str]) -> None:
        self.faculty_reply = value

    def __repr__(self) -> str:
        return f"<Ticket(id={self.id}, student_id={self.student_id}, status='{self.status}')>"


class QueryLog(Base):
    """Audit log of student queries used for genuine dynamic FAQ aggregation."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String(255), index=True, nullable=False)
    is_relevant = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, query_text='{self.query_text[:30]}...', is_relevant={self.is_relevant})>"


class KeyDate(Base):
    """Campus key dates and academic calendar deadlines."""

    __tablename__ = "key_dates"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    date_str = Column(String(50), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<KeyDate(id={self.id}, title='{self.title}', date_str='{self.date_str}')>"


class ChatHistory(Base):
    """User conversation history with 30-day retention window."""

    __tablename__ = "chat_histories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="answered")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Relationships
    user = relationship("User", backref="chat_histories")

    def __repr__(self) -> str:
        return f"<ChatHistory(id={self.id}, user_id={self.user_id}, query='{self.query_text[:30]}...')>"


