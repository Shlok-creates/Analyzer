"""Closed-loop student ticket escalation and faculty triage API endpoints."""

from datetime import datetime, timezone
import logging
import re
from typing import List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_role
from app.models import Ticket, User
from app.schemas import (
    TicketCreateRequest,
    TicketOut,
    TicketResolveRequest,
    TicketResolveResponse,
)

logger = logging.getLogger("tickets")
router = APIRouter(prefix="/tickets", tags=["Tickets & Escalation Triage"])

# ---------------------------------------------------------------------------
# Campus Bouncer & Comprehensive 300+ Word Academic Dictionary
# ---------------------------------------------------------------------------
PROFANITY_HARASSMENT_PATTERN = re.compile(
    r"\b(fuck|shit|bitch|bastard|asshole|idiot|stupid|moron|dumb|kill|nude|sex|porn|dick|pussy|slut|whore|harass|abuse|rape)\b",
    re.IGNORECASE,
)

MALPRACTICE_PATTERN = re.compile(
    r"\b(cheat|cheating|leak|leaked|paper leak|exam scam|proxy|fake certificate|bribe|bypass biometric|forge|impersonat|copying exam)\b",
    re.IGNORECASE,
)

# 300+ Word Comprehensive College & Academic Dictionary
CAMPUS_DICTIONARY_WORDS = [
    # Academics & Examinations
    "assessment", "objective", "syllabus", "assignment", "practical", "viva", "marks", "result",
    "revaluation", "exam", "examination", "hall ticket", "admit card", "timetable", "schedule",
    "curriculum", "credits", "grade", "gpa", "cgpa", "sgpa", "backlog", "supplementary", "arrear",
    "semester", "trimester", "annual", "midterm", "internal", "external", "moderation", "grace marks",
    "topper", "rank", "medal", "distinction", "pass", "fail", "re-exam", "answer script", "paper",
    "question paper", "scheme", "rubric", "test", "quiz", "invigilator", "controller", "evaluation",

    # Admissions, Identity & Documentation
    "admission", "enrollment", "registration", "counseling", "seat", "allotment", "quota", "merit",
    "id card", "smart card", "roll no", "enrollment no", "registration no", "prn", "usn",
    "bonafide", "tc", "transfer certificate", "migration", "provisional", "degree", "diploma",
    "transcript", "marksheet", "convocation", "attestation", "verification", "lor", "recommendation",
    "character certificate", "conduct certificate", "study certificate", "affidavit", "undertaking",
    "prospectus", "brochure", "handbook", "manual", "declaration", "certificate", "document",

    # Courses & Curriculum
    "course", "elective", "core", "major", "minor", "specialization", "department", "branch",
    "engineering", "science", "arts", "commerce", "management", "medicine", "pharmacy", "law",
    "laboratory", "lab", "experiment", "manual", "record", "observation", "project", "capstone",
    "mini project", "major project", "thesis", "dissertation", "seminar", "workshop", "symposium",
    "conference", "hackathon", "publication", "paper presentation", "presentation", "lecture",

    # Fees, Accounts & Financial Aid
    "fees", "fee", "tuition", "tuition fee", "exam fee", "hostel fee", "mess fee", "bus fee",
    "transport fee", "development fee", "caution money", "caution deposit", "security deposit",
    "refund", "excess payment", "late fee", "fine", "penalty", "installment", "challan", "receipt",
    "payment", "online payment", "upi", "net banking", "dues", "no dues", "clearance", "waiver",
    "concession", "scholarship", "freeship", "fellowship", "stipend", "financial aid", "bursary",
    "subsidy", "bank loan", "education loan", "accounts", "cashier", "accountant",

    # Faculty, Administration & Governance
    "faculty", "teacher", "professor", "assistant professor", "associate professor", "lecturer",
    "instructor", "hod", "head of department", "dean", "director", "principal", "vice principal",
    "vice chancellor", "vc", "registrar", "rector", "provost", "warden", "assistant warden",
    "mentor", "proctor", "tutor", "advisor", "class teacher", "counselor", "staff", "clerk",
    "peon", "security", "guard", "management", "trustee", "governing body", "syndicate", "senate",
    "academic council", "board of studies", "committee", "anti ragging", "grievance", "icc",
    "disciplinary", "discipline", "circular", "notice", "notification", "order", "guidelines",
    "ordinance", "statute", "rule", "regulations", "policy", "code of conduct",

    # Campus Life, Housing & Facilities
    "hostel", "dormitory", "room", "roommate", "room change", "vacating", "mess", "food",
    "dining", "canteen", "cafeteria", "night canteen", "mess menu", "curfew", "in-time", "out-time",
    "outing", "gate pass", "day pass", "visitor", "guest house", "laundry", "cleanliness",
    "maintenance", "electrician", "plumber", "water", "drinking water", "hot water", "ac",
    "cooler", "wifi", "internet", "lan", "network", "campus", "building", "block", "floor",
    "classroom", "auditorium", "seminar hall", "amphitheater", "parking", "vehicle", "helmet",

    # Library & Information Services
    "library", "librarian", "book", "textbook", "reference book", "journal", "periodical",
    "magazine", "newspaper", "borrow", "return", "renewal", "due date", "fine slip", "barcode",
    "rfid", "opac", "digital library", "e-book", "e-journal", "delnet", "ieee", "nptel", "swayam",
    "reading room", "study room", "stack area", "reprography", "photocopy", "xerox",

    # Transport & Commute
    "transport", "bus", "college bus", "route", "bus stop", "pickup", "drop", "bus pass",
    "driver", "conductor", "shuttle", "commute", "day scholar", "concession pass",

    # Placements, Internships & Careers
    "placement", "campus placement", "training", "placement cell", "internship", "intern",
    "job", "offer", "offer letter", "ctc", "package", "recruiter", "company", "drive",
    "campus drive", "aptitude", "interview", "technical interview", "hr interview", "group discussion",
    "gd", "resume", "cv", "portfolio", "skill", "mou", "industry", "alumni", "alumni network",

    # Health, Welfare & Student Activities
    "medical", "doctor", "nurse", "infirmary", "health center", "dispensary", "first aid",
    "ambulance", "emergency", "sick leave", "medical leave", "medical certificate", "insurance",
    "blood donation", "counseling", "mental health", "sports", "games", "athletics", "gym",
    "ground", "court", "cricket", "football", "basketball", "volleyball", "badminton", "table tennis",
    "indoor games", "sports meet", "annual sports", "inter-college", "trophy", "event", "fest",
    "cultural fest", "technical fest", "annual day", "freshers", "farewell", "club", "society",
    "committee", "nss", "ncc", "rotaract", "coding club", "robotics", "drama", "music", "dance",
    "photography", "literary", "debate", "volunteer", "student council", "union",

    # Calendar & Operations
    "holiday", "vacation", "summer vacation", "winter vacation", "semester break", "academic calendar",
    "working day", "shift", "timing", "batch", "section", "attendance", "shortage", "condonation",
    "onduty", "od", "leave application", "permission", "portal", "erp", "lms", "moodle", "login"
]

CAMPUS_DICTIONARY_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in CAMPUS_DICTIONARY_WORDS) + r")\b",
    re.IGNORECASE,
)


def validate_campus_ticket(text: str) -> Tuple[bool, Optional[str]]:
    """Vets student ticket submissions with a relaxed bouncer allowing all student inquiries and doubts.

    Blocks strictly:
    a) Abusive or profane language,
    b) Derogatory remarks or insults,
    c) Live exam cheating requests or malpractice,
    d) Pure spam / meaningless rubbish gibberish.

    Returns:
        Tuple of (is_valid: bool, rejection_reason: Optional[str])
    """
    clean_text = text.strip()

    # 1. Length & Basic Validity Check (Minimum 2 words, Maximum 200 words)
    words = clean_text.split()
    if len(words) < 2:
        return False, "Query is too short. Please describe your inquiry in at least 2 words."
    if len(words) > 200:
        return False, "Query exceeds the 200-word limit. Please describe your inquiry within 200 words."

    # 2. Pure Spam & Keyboard Gibberish Filter
    # e.g., 'aaaaaaa', 'zzzzzzzz' or long unpronounceable consonant clusters
    repeat_char_pattern = re.compile(r"([a-zA-Z0-9])\1{6,}")
    gibberish_pattern = re.compile(r"[bcdfghjklmnpqrstvwxyz]{7,}", re.IGNORECASE)
    if repeat_char_pattern.search(clean_text) or gibberish_pattern.search(clean_text):
        return False, "Query appears to contain spam, keyboard gibberish, or unreadable text."

    # 3. Abuse, Profane Language & Derogatory Remarks Filter
    if PROFANITY_HARASSMENT_PATTERN.search(clean_text):
        return False, "Abusive, derogatory, or profane language is strictly prohibited."

    # 4. Live Exam Cheating & Malpractice Filter
    if MALPRACTICE_PATTERN.search(clean_text):
        return False, "Requests involving live exam cheating, question paper leaks, or proxy attendance are strictly barred."

    # All student inquiries, doubts, and academic questions are permitted
    return True, None


# ---------------------------------------------------------------------------
# Student Ticket Escalation
# ---------------------------------------------------------------------------
@router.post(
    "/escalate",
    status_code=status.HTTP_201_CREATED,
    summary="Escalate an unanswered query to faculty triage (Student Only)",
)
def escalate_ticket(
    ticket_in: TicketCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["student", "faculty", "superadmin"])),
):
    """Escalate a query to departmental faculty with automated Bouncer filtering and debounce locks."""
    normalized_query = ticket_in.query_text.strip()

    # Step 1: Daily Rate Limit - Maximum 5 tickets per student per calendar day (UTC)
    now_utc = datetime.now(timezone.utc)
    midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    today_ticket_count = (
        db.query(Ticket)
        .filter(
            Ticket.student_id == current_user.id,
            Ticket.created_at >= midnight_utc,
        )
        .count()
    )
    if today_ticket_count >= 5:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily limit reached. You can submit up to 5 tickets per calendar day.",
        )

    # Step 2: Debounce Lock - Prevent duplicate pending tickets from same student
    duplicate_ticket = (
        db.query(Ticket)
        .filter(
            Ticket.student_id == current_user.id,
            Ticket.query_text == normalized_query,
            Ticket.status == "pending",
        )
        .first()
    )

    if duplicate_ticket:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"You already have an identical pending ticket (#{duplicate_ticket.id}) awaiting faculty review.",
        )

    # Step 3: Negative Rejection Bouncer Filter
    is_valid, refusal_reason = validate_campus_ticket(normalized_query)
    if not is_valid:
        logger.warning(
            "Bouncer REJECTED ticket from student '%s': %s",
            current_user.username,
            refusal_reason,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=refusal_reason,
        )

    # Step 4: Persist approved ticket
    ticket = Ticket(
        student_id=current_user.id,
        query_text=normalized_query,
        status="pending",
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    logger.info(
        "Ticket #%d successfully escalated by student '%s'",
        ticket.id,
        current_user.username,
    )

    return {
        "status": "escalated",
        "ticket_id": ticket.id,
        "message": "Your question has been forwarded to the Faculty Desk. You can track progress in My Tickets.",
    }


# ---------------------------------------------------------------------------
# Student: My Tickets View
# ---------------------------------------------------------------------------
@router.get(
    "/my",
    response_model=List[TicketOut],
    summary="List all tickets raised by the currently logged-in student",
)
def list_my_tickets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve all tickets created by the active student, showing status and faculty reply."""
    tickets = (
        db.query(Ticket)
        .filter(Ticket.student_id == current_user.id)
        .order_by(Ticket.created_at.desc())
        .all()
    )
    results = []
    for t in tickets:
        results.append(
            TicketOut(
                id=t.id,
                student_id=t.student_id,
                student_username=current_user.username,
                query_text=t.query_text,
                status=t.status,
                faculty_reply=t.faculty_reply,
                resolved_at=t.resolved_at,
                created_at=t.created_at,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Faculty Ticket Triage & Resolution
# ---------------------------------------------------------------------------
@router.get(
    "",
    response_model=List[TicketOut],
    summary="List tickets for review (Faculty/Admin sees all, Student sees own)",
)
def list_tickets(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by ticket status: pending, resolved, rejected",
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List tickets for review and triage."""
    query = db.query(Ticket)

    # If requester is a student, restrict to their own tickets
    if current_user.role == "student":
        query = query.filter(Ticket.student_id == current_user.id)

    if status_filter:
        clean_status = status_filter.strip().lower()
        if clean_status != "all":
            query = query.filter(Ticket.status == clean_status)

    tickets = query.order_by(Ticket.created_at.desc()).offset(skip).limit(limit).all()

    results = []
    for t in tickets:
        student_username = t.student.username if t.student else None
        results.append(
            TicketOut(
                id=t.id,
                student_id=t.student_id,
                student_username=student_username,
                query_text=t.query_text,
                status=t.status,
                faculty_reply=t.faculty_reply,
                resolved_at=t.resolved_at,
                created_at=t.created_at,
            )
        )
    return results


@router.post(
    "/{ticket_id}/resolve",
    response_model=TicketResolveResponse,
    summary="Resolve or close a student ticket with faculty notes (Faculty/SuperAdmin)",
)
def resolve_ticket(
    ticket_id: int,
    resolve_in: TicketResolveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["faculty", "superadmin"])),
):
    """Record faculty response, mark ticket as 'resolved' or 'rejected', and record timestamp."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket #{ticket_id} does not exist.",
        )

    target_status = (resolve_in.status or "resolved").lower().strip()
    if target_status not in ("resolved", "rejected"):
        target_status = "resolved"

    ticket.status = target_status
    ticket.faculty_reply = resolve_in.resolution_notes.strip()
    ticket.resolved_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(ticket)

    logger.info(
        "Ticket #%d marked as '%s' by faculty '%s'",
        ticket.id,
        ticket.status,
        current_user.username,
    )

    return TicketResolveResponse(
        status="success",
        ticket_id=ticket.id,
        ticket_status=ticket.status,
        resolved_at=ticket.resolved_at,
    )
