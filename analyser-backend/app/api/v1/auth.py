"""Authentication and user provisioning API endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_role
from app.core.security import create_access_token, get_password_hash, verify_password
from app.models import User
from app.schemas import LoginRequest, TokenResponse, UserCreate, UserOut, UserRegister

router = APIRouter(prefix="/auth", tags=["Authentication & RBAC"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Open registration for new students and teachers",
)
def register(
    user_in: UserRegister,
    db: Session = Depends(get_db),
):
    """Register a new student or faculty account directly and return an active session token."""
    resolved_username = user_in.get_resolved_username()
    if not resolved_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Roll Number or Employee ID is required.",
        )

    existing_user = (
        db.query(User)
        .filter(func.lower(User.username) == resolved_username.lower())
        .first()
    )
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Roll/Employee ID '{resolved_username}' is already registered.",
        )

    target_role = user_in.get_normalized_role()
    if target_role == "faculty":
        code = (user_in.faculty_code or "").strip()
        if code != "Pen is mightier than sword":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid Faculty Authorization Code.",
            )

    resolved_name = user_in.get_resolved_name()
    dept = user_in.department.strip() if user_in.department else None

    hashed_password = get_password_hash(user_in.password)
    new_user = User(
        username=resolved_username,
        full_name=resolved_name,
        hashed_password=hashed_password,
        role=target_role,
        department=dept,
        is_active=True,
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}",
        )

    access_token = create_access_token(
        data={"sub": new_user.username, "role": new_user.role, "uid": new_user.id}
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserOut.model_validate(new_user),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user and issue JWT bearer token",
)
async def login(
    request: Request,
    db: Session = Depends(get_db),
):
    """Authenticate student, faculty, or superadmin and return RFC 6750 Bearer token.

    Supports both JSON payloads and application/x-www-form-urlencoded (OAuth2 standard).
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            username = (body.get("username") or "").strip()
            password = body.get("password") or ""
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload.",
            )
    else:
        try:
            form = await request.form()
            username = (form.get("username") or "").strip()
            password = form.get("password") or ""
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid form payload.",
            )

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required.",
        )

    user = (
        db.query(User)
        .filter(func.lower(User.username) == username.lower())
        .first()
    )
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ID or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact University Administration.",
        )

    access_token = create_access_token(
        data={"sub": user.username, "role": user.role, "uid": user.id}
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserOut.model_validate(user),
    )


@router.post(
    "/users",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Provision new Faculty, Admin, or Student accounts (SuperAdmin Only)",
)
def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_role(["superadmin"])),
):
    """Dynamically provision new faculty or administrative accounts.

    Restricted exclusively to SuperAdmin accounts.
    """
    normalized_username = user_in.username.strip()
    existing_user = db.query(User).filter(User.username == normalized_username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{normalized_username}' is already registered.",
        )

    # Validate target role
    target_role = user_in.role.lower().strip()
    valid_roles = {"student", "faculty", "admin", "superadmin", "super_admin"}
    if target_role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Allowed roles: {sorted(valid_roles)}",
        )

    hashed_password = get_password_hash(user_in.password)
    new_user = User(
        username=normalized_username,
        full_name=user_in.full_name.strip() if user_in.full_name else None,
        hashed_password=hashed_password,
        role=target_role,
        department=user_in.department.strip() if user_in.department else None,
        is_active=True,
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User provisioning failed: {str(e)}",
        )

    return UserOut.model_validate(new_user)


@router.get(
    "/me",
    response_model=UserOut,
    summary="Retrieve current authenticated user profile",
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Return the profile information for the actively authenticated user."""
    return UserOut.model_validate(current_user)

