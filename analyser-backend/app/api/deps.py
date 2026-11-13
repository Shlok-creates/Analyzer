"""FastAPI dependencies for database sessions and role-based JWT authentication."""

from typing import Callable, List, Optional, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.database import get_db
from app.models import User

# OAuth2 Bearer token extraction pointing to the login route
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# Canonical role synonyms and hierarchy mappings
ROLE_ALIASES = {
    "student": {"student"},
    "faculty": {"faculty", "admin", "superadmin", "super_admin"},
    "admin": {"faculty", "admin", "superadmin", "super_admin"},
    "superadmin": {"superadmin", "super_admin"},
    "super_admin": {"superadmin", "super_admin"},
}


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validate bearer token and resolve authenticated database user.

    Raises:
        HTTPException 401: If token is invalid, expired, or user not found/inactive.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return user


def get_optional_current_user(
    token: Optional[str] = Depends(oauth2_scheme_optional),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Resolve current user if bearer token is provided, or return None for guests."""
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        if not username:
            return None
        user = db.query(User).filter(User.username == username).first()
        if user and user.is_active:
            return user
        return None
    except Exception:
        return None


def require_role(allowed_roles: Union[str, List[str]]) -> Callable[[User], User]:
    """Dependency factory enforcing strict role-based access control (RBAC).

    Args:
        allowed_roles: Role or list of roles permitted to access the endpoint.
                       Accepts 'student', 'faculty', 'admin', 'superadmin'.

    Returns:
        Dependency callable verifying current user's role against permitted set.
    """
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]

    # Expand allowed roles through role hierarchies and aliases
    permitted_roles = set()
    for role in allowed_roles:
        normalized = role.lower().strip()
        permitted_roles.add(normalized)
        # If superadmin requested, also allow super_admin
        if normalized in ("superadmin", "super_admin"):
            permitted_roles.update(["superadmin", "super_admin"])
        # If faculty requested, also allow admin and superadmin
        if normalized in ("faculty", "admin"):
            permitted_roles.update(["faculty", "admin", "superadmin", "super_admin"])

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = (current_user.role or "").lower().strip()

        # Superadmin accounts always bypass administrative role gates
        if user_role in ("superadmin", "super_admin"):
            return current_user

        if user_role not in permitted_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required role: {allowed_roles}, your role: '{current_user.role}'",
            )
        return current_user

    return role_checker

