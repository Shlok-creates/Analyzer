"""Database connection and session management.

Configures SQLAlchemy engine, session maker, declarative base, and FastAPI session dependency.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings

# Engine configuration with health check pre-ping to recycle dropped connections
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and ensures clean closure."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

