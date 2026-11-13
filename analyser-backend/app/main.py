"""Main FastAPI application entry point for Analyser.

Assembles routers, CORS middleware, OpenAPI documentation, and system health checks.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api.v1.auth import router as auth_router
from app.api.v1.circulars import router as circulars_router
from app.api.v1.query import router as query_router
from app.api.v1.tickets import router as tickets_router
from app.database import Base, engine
from app.models import ChatHistory, Document, KeyDate, QueryLog, Ticket, User
from app.services.vector_service import preload_models

logger = logging.getLogger("analyser.main")


def run_db_migrations():
    """Ensure database schema, required columns, and constraints are in place."""
    try:
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS department VARCHAR(255);"))
                conn.execute(text("ALTER TABLE users ALTER COLUMN department TYPE VARCHAR(255);"))
                conn.execute(text("ALTER TABLE users ALTER COLUMN username TYPE VARCHAR(100);"))
                conn.execute(text("ALTER TABLE users DROP CONSTRAINT IF EXISTS check_valid_user_role;"))
                conn.execute(text(
                    "ALTER TABLE users ADD CONSTRAINT check_valid_user_role "
                    "CHECK (role IN ('student', 'faculty', 'admin', 'superadmin', 'super_admin'));"
                ))
                conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;"))
                conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) DEFAULT 'public';"))
            elif engine.dialect.name == "sqlite":
                cursor = conn.execute(text("PRAGMA table_info(users)"))
                cols = [row[1] for row in cursor.fetchall()]
                if "full_name" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN full_name VARCHAR(255);"))
                if "department" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN department VARCHAR(255);"))
                doc_cursor = conn.execute(text("PRAGMA table_info(documents)"))
                doc_cols = [row[1] for row in doc_cursor.fetchall()]
                if "is_active" not in doc_cols:
                    conn.execute(text("ALTER TABLE documents ADD COLUMN is_active BOOLEAN DEFAULT 1;"))
                if "visibility" not in doc_cols:
                    conn.execute(text("ALTER TABLE documents ADD COLUMN visibility VARCHAR(20) DEFAULT 'public';"))
        logger.info("Database schema and role constraints validated successfully.")
    except Exception as e:
        logger.warning(f"Database migration check encountered warning: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager: preloads embedding models once at server startup.

    Eliminates request-time lazy loading cold starts for instant (<1s) subsecond page and query latency.
    """
    run_db_migrations()
    try:
        preload_models()
    except Exception as e:
        logger.warning(f"Embedding model preloading deferred: {e}")
    yield


app = FastAPI(
    title="Analyser - Campus Circular Query Engine & RBAC API",
    description=(
        "Authoritative campus circular query engine with 0.45 cosine distance gating, "
        "Groq RAG synthesis, closed-loop ticket escalation, and role-based access control."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development and API consumers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Version 1 API routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(circulars_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(tickets_router, prefix="/api/v1")

# Static Assets Mounting
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = STATIC_DIR / "index.html"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Uploaded Circular Files Mounting
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


@app.get("/", tags=["UI & Entry"], include_in_schema=False)
def serve_ui():
    """Serve the frontend split-view console interface at root instantly without model/DB overhead."""
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE))
    return {
        "name": "Analyser Campus Query Engine API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health & Status"])
@app.get("/api/v1/health", tags=["Health & Status"])
def health_check():
    """System health check endpoint for monitoring probes."""
    return {
        "status": "healthy",
        "engine": "Analyser V1",
    }


