"""Application configuration management using Pydantic Settings.

Reads settings from environment variables and/or .env file with production-ready defaults.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CONFIG_DIR.parent
_ROOT_DIR = _BACKEND_DIR.parent


class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/analyser_db",
        description="PostgreSQL connection string.",
    )

    # JWT Authentication & Security
    SECRET_KEY: str = Field(
        default="analyser_insecure_default_secret_key_change_in_production_min_32_bytes!",
        description="Cryptographic secret key for signing JWT tokens.",
    )
    ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm.",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=1440,
        description="Access token lifespan in minutes (default: 24 hours / 1440 mins).",
    )

    # Vector Database & Embeddings Storage
    CHROMA_PERSIST_DIR: str = Field(
        default="./data/chroma",
        description="Local directory path for persisting ChromaDB vector embeddings.",
    )

    # External LLM Services
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google Gemini API Key for document analysis and query synthesis.",
    )
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API Key for fast LLM inference and grounded query synthesis.",
    )

    # Initial SuperAdmin Provisioning Credentials
    SUPERADMIN_USERNAME: str = Field(
        default="superadmin",
        description="Default root SuperAdmin handle for system bootstrap.",
    )
    SUPERADMIN_PASSWORD: str = Field(
        default="admin123",
        description="Default root SuperAdmin password for initial provisioning.",
    )

    model_config = SettingsConfigDict(
        env_file=(
            str(_BACKEND_DIR / ".env"),
            str(_ROOT_DIR / ".env"),
            ".env",
            "../.env",
            "analyser-backend/.env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

