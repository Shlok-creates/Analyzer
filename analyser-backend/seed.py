"""SuperAdmin Root Seeder for Analyser Backend.

Initializes the database schema and provisions the initial root SuperAdmin credentials.
Strict Rule: Seeds ONLY root administrative credentials for system initialization;
no mock circulars or dummy FAQs are created.
"""

import logging
import sys
from pathlib import Path

# Ensure application root is in Python path for direct script execution
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from passlib.context import CryptContext
from sqlalchemy import text

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import Document, QueryLog, Ticket, User

# Configure structured terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("seed")

# Password hashing context using standard bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return pwd_context.hash(password)


def sync_postgres_sequences() -> None:
    """Synchronize PostgreSQL auto-increment sequences with current max(id) to avoid PK collisions."""
    if engine.dialect.name == "postgresql":
        logger.info("Synchronizing PostgreSQL auto-increment sequences...")
        try:
            with engine.connect() as connection:
                # Synchronize sequence for 'users' table
                sync_query = text(
                    """
                    DO $$
                    DECLARE
                        seq_name text;
                    BEGIN
                        seq_name := pg_get_serial_sequence('users', 'id');
                        IF seq_name IS NOT NULL THEN
                            PERFORM setval(
                                seq_name,
                                COALESCE((SELECT MAX(id) FROM users), 1),
                                (SELECT MAX(id) FROM users) IS NOT NULL
                            );
                        END IF;
                    END $$;
                    """
                )
                connection.execute(sync_query)
                connection.commit()
                logger.info("PostgreSQL sequence for 'users' table verified and synchronized.")
        except Exception as exc:
            logger.warning("Could not synchronize PostgreSQL sequence automatically: %s", exc)


def seed_database() -> None:
    """Perform database schema creation and root SuperAdmin provisioning."""
    logger.info("=" * 60)
    logger.info("Starting Analyser Database Foundation & Seeder")
    logger.info("=" * 60)

    # Step 1: Create all database tables
    logger.info("Creating database tables if not present...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

    # Step 2: Provision Root SuperAdmin
    db = SessionLocal()
    try:
        username = settings.SUPERADMIN_USERNAME.strip()
        existing_user = db.query(User).filter(User.username == username).first()

        if existing_user:
            logger.info(
                "SuperAdmin user '%s' already exists (ID: %s, Role: %s). Skipping creation.",
                existing_user.username,
                existing_user.id,
                existing_user.role,
            )
        else:
            logger.info("Root SuperAdmin '%s' not found. Provisioning credentials...", username)
            hashed_pwd = hash_password(settings.SUPERADMIN_PASSWORD)

            super_admin = User(
                username=username,
                hashed_password=hashed_pwd,
                role="super_admin",
                department="System Administration",
                is_active=True,
            )

            db.add(super_admin)
            db.commit()
            db.refresh(super_admin)

            logger.info(
                "Successfully created root SuperAdmin user! ID: %s | Username: '%s' | Role: '%s'",
                super_admin.id,
                super_admin.username,
                super_admin.role,
            )

        # Step 3: Resync sequences if using PostgreSQL
        sync_postgres_sequences()

    except Exception as exc:
        db.rollback()
        logger.error("Failed during database seeding: %s", exc, exc_info=True)
        raise exc
    finally:
        db.close()

    logger.info("=" * 60)
    logger.info("Analyser Phase 1 Database Seeder completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    seed_database()

