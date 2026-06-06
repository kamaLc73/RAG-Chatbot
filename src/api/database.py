from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import func, or_, select
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.async_database_url, future=True, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False, autocommit=False)


def _missing_columns(sync_conn, table_name: str, expected: set[str]) -> set[str]:
    inspector = inspect(sync_conn)
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    return expected - existing


def _has_column(sync_conn, table_name: str, column_name: str) -> bool:
    inspector = inspect(sync_conn)
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


async def ensure_runtime_columns() -> None:
    async with engine.begin() as conn:
        missing_messages = await conn.run_sync(
            _missing_columns,
            "messages",
            {"latency_seconds", "feedback", "feedback_comment", "feedback_at"},
        )
        message_statements = {
            "latency_seconds": "ALTER TABLE messages ADD COLUMN latency_seconds FLOAT",
            "feedback": "ALTER TABLE messages ADD COLUMN feedback INTEGER",
            "feedback_comment": "ALTER TABLE messages ADD COLUMN feedback_comment TEXT",
            "feedback_at": "ALTER TABLE messages ADD COLUMN feedback_at TIMESTAMP",
        }
        for column_name in message_statements:
            if column_name in missing_messages:
                await conn.execute(text(message_statements[column_name]))

        missing_conversations = await conn.run_sync(
            _missing_columns,
            "conversations",
            {"title_source", "title_generated_at"},
        )
        conversation_statements = {
            "title_source": "ALTER TABLE conversations ADD COLUMN title_source VARCHAR(30) DEFAULT 'auto_pending' NOT NULL",
            "title_generated_at": "ALTER TABLE conversations ADD COLUMN title_generated_at TIMESTAMP",
        }
        for column_name in conversation_statements:
            if column_name in missing_conversations:
                await conn.execute(text(conversation_statements[column_name]))

        if conn.dialect.name == "postgresql":
            has_intent_tier = await conn.run_sync(_has_column, "intent_definitions", "tier")
            if has_intent_tier:
                await conn.execute(text("ALTER TABLE intent_definitions ALTER COLUMN tier DROP NOT NULL"))


async def init_db() -> None:
    from . import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await ensure_runtime_columns()

    from .services.intent_store import seed_intents_from_files

    async with SessionLocal() as db:
        await seed_intents_from_files(db)

    if settings.bootstrap_superuser_email and settings.bootstrap_superuser_password:
        from .models import User
        from .security import get_password_hash

        email = settings.bootstrap_superuser_email.strip().lower()
        username = (settings.bootstrap_superuser_username or email.split("@", 1)[0]).strip().lower()

        async with SessionLocal() as db:
            result = await db.execute(select(User).where(or_(User.email == email, User.username == username)))
            existing = result.scalar_one_or_none()
            if existing is None:
                db.add(
                    User(
                        email=email,
                        username=username,
                        hashed_password=get_password_hash(settings.bootstrap_superuser_password),
                        full_name=settings.bootstrap_superuser_full_name,
                        is_active=True,
                        is_superuser=True,
                    )
                )
            else:
                # Bootstrap must not overwrite admin edits made from the UI.
                # It only guarantees that the local bootstrap account remains usable as an admin.
                existing.is_active = True
                existing.is_superuser = True
            await db.commit()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as db:
        yield db


async def count_rows(db: AsyncSession, model: type[Base]) -> int:
    result = await db.execute(select(func.count()).select_from(model))
    return int(result.scalar_one())
