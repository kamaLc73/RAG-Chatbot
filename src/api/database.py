from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.async_database_url, future=True, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False, autocommit=False)


async def init_db() -> None:
    from . import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
                existing.email = email
                existing.username = username
                existing.hashed_password = get_password_hash(settings.bootstrap_superuser_password)
                existing.full_name = settings.bootstrap_superuser_full_name
                existing.is_active = True
                existing.is_superuser = True
            await db.commit()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as db:
        yield db


async def count_rows(db: AsyncSession, model: type[Base]) -> int:
    result = await db.execute(select(func.count()).select_from(model))
    return int(result.scalar_one())
