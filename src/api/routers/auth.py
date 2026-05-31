from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..security import create_access_token, get_password_hash, verify_password


router = APIRouter(prefix="/auth", tags=["auth"])


class SignupRequest(BaseModel):
    email: str
    password: str
    username: str | None = None
    full_name: str | None = None
    name: str | None = None
    organization: str | None = None


class LoginRequest(BaseModel):
    email: str | None = None
    username: str | None = None
    password: str
    mode: str = "user"


def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if "@" not in normalized or "." not in normalized.rsplit("@", 1)[-1]:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid email")
    return normalized


def _normalize_username(value: str) -> str:
    username = value.strip().lower()
    if len(username) < 3:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Username must contain 3 characters")
    return username


def _user_dict(user: User) -> dict[str, object]:
    return {
        "id": str(user.id),
        "email": user.email,
        "username": user.username,
        "name": user.full_name or user.username,
        "full_name": user.full_name,
        "role": "admin" if user.is_superuser else "user",
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "created_at": user.created_at.isoformat(),
    }


def _auth_response(user: User) -> dict[str, object]:
    token = create_access_token(str(user.id), timedelta(minutes=settings.access_token_minutes))
    return {"access_token": token, "token_type": "bearer", "user": _user_dict(user)}


async def _email_or_username_exists(db: AsyncSession, email: str, username: str) -> bool:
    result = await db.execute(select(User).where(or_(User.email == email, User.username == username)))
    return result.scalar_one_or_none() is not None


@router.post("/signup")
async def signup(
    payload: SignupRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    email = _normalize_email(payload.email)
    username = _normalize_username(payload.username or email.split("@", 1)[0])
    if await _email_or_username_exists(db, email, username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already registered")

    total = await db.execute(select(func.count()).select_from(User))
    user = User(
        email=email,
        username=username,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name or payload.name,
        is_active=True,
        is_superuser=int(total.scalar_one()) == 0,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return _auth_response(user)


@router.post("/register")
async def register(
    payload: SignupRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    return await signup(payload, db)


@router.post("/login")
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    login_value = payload.email or payload.username
    if not login_value:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Email or username is required")

    normalized = login_value.strip().lower()
    result = await db.execute(select(User).where(or_(User.email == normalized, User.username == normalized)))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    if payload.mode.lower() == "admin" and not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return _auth_response(user)


@router.post("/logout")
async def logout() -> dict[str, bool]:
    return {"ok": True}


@router.post("/refresh")
async def refresh(user: User = Depends(get_current_user)) -> dict[str, object]:
    return _auth_response(user)


@router.get("/me")
async def me(user: User = Depends(get_current_user)) -> dict[str, object]:
    return _user_dict(user)
