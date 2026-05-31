from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...deps import get_current_superuser
from ...models import Conversation, User
from ...security import get_password_hash


router = APIRouter(prefix="/admin/users", tags=["admin-users"], dependencies=[Depends(get_current_superuser)])


class UserCreate(BaseModel):
    email: str
    password: str
    username: str | None = None
    full_name: str | None = None
    name: str | None = None
    is_superuser: bool = False
    is_active: bool = True


class UserUpdate(BaseModel):
    is_active: bool | None = None
    is_superuser: bool | None = None
    email: str | None = None
    username: str | None = None
    full_name: str | None = None
    name: str | None = None
    password: str | None = None


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


def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if "@" not in normalized or "." not in normalized.rsplit("@", 1)[-1]:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Email invalide")
    return normalized


def _normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if len(normalized) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="L'identifiant doit contenir au moins 3 caracteres",
        )
    return normalized


def _has_field(payload: BaseModel, field_name: str) -> bool:
    fields_set = getattr(payload, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(payload, "__fields_set__", set())
    return field_name in fields_set


async def _ensure_unique_user_fields(db: AsyncSession, user_id: int | None, email: str, username: str) -> None:
    result = await db.execute(select(User).where(or_(User.email == email, User.username == username)))
    for existing in result.scalars().all():
        if existing.id != user_id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email ou identifiant deja utilise")


@router.get("")
async def list_users(db: AsyncSession = Depends(get_db)) -> list[dict[str, object]]:
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return [_user_dict(user) for user in result.scalars().all()]


@router.post("")
async def create_user(payload: UserCreate, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    email = _normalize_email(payload.email)
    username = _normalize_username(payload.username or email.split("@", 1)[0])
    await _ensure_unique_user_fields(db, None, email, username)
    user = User(
        email=email,
        username=username,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name or payload.name,
        is_active=payload.is_active,
        is_superuser=payload.is_superuser,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return _user_dict(user)


@router.patch("/{user_id}")
async def update_user(user_id: int, payload: UserUpdate, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utilisateur introuvable")
    next_email = _normalize_email(payload.email) if payload.email is not None else user.email
    next_username = _normalize_username(payload.username) if payload.username is not None else user.username
    await _ensure_unique_user_fields(db, user.id, next_email, next_username)
    user.email = next_email
    user.username = next_username
    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.is_superuser is not None:
        user.is_superuser = payload.is_superuser
    if _has_field(payload, "full_name") or _has_field(payload, "name"):
        user.full_name = payload.full_name or payload.name or None
    if payload.password:
        user.hashed_password = get_password_hash(payload.password)
    await db.commit()
    await db.refresh(user)
    return _user_dict(user)


@router.patch("/{user_id}/active")
async def set_user_active(user_id: int, is_active: bool, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utilisateur introuvable")
    user.is_active = is_active
    await db.commit()
    await db.refresh(user)
    return _user_dict(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
) -> None:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utilisateur introuvable")
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de supprimer le compte administrateur connecte",
        )
    await db.execute(update(Conversation).where(Conversation.user_id == user.id).values(user_id=None))
    await db.delete(user)
    await db.commit()
