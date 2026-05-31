from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db
from .models import User
from .security import JWTError, decode_access_token


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_rag_pipeline(request: Request) -> Any:
    if getattr(request.app.state, "rag", None) is None:
        from src.chatbot.rag_pipeline import RAGPipeline

        request.app.state.rag = RAGPipeline()
    return request.app.state.rag


def get_audio_transcriber(request: Request) -> Any:
    if getattr(request.app.state, "audio_transcriber", None) is None:
        from src.audio.transcriber import AudioTranscriber

        request.app.state.audio_transcriber = AudioTranscriber(
            model_size=request.app.state.settings.audio_model_size,
            language=request.app.state.settings.audio_language,
        )
    return request.app.state.audio_transcriber


async def get_optional_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        return None
    return await db.get(User, user_id)


def get_current_user(user: User | None = Depends(get_optional_current_user)) -> User:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return user


def get_current_superuser(user: User = Depends(get_current_user)) -> User:
    if not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser required")
    return user
