from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ..database import get_db
from ..deps import get_audio_transcriber, get_optional_current_user, get_rag_pipeline
from ..models import Conversation, Message, User
from .conversations import normalize_org


router = APIRouter(prefix="/chat", tags=["chat"])

try:
    import multipart  # noqa: F401

    HAS_MULTIPART = True
except ImportError:
    HAS_MULTIPART = False


class ChatQueryRequest(BaseModel):
    query: str | None = None
    message: str | None = None
    org: str | None = None
    organization: str | None = None
    conversation_id: int | None = None
    include_contexts: bool = False
    use_intent_classifier: bool = True


def _resources(videos: object, forms: object) -> list[dict[str, object]]:
    resources: list[dict[str, object]] = []
    if isinstance(videos, list):
        resources.extend({"type": "video", **item} for item in videos if isinstance(item, dict))
    if isinstance(forms, list):
        resources.extend({"type": "form", **item} for item in forms if isinstance(item, dict))
    return resources


async def _ensure_conversation(
    db: AsyncSession,
    payload: ChatQueryRequest,
    user: User | None,
    question: str,
    org: str,
) -> Conversation:
    if payload.conversation_id is not None:
        conversation = await db.get(Conversation, payload.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        if conversation.user_id is not None and (user is None or conversation.user_id != user.id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Conversation access denied")
        return conversation

    title = question[:80] or "Nouvelle conversation"
    conversation = Conversation(user_id=user.id if user else None, title=title, org=org)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def _query_chat_impl(
    payload: ChatQueryRequest,
    db: AsyncSession,
    user: User | None,
    rag,
) -> dict[str, object]:
    question = (payload.query or payload.message or "").strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Query is required")

    org = normalize_org(payload.org or payload.organization)
    conversation = await _ensure_conversation(db, payload, user, question, org)
    db.add(Message(conversation_id=conversation.id, role="user", content=question, org=org))

    result = await run_in_threadpool(
        rag.query,
        question,
        org=org,
        include_contexts=payload.include_contexts,
        use_intent_classifier=payload.use_intent_classifier,
    )
    videos = result.get("videos", [])
    forms = result.get("forms", [])
    answer = str(result.get("response", ""))

    db.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            org=str(result.get("org", org)),
            intent=result.get("intent"),
            intent_confidence=result.get("intent_confidence"),
            context_docs=result.get("context_docs"),
            payload_json=json.dumps(result, ensure_ascii=True, default=str),
        )
    )
    conversation.org = org
    await db.commit()

    resources = _resources(videos, forms)
    return {
        "conversation_id": str(conversation.id),
        "answer": answer,
        "response": answer,
        "videos": videos,
        "forms": forms,
        "resources": resources,
        "context_docs": result.get("context_docs"),
        "intent": result.get("intent"),
        "intent_confidence": result.get("intent_confidence"),
        "org": result.get("org", org),
        "cached": result.get("cached", False),
    }


@router.post("")
async def chat(
    payload: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_current_user),
    rag=Depends(get_rag_pipeline),
) -> dict[str, object]:
    return await _query_chat_impl(payload, db, user, rag)


@router.post("/query")
async def query_chat(
    payload: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_current_user),
    rag=Depends(get_rag_pipeline),
) -> dict[str, object]:
    return await _query_chat_impl(payload, db, user, rag)


if HAS_MULTIPART:

    @router.post("/transcribe")
    @router.post("/audio")
    async def transcribe_audio(
        file: UploadFile = File(...),
        transcriber=Depends(get_audio_transcriber),
    ) -> dict[str, str]:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Audio file is empty")
        text = await run_in_threadpool(transcriber.transcribe, audio_bytes)
        return {"text": text}

else:

    @router.post("/transcribe")
    @router.post("/audio")
    async def transcribe_audio_without_multipart(request: Request) -> dict[str, str]:
        await request.body()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Install python-multipart to enable UploadFile transcription.",
        )
