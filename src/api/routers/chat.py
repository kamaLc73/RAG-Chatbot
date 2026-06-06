from __future__ import annotations

import json
from datetime import datetime
from time import perf_counter

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ..database import get_db
from ..deps import get_audio_transcriber, get_current_user, get_optional_current_user, get_rag_pipeline
from ..models import Conversation, Message, User
from ..services.title_generator import generate_conversation_title_task
from .conversations import _invalidate_admin_conversation_cache, normalize_org


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


class FeedbackRequest(BaseModel):
    feedback: int | None = None
    comment: str | None = None


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
    conversation = Conversation(user_id=user.id if user else None, title=title, title_source="auto_pending", org=org)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    _invalidate_admin_conversation_cache()
    return conversation


async def _query_chat_impl(
    payload: ChatQueryRequest,
    db: AsyncSession,
    user: User | None,
    rag,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    question = (payload.query or payload.message or "").strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Query is required")

    org = normalize_org(payload.org or payload.organization)
    conversation = await _ensure_conversation(db, payload, user, question, org)
    db.add(Message(conversation_id=conversation.id, role="user", content=question, org=org))

    started_at = perf_counter()
    result = await run_in_threadpool(
        rag.query,
        question,
        org=org,
        include_contexts=True,
        use_intent_classifier=payload.use_intent_classifier,
    )
    latency_seconds = perf_counter() - started_at
    videos = result.get("videos", [])
    forms = result.get("forms", [])
    answer = str(result.get("response", ""))
    context_docs = result.get("context_docs")
    if context_docs is None and isinstance(result.get("contexts"), list):
        context_docs = len(result["contexts"])

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        org=str(result.get("org", org)),
        intent=result.get("intent"),
        intent_confidence=result.get("intent_confidence"),
        context_docs=context_docs,
        latency_seconds=latency_seconds,
        payload_json=json.dumps(result, ensure_ascii=True, default=str),
    )
    db.add(assistant_message)
    conversation.org = org
    await db.commit()
    await db.refresh(assistant_message)
    _invalidate_admin_conversation_cache()
    background_tasks.add_task(
        generate_conversation_title_task,
        conversation.id,
        question,
        answer,
        org,
        result.get("intent"),
        rag,
    )

    resources = _resources(videos, forms)
    return {
        "conversation_id": str(conversation.id),
        "message_id": str(assistant_message.id),
        "answer": answer,
        "response": answer,
        "videos": videos,
        "forms": forms,
        "resources": resources,
        "context_docs": context_docs,
        "intent": result.get("intent"),
        "intent_confidence": result.get("intent_confidence"),
        "org": result.get("org", org),
        "cached": result.get("cached", False),
        "latency_seconds": latency_seconds,
        "timings": result.get("timings"),
    }


@router.post("")
async def chat(
    background_tasks: BackgroundTasks,
    payload: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_current_user),
    rag=Depends(get_rag_pipeline),
) -> dict[str, object]:
    return await _query_chat_impl(payload, db, user, rag, background_tasks)


@router.post("/query")
async def query_chat(
    background_tasks: BackgroundTasks,
    payload: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_current_user),
    rag=Depends(get_rag_pipeline),
) -> dict[str, object]:
    return await _query_chat_impl(payload, db, user, rag, background_tasks)


@router.post("/messages/{message_id}/feedback")
async def set_message_feedback(
    message_id: int,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    if payload.feedback not in (None, 1, -1):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Feedback must be 1, -1 or null")

    result = await db.execute(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.id == message_id, Conversation.user_id == user.id)
    )
    message = result.scalar_one_or_none()
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    if message.role != "assistant":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Feedback is only allowed on assistant messages")

    message.feedback = payload.feedback
    message.feedback_comment = payload.comment.strip() if payload.comment else None
    message.feedback_at = datetime.utcnow() if payload.feedback is not None else None
    await db.commit()
    await db.refresh(message)
    _invalidate_admin_conversation_cache()
    return {
        "message_id": str(message.id),
        "feedback": message.feedback,
        "feedback_comment": message.feedback_comment,
        "feedback_at": message.feedback_at.isoformat() if message.feedback_at else None,
    }


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
