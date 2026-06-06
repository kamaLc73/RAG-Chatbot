from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...cache import api_cache
from ...config import settings
from ...database import get_db
from ...deps import get_current_superuser
from ...models import Conversation, Message
from ..conversations import _conversation_dict, _message_dict


router = APIRouter(
    prefix="/admin/conversations",
    tags=["admin-conversations"],
    dependencies=[Depends(get_current_superuser)],
)

ADMIN_CONVERSATIONS_CACHE_KEY = "admin:conversations:list"
ADMIN_CONVERSATIONS_CACHE_TTL_SECONDS = settings.admin_conversations_cache_ttl_seconds


def invalidate_admin_conversation_cache() -> None:
    api_cache.clear_prefix("admin:conversations")
    api_cache.clear("admin:stats")


def _admin_conversation_dict(conversation: Conversation, include_messages: bool = True) -> dict[str, object]:
    messages = list(conversation.messages)
    assistant_messages = [message for message in messages if message.role == "assistant"]
    feedback_messages = [message for message in assistant_messages if message.feedback is not None]
    latencies = [message.latency_seconds for message in assistant_messages if message.latency_seconds is not None]
    context_counts = [message.context_docs or 0 for message in assistant_messages]
    user = conversation.user

    data = _conversation_dict(conversation, include_messages=include_messages)
    data.update(
        {
            "user_id": str(conversation.user_id) if conversation.user_id is not None else None,
            "user_name": user.full_name if user else None,
            "username": user.username if user else None,
            "user_email": user.email if user else None,
            "message_count": len(messages),
            "question_count": sum(1 for message in messages if message.role == "user"),
            "answer_count": len(assistant_messages),
            "feedback_count": len(feedback_messages),
            "positive_feedback": sum(1 for message in feedback_messages if message.feedback == 1),
            "negative_feedback": sum(1 for message in feedback_messages if message.feedback == -1),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "total_context_docs": sum(context_counts),
            "max_context_docs": max(context_counts) if context_counts else 0,
        }
    )
    return data


@router.get("")
async def list_all_conversations(db: AsyncSession = Depends(get_db)) -> list[dict[str, object]]:
    cached = api_cache.get(ADMIN_CONVERSATIONS_CACHE_KEY)
    if cached is not None:
        return cached

    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.user), selectinload(Conversation.messages))
        .order_by(Conversation.updated_at.desc())
    )
    payload = [_admin_conversation_dict(row) for row in result.scalars().all()]
    return api_cache.set(ADMIN_CONVERSATIONS_CACHE_KEY, payload, ttl_seconds=ADMIN_CONVERSATIONS_CACHE_TTL_SECONDS)


@router.get("/{conversation_id}")
async def get_any_conversation(conversation_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.user), selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return _admin_conversation_dict(conversation, include_messages=True)


@router.get("/{conversation_id}/messages")
async def list_messages(conversation_id: int, db: AsyncSession = Depends(get_db)) -> list[dict[str, object]]:
    if await db.get(Conversation, conversation_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    result = await db.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
    )
    return [_message_dict(row) for row in result.scalars().all()]


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_any_conversation(conversation_id: int, db: AsyncSession = Depends(get_db)) -> None:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await db.delete(conversation)
    await db.commit()
    invalidate_admin_conversation_cache()
