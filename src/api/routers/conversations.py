from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..cache import api_cache
from ..database import get_db
from ..deps import get_current_user
from ..models import Conversation, Message, User


router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = "Nouvelle conversation"
    org: str | None = None
    organization: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None


def _invalidate_admin_conversation_cache() -> None:
    api_cache.clear_prefix("admin:conversations")
    api_cache.clear("admin:stats")


def normalize_org(value: str | None) -> str:
    text = (value or "all").strip().lower()
    mapping = {
        "cnra": "cnra",
        "rcar": "rcar",
        "all": "all",
        "cnra & rcar": "all",
        "cnra/rcar": "all",
        "les deux": "all",
    }
    return mapping.get(text, "all")


def display_org(org: str) -> str:
    return {"cnra": "CNRA", "rcar": "RCAR", "all": "CNRA & RCAR"}.get(org, "CNRA & RCAR")


def _message_dict(message: Message) -> dict[str, object]:
    payload = None
    resources = []
    chunks = []
    if message.payload_json:
        try:
            payload = json.loads(message.payload_json)
            videos = payload.get("videos", []) if isinstance(payload, dict) else []
            forms = payload.get("forms", []) if isinstance(payload, dict) else []
            resources = [
                {"type": "video", **item} for item in videos if isinstance(item, dict)
            ] + [
                {"type": "form", **item} for item in forms if isinstance(item, dict)
            ]
            if isinstance(payload, dict):
                context_items = payload.get("context_items")
                context_metadata = payload.get("context_metadata")
                contexts = payload.get("contexts")
                if isinstance(context_items, list):
                    chunks = context_items
                elif isinstance(contexts, list):
                    chunks = [
                        {
                            "text": context,
                            "metadata": context_metadata[index] if isinstance(context_metadata, list) and index < len(context_metadata) else {},
                        }
                        for index, context in enumerate(contexts)
                    ]
        except json.JSONDecodeError:
            payload = None
    return {
        "id": str(message.id),
        "role": message.role,
        "content": message.content,
        "org": message.org,
        "intent": message.intent,
        "intent_confidence": message.intent_confidence,
        "context_docs": message.context_docs,
        "latency_seconds": message.latency_seconds,
        "feedback": message.feedback,
        "feedback_comment": message.feedback_comment,
        "feedback_at": message.feedback_at.isoformat() if message.feedback_at else None,
        "payload": payload,
        "resources": resources,
        "chunks": chunks,
        "created_at": message.created_at.isoformat(),
    }


def _conversation_dict(conversation: Conversation, include_messages: bool = False) -> dict[str, object]:
    data = {
        "id": str(conversation.id),
        "title": conversation.title,
        "title_source": conversation.title_source,
        "title_generated_at": conversation.title_generated_at.isoformat() if conversation.title_generated_at else None,
        "org": conversation.org,
        "organization": display_org(conversation.org),
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }
    if include_messages:
        data["messages"] = [_message_dict(message) for message in conversation.messages]
    return data


@router.get("")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[dict[str, object]]:
    result = await db.execute(
        select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc())
    )
    return [_conversation_dict(row) for row in result.scalars().all()]


@router.post("")
async def create_conversation(
    payload: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    org = normalize_org(payload.org or payload.organization)
    conversation = Conversation(user_id=user.id, title=payload.title, title_source="auto_pending", org=org)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    _invalidate_admin_conversation_cache()
    return _conversation_dict(conversation)


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return _conversation_dict(conversation, include_messages=True)


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[dict[str, object]]:
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return [_message_dict(message) for message in conversation.messages]


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: int,
    payload: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    if payload.title is not None:
        title = payload.title.strip()
        if not title:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Title is required")
        conversation.title = title[:255]
        conversation.title_source = "user"
        conversation.title_generated_at = None

    await db.commit()
    await db.refresh(conversation)
    _invalidate_admin_conversation_cache()
    return _conversation_dict(conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await db.delete(conversation)
    await db.commit()
    _invalidate_admin_conversation_cache()
