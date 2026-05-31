from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..deps import get_current_user
from ..models import Conversation, Message, User


router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = "Nouvelle conversation"
    org: str | None = None
    organization: str | None = None


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
        "payload": payload,
        "resources": resources,
        "created_at": message.created_at.isoformat(),
    }


def _conversation_dict(conversation: Conversation, include_messages: bool = False) -> dict[str, object]:
    data = {
        "id": str(conversation.id),
        "title": conversation.title,
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
    conversation = Conversation(user_id=user.id, title=payload.title, org=org)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
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
