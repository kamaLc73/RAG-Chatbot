from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...database import get_db
from ...deps import get_current_superuser
from ...models import Conversation, Message
from ..conversations import _conversation_dict, _message_dict


router = APIRouter(
    prefix="/admin/conversations",
    tags=["admin-conversations"],
    dependencies=[Depends(get_current_superuser)],
)


@router.get("")
async def list_all_conversations(db: AsyncSession = Depends(get_db)) -> list[dict[str, object]]:
    result = await db.execute(select(Conversation).order_by(Conversation.updated_at.desc()))
    return [_conversation_dict(row) | {"user_id": row.user_id} for row in result.scalars().all()]


@router.get("/{conversation_id}")
async def get_any_conversation(conversation_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    result = await db.execute(
        select(Conversation).options(selectinload(Conversation.messages)).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    data = _conversation_dict(conversation, include_messages=True)
    data["user_id"] = conversation.user_id
    return data


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
