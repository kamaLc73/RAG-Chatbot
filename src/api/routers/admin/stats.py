from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import count_rows, get_db
from ...deps import get_current_superuser
from ...models import Conversation, User


router = APIRouter(prefix="/admin", tags=["admin-stats"], dependencies=[Depends(get_current_superuser)])


@router.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)) -> dict[str, int]:
    intent_count = 0
    intents_dir = Path("data/intents")
    if intents_dir.exists():
        intent_count = sum(1 for path in intents_dir.iterdir() if path.is_dir())
    return {
        "users": await count_rows(db, User),
        "conversations": await count_rows(db, Conversation),
        "documents": 0,
        "intents": intent_count,
    }
