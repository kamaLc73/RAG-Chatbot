from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...cache import api_cache
from ...config import settings
from ...database import count_rows, get_db
from ...deps import get_current_superuser
from ...models import Conversation, IntentDefinition, User


router = APIRouter(prefix="/admin", tags=["admin-stats"], dependencies=[Depends(get_current_superuser)])
ADMIN_STATS_CACHE_KEY = "admin:stats"
ADMIN_STATS_CACHE_TTL_SECONDS = settings.admin_stats_cache_ttl_seconds


@router.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)) -> dict[str, int]:
    cached = api_cache.get(ADMIN_STATS_CACHE_KEY)
    if cached is not None:
        return cached

    payload = {
        "users": await count_rows(db, User),
        "conversations": await count_rows(db, Conversation),
        "documents": 0,
        "intents": await count_rows(db, IntentDefinition),
    }
    return api_cache.set(ADMIN_STATS_CACHE_KEY, payload, ttl_seconds=ADMIN_STATS_CACHE_TTL_SECONDS)
