from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import re
from typing import Any

from starlette.concurrency import run_in_threadpool

from ..cache import api_cache
from ..database import SessionLocal
from ..models import Conversation


logger = logging.getLogger(__name__)


TITLE_TIMEOUT_SECONDS = 8.0
MAX_TITLE_WORDS = 8
MAX_TITLE_CHARS = 80


def _org_label(org: str) -> str:
    return {"rcar": "RCAR", "cnra": "CNRA", "all": "RCAR/CNRA"}.get((org or "all").lower(), "RCAR/CNRA")


def _clean_title(value: str) -> str:
    title = str(value or "").strip()
    title = re.sub(r"^\s*(titre|title)\s*:\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[\r\n\t]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" \"'`.,;:!?")
    words = title.split()
    if len(words) > MAX_TITLE_WORDS:
        title = " ".join(words[:MAX_TITLE_WORDS])
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].strip() or title[:MAX_TITLE_CHARS].strip()
    return title


def _fallback_title(question: str, org: str) -> str:
    text = str(question or "").strip().lower()
    text = re.sub(r"\b(bonjour|salut|salam|svp|s'il vous plait|merci|donne moi|peux tu|pouvez vous)\b", " ", text)
    text = re.sub(r"[^\w' -]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()[:MAX_TITLE_WORDS]
    title = " ".join(words) if words else f"Conversation {_org_label(org)}"
    if title:
        title = title[0].upper() + title[1:]
    return _clean_title(title) or f"Conversation {_org_label(org)}"


def _build_title_prompt(question: str, answer: str, org: str, intent: str | None) -> str:
    short_answer = str(answer or "")[:900]
    return f"""Tu generes le titre court d'une conversation pour Assistant CDG Prevoyance.

Contraintes:
- francais correct
- 3 a {MAX_TITLE_WORDS} mots
- pas de ponctuation finale
- pas de guillemets
- ne pas inventer un sujet absent
- garder RCAR ou CNRA si utile
- retourner uniquement le titre

Organisme: {_org_label(org)}
Intention: {intent or "retrieval"}
Question utilisateur:
{question}

Reponse assistant:
{short_answer}
"""


def _invoke_title_llm(rag: Any, prompt: str) -> str:
    llm = getattr(rag, "llm", None)
    if llm is None:
        raise RuntimeError("LLM non charge")
    result = llm.invoke(prompt)
    content = getattr(result, "content", result)
    if isinstance(content, list):
        content = " ".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


async def _reserve_title_generation(conversation_id: int) -> bool:
    async with SessionLocal() as db:
        conversation = await db.get(Conversation, conversation_id)
        if conversation is None:
            return False
        if conversation.title_source in {"user", "auto", "auto_generating"}:
            return False
        conversation.title_source = "auto_generating"
        await db.commit()
        return True


async def generate_conversation_title_task(
    conversation_id: int,
    question: str,
    answer: str,
    org: str,
    intent: str | None,
    rag: Any,
) -> None:
    if not await _reserve_title_generation(conversation_id):
        return

    fallback = _fallback_title(question, org)
    title = fallback
    try:
        prompt = _build_title_prompt(question, answer, org, intent)
        raw_title = await asyncio.wait_for(
            run_in_threadpool(_invoke_title_llm, rag, prompt),
            timeout=TITLE_TIMEOUT_SECONDS,
        )
        title = _clean_title(raw_title) or fallback
    except Exception as exc:
        logger.info("Titre conversation fallback utilise: %s", exc)

    async with SessionLocal() as db:
        conversation = await db.get(Conversation, conversation_id)
        if conversation is None:
            return
        if conversation.title_source != "auto_generating":
            return
        conversation.title = title[:255]
        conversation.title_source = "auto"
        conversation.title_generated_at = datetime.utcnow()
        await db.commit()

    api_cache.clear_prefix("admin:conversations")
