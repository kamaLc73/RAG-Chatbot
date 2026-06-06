from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

from ...cache import api_cache
from ...database import get_db
from ...deps import get_current_superuser
from ...models import IntentDefinition, IntentExample
from ...services.intent_store import (
    count_intent_examples,
    example_to_dict,
    export_intents_to_files,
    intent_to_dict,
    load_active_examples_by_intent,
    seed_intents_from_files,
)


router = APIRouter(prefix="/admin/intents", tags=["admin-intents"], dependencies=[Depends(get_current_superuser)])


class IntentCheckRequest(BaseModel):
    query: str
    org: str = "all"


class IntentUpdateRequest(BaseModel):
    enabled: bool | None = None
    label_fr: str | None = None
    description: str | None = None


class IntentExampleCreateRequest(BaseModel):
    text: str


class IntentExampleUpdateRequest(BaseModel):
    text: str | None = None
    enabled: bool | None = None


async def _get_intent_by_name(db: AsyncSession, intent_name: str) -> IntentDefinition:
    result = await db.execute(
        select(IntentDefinition)
        .options(selectinload(IntentDefinition.examples))
        .where(IntentDefinition.name == intent_name)
    )
    intent = result.scalar_one_or_none()
    if intent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Intention introuvable")
    return intent


async def _reload_runtime_classifier(request: Request, db: AsyncSession) -> dict[str, object]:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        return {"status": "skipped", "reason": "RAGPipeline non chargé"}

    examples_by_intent = await load_active_examples_by_intent(db)
    if not examples_by_intent:
        return {"status": "skipped", "reason": "Aucun exemple actif"}

    from src.chatbot.intent_classifier import IntentClassifier

    start = time.perf_counter()
    classifier = await run_in_threadpool(
        IntentClassifier,
        shared_embeddings=getattr(rag, "embeddings", None),
        examples_by_intent=examples_by_intent,
    )
    if not getattr(classifier, "is_loaded", False):
        return {"status": "failed", "reason": "Classificateur non chargé"}

    rag.intent_classifier = classifier
    payload = {
        "status": "ok",
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
        "examples": sum(len(examples) for examples in examples_by_intent.values()),
        "intents": len(examples_by_intent),
        "loaded": True,
    }
    request.app.state.intent_last_reload = payload
    return payload


async def _persist_and_reload_intents(request: Request, db: AsyncSession) -> dict[str, object]:
    await export_intents_to_files(db)
    api_cache.clear("admin:stats")
    return await _reload_runtime_classifier(request, db)


@router.post("/sync")
async def sync_intents(request: Request, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    await seed_intents_from_files(db)
    reload_status = await _persist_and_reload_intents(request, db)
    return {"status": "ok", "message": "Intentions synchronisées depuis data/intents.", "reload": reload_status}


@router.get("")
async def list_intents(db: AsyncSession = Depends(get_db)) -> list[dict[str, object]]:
    result = await db.execute(
        select(IntentDefinition)
        .options(selectinload(IntentDefinition.examples))
        .order_by(IntentDefinition.name.asc())
    )
    return [intent_to_dict(intent) for intent in result.scalars().all()]


@router.get("/status")
async def intent_status(request: Request, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    rag = getattr(request.app.state, "rag", None)
    classifier = getattr(rag, "intent_classifier", None)
    stats = classifier.get_stats() if classifier is not None and getattr(classifier, "is_loaded", False) else {"loaded": False}
    examples = await count_intent_examples(db)
    return {
        "status": "ok",
        "runtime": stats,
        "database": examples,
        "last_reload": getattr(request.app.state, "intent_last_reload", None),
    }


@router.post("/classify")
def classify_intent(payload: IntentCheckRequest, request: Request) -> dict[str, object]:
    try:
        rag = getattr(request.app.state, "rag", None)
        classifier = getattr(rag, "intent_classifier", None)
        if rag is None or classifier is None or not getattr(classifier, "is_loaded", False):
            return {"status": "unavailable", "intent": "retrieval", "confidence": 0.0}
        classification = rag._classify_intent(payload.query)  # Existing internal API used only for admin diagnostics.
        return {"status": "ok", **classification}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


@router.post("/reload")
async def reload_intents(request: Request, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    await export_intents_to_files(db)
    payload = await _reload_runtime_classifier(request, db)
    if payload.get("status") != "ok":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(payload.get("reason", "Rechargement impossible")))
    return payload


@router.patch("/{intent_name}")
async def update_intent(intent_name: str, payload: IntentUpdateRequest, request: Request, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    intent = await _get_intent_by_name(db, intent_name)
    if payload.enabled is not None:
        intent.enabled = payload.enabled
    if payload.label_fr is not None:
        clean_label = payload.label_fr.strip()
        if clean_label:
            intent.label_fr = clean_label
    if payload.description is not None:
        intent.description = payload.description.strip() or None
    await db.commit()
    await db.refresh(intent)
    await _persist_and_reload_intents(request, db)
    return intent_to_dict(intent)


@router.get("/{intent_name}/examples")
async def list_examples(intent_name: str, db: AsyncSession = Depends(get_db)) -> list[dict[str, object]]:
    intent = await _get_intent_by_name(db, intent_name)
    return [example_to_dict(example) for example in sorted(intent.examples, key=lambda item: (not item.enabled, item.id))]


@router.post("/{intent_name}/examples", status_code=status.HTTP_201_CREATED)
async def create_example(
    intent_name: str,
    payload: IntentExampleCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    intent = await _get_intent_by_name(db, intent_name)
    text = payload.text.strip()
    if len(text) < 3:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Exemple trop court")

    example = IntentExample(intent_id=intent.id, text=text, enabled=True, source="ui")
    db.add(example)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Cet exemple existe déjà") from exc
    await db.refresh(example)
    await _persist_and_reload_intents(request, db)
    return example_to_dict(example)


@router.patch("/examples/{example_id}")
async def update_example(
    example_id: int,
    payload: IntentExampleUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    result = await db.execute(select(IntentExample).where(IntentExample.id == example_id))
    example = result.scalar_one_or_none()
    if example is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exemple introuvable")

    if payload.text is not None:
        text = payload.text.strip()
        if len(text) < 3:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Exemple trop court")
        example.text = text
    if payload.enabled is not None:
        example.enabled = payload.enabled

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Cet exemple existe déjà") from exc
    await db.refresh(example)
    await _persist_and_reload_intents(request, db)
    return example_to_dict(example)


@router.delete("/examples/{example_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_example(example_id: int, request: Request, db: AsyncSession = Depends(get_db)) -> None:
    result = await db.execute(select(IntentExample).where(IntentExample.id == example_id))
    example = result.scalar_one_or_none()
    if example is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exemple introuvable")
    await db.delete(example)
    await db.commit()
    await _persist_and_reload_intents(request, db)
