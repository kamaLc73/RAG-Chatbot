from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from pathlib import Path

from ...deps import get_current_superuser


router = APIRouter(prefix="/admin/intents", tags=["admin-intents"], dependencies=[Depends(get_current_superuser)])


class IntentCheckRequest(BaseModel):
    query: str
    org: str = "all"


@router.post("/classify")
def classify_intent(payload: IntentCheckRequest) -> dict[str, object]:
    try:
        from src.chatbot.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        classifier = getattr(rag, "intent_classifier", None)
        if classifier is None or not getattr(classifier, "is_loaded", False):
            return {"status": "unavailable", "intent": "retrieval", "confidence": 0.0}
        classification = rag._classify_intent(payload.query)  # Existing internal API used only for admin diagnostics.
        return {"status": "ok", **classification}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


@router.get("")
def list_intents() -> list[dict[str, object]]:
    intents_dir = Path("data/intents")
    if not intents_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(intents_dir.iterdir()):
        if not path.is_dir():
            continue
        examples = sum(1 for file_path in path.rglob("*") if file_path.is_file())
        rows.append({"id": path.name, "name": path.name, "examples": examples, "enabled": True})
    return rows


@router.get("/status")
def intent_status() -> dict[str, object]:
    return {"status": "stub", "message": "Intent administration is read-only in this API layer."}
