from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import get_current_superuser


router = APIRouter(prefix="/admin/kb", tags=["admin-kb"], dependencies=[Depends(get_current_superuser)])


@router.get("/status")
def knowledge_base_status() -> dict[str, object]:
    try:
        from src.store.vespa_store import count_schema, make_vespa_app

        app = make_vespa_app()
        schemas = {
            "docs": count_schema(app, "doc"),
            "videos": count_schema(app, "video"),
            "forms": count_schema(app, "form"),
        }
        return {"status": "ok", "schemas": schemas}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "schemas": {}}


@router.get("/stats")
def knowledge_base_stats() -> dict[str, object]:
    return knowledge_base_status()


@router.get("")
def list_kb_stub() -> list[dict[str, object]]:
    status = knowledge_base_status()
    schemas = status.get("schemas", {})
    if not isinstance(schemas, dict):
        schemas = {}
    return [
        {"id": "doc", "title": "Documents", "source": "Vespa schema doc", "status": str(schemas.get("docs", 0))},
        {"id": "form", "title": "Formulaires", "source": "Vespa schema form", "status": str(schemas.get("forms", 0))},
        {"id": "video", "title": "Videos", "source": "Vespa schema video", "status": str(schemas.get("videos", 0))},
    ]


@router.post("/reindex")
def reindex_stub() -> dict[str, object]:
    return {
        "status": "not_started",
        "message": "Reindexing is intentionally not launched by the API stub. Run the existing indexers from the backend host.",
    }
