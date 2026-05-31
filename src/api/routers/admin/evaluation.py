from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import get_current_superuser


router = APIRouter(prefix="/admin/evaluation", tags=["admin-evaluation"], dependencies=[Depends(get_current_superuser)])


@router.get("/status")
def evaluation_status() -> dict[str, object]:
    return {
        "status": "available",
        "message": "Evaluation runs are exposed as an admin stub; launch existing evaluation scripts from the backend host.",
    }


@router.post("/runs")
def create_evaluation_run_stub() -> dict[str, object]:
    return {
        "status": "not_started",
        "message": "No evaluation job was started. Use src/evaluation scripts for controlled benchmark runs.",
    }


@router.post("/run")
def run_evaluation_stub() -> dict[str, object]:
    return {"score": 0.0, "passed": 0, "failed": 0, "status": "not_started"}
