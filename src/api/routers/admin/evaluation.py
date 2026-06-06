from __future__ import annotations

import json
import re
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...cache import api_cache
from ...database import get_db
from ...deps import get_current_superuser
from ...models import EvaluationRun
from ...services.evaluation_store import (
    PROJECT_ROOT,
    RUNS_DIR,
    SUMMARY_FILES,
    evaluation_run_dict,
    evaluation_stats,
    metric_options_for_runs,
    sync_evaluation_runs,
)


router = APIRouter(prefix="/admin/evaluation", tags=["admin-evaluation"], dependencies=[Depends(get_current_superuser)])

EVALUATION_CACHE_KEY = "admin:evaluation:runs"
EVALUATION_CACHE_TTL_SECONDS = 60.0
EVALUATION_LOG_DIR = PROJECT_ROOT / "logs" / "evaluation_api"
EVALUATION_SCRIPTS = {
    "docs": PROJECT_ROOT / "src" / "evaluation" / "eval_docs.py",
    "forms": PROJECT_ROOT / "src" / "evaluation" / "eval_forms.py",
    "videos": PROJECT_ROOT / "src" / "evaluation" / "eval_videos.py",
    "intentions": PROJECT_ROOT / "src" / "evaluation" / "eval_intent.py",
}
EVALUATION_DATASETS: dict[str, dict[str, Any]] = {
    "docs": {
        "label": "Documents",
        "path": PROJECT_ROOT / "data" / "evaluation" / "docs_test_set_48.jsonl",
        "columns": ["question", "expected_answer", "organization", "source_type", "category", "source_doc"],
    },
    "forms": {
        "label": "Formulaires",
        "path": PROJECT_ROOT / "data" / "evaluation" / "forms_test_set_60.jsonl",
        "columns": ["query", "expected_titles", "expected_form_ids", "org", "category", "source"],
    },
    "videos": {
        "label": "Vidéos",
        "path": PROJECT_ROOT / "data" / "evaluation" / "videos_test_set_45.jsonl",
        "columns": ["query", "expected_titles", "expected_video_ids", "expected_org", "org", "source"],
    },
    "intentions": {
        "label": "Intentions",
        "path": PROJECT_ROOT / "data" / "evaluation" / "intent_test_set.jsonl",
        "columns": ["query", "expected_intent", "expected_video_gate", "expected_form_gate", "source"],
    },
}


class EvaluationRunRequest(BaseModel):
    task: Literal["docs", "forms", "videos", "intentions"] = "docs"
    mode: Literal["smoke", "full"] = "smoke"
    skip_ragas: bool = True
    limit: int | None = None
    run_name: str | None = None


class EvaluationRunRename(BaseModel):
    run_name: str


def _normalize_run_name(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip()
    if not raw:
        raw = fallback
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    normalized = re.sub(r"_+", "_", normalized)
    return (normalized or fallback)[:120]


def _safe_run_path(run: EvaluationRun) -> Path:
    runs_root = RUNS_DIR.resolve()
    run_path = (PROJECT_ROOT / run.run_dir).resolve()
    if run_path == runs_root or runs_root not in run_path.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chemin de run invalide")
    return run_path


def _summary_path_for_run(run: EvaluationRun) -> Path | None:
    run_path = _safe_run_path(run)
    if not run_path.exists() or not run_path.is_dir():
        return None
    for filename, task in SUMMARY_FILES.items():
        if task == run.task:
            candidate = run_path / filename
            if candidate.exists():
                return candidate
    for filename in SUMMARY_FILES:
        candidate = run_path / filename
        if candidate.exists():
            return candidate
    return None


def _build_evaluation_command(payload: EvaluationRunRequest) -> tuple[str, list[str], Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_name = f"api_{payload.task}_{payload.mode}_{timestamp}"
    run_name = _normalize_run_name(payload.run_name, fallback_name)
    script = EVALUATION_SCRIPTS[payload.task]
    command = [sys.executable, str(script), "--run-name", run_name]

    limit = payload.limit
    if payload.mode == "smoke" and limit is None:
        limit = 3
    if limit is not None:
        command.extend(["--limit", str(max(1, limit))])
    if payload.task == "docs" and payload.skip_ragas:
        command.append("--skip-ragas")

    log_path = EVALUATION_LOG_DIR / f"{run_name}.log"
    return run_name, command, log_path


async def _load_runs(db: AsyncSession, *, force_sync: bool = False) -> list[dict[str, Any]]:
    if not force_sync:
        cached = api_cache.get(EVALUATION_CACHE_KEY)
        if isinstance(cached, list):
            return cached

    if force_sync:
        records = await sync_evaluation_runs(db)
    else:
        result = await db.execute(select(EvaluationRun).order_by(EvaluationRun.created_at_label.desc(), EvaluationRun.id.desc()))
        records = list(result.scalars().all())
        if not records:
            records = await sync_evaluation_runs(db)

    runs = [evaluation_run_dict(record) for record in records]
    api_cache.set(EVALUATION_CACHE_KEY, runs, ttl_seconds=EVALUATION_CACHE_TTL_SECONDS)
    return runs


def _response(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "ok",
        "runs": runs,
        "stats": evaluation_stats(runs),
        "metric_options": metric_options_for_runs(runs),
    }


@router.get("/status")
async def evaluation_status(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    runs = await _load_runs(db)
    return {
        "status": "available",
        "runs": len(runs),
        "message": "Les runs existants sont disponibles depuis data/evaluation/runs.",
    }


@router.get("/runs")
async def list_evaluation_runs(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    runs = await _load_runs(db)
    return _response(runs)


@router.post("/runs/sync")
async def sync_runs(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    api_cache.clear(EVALUATION_CACHE_KEY)
    runs = await _load_runs(db, force_sync=True)
    return _response(runs)


@router.get("/datasets/{task}")
async def get_evaluation_dataset(task: Literal["docs", "forms", "videos", "intentions"]) -> dict[str, Any]:
    config = EVALUATION_DATASETS[task]
    dataset_path = Path(config["path"])
    if not dataset_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset introuvable")

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = {"raw": stripped}
            if isinstance(payload, dict):
                rows.append({"index": index, **payload})
            else:
                rows.append({"index": index, "value": payload})

    return {
        "status": "ok",
        "task": task,
        "task_label": config["label"],
        "path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "total": len(rows),
        "columns": config["columns"],
        "rows": rows,
    }


@router.post("/runs")
async def create_evaluation_run_stub(payload: EvaluationRunRequest) -> dict[str, Any]:
    run_name, command, log_path = _build_evaluation_command(payload)
    EVALUATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            creationflags=creationflags,
        )
    api_cache.clear(EVALUATION_CACHE_KEY)
    return {
        "status": "started",
        "pid": process.pid,
        "run_name": run_name,
        "task": payload.task,
        "mode": payload.mode,
        "skip_ragas": payload.skip_ragas,
        "limit": payload.limit,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "message": "Run lancé en arrière-plan. Synchronise la liste après la fin pour afficher les résultats.",
    }


@router.patch("/runs/{run_id}")
async def rename_evaluation_run(
    run_id: int,
    payload: EvaluationRunRename,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    run = await db.get(EvaluationRun, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run introuvable")

    next_name = _normalize_run_name(payload.run_name, run.run_name)
    run.run_name = next_name
    run.summary_json = {**(run.summary_json or {}), "run_name": next_name}

    summary_path = _summary_path_for_run(run)
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, dict):
            summary["run_name"] = next_name
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    await db.commit()
    await db.refresh(run)
    api_cache.clear(EVALUATION_CACHE_KEY)
    return evaluation_run_dict(run)


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation_run(run_id: int, db: AsyncSession = Depends(get_db)) -> None:
    run = await db.get(EvaluationRun, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run introuvable")

    run_path = _safe_run_path(run)
    if run_path.exists():
        if run_path.is_dir():
            shutil.rmtree(run_path)
        else:
            run_path.unlink()

    await db.delete(run)
    await db.commit()
    api_cache.clear(EVALUATION_CACHE_KEY)


@router.post("/run")
async def run_evaluation_stub() -> dict[str, Any]:
    return {
        "score": 0.0,
        "passed": 0,
        "failed": 0,
        "status": "not_started",
        "message": "Utilisez /admin/evaluation/runs pour préparer un run contrôlé.",
    }
