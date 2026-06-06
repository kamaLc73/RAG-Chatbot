from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EvaluationRun


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / "data" / "evaluation" / "runs"

SUMMARY_FILES = {
    "ragas_summary.json": "docs",
    "forms_summary.json": "forms",
    "videos_summary.json": "videos",
    "intent_summary.json": "intentions",
}

TASK_LABELS = {
    "docs": "Documents",
    "forms": "Formulaires",
    "videos": "Vidéos",
    "intentions": "Intentions",
}

METRIC_LABELS = {
    "primary_score": "Score principal",
    "latency_mean": "Latence moyenne",
    "latency_p95": "Latence p95",
    "errors": "Erreurs",
    "answer_correctness": "Exactitude",
    "faithfulness": "Fidélité",
    "answer_relevancy": "Pertinence",
    "context_precision": "Précision contexte",
    "context_recall": "Rappel contexte",
    "hit_at_1": "Hit@1",
    "hit_at_k": "Hit@K",
    "precision_at_k": "Précision@K",
    "reciprocal_rank": "Rang réciproque",
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _summary_file(run_dir: Path) -> tuple[Path, str] | None:
    for filename, task in SUMMARY_FILES.items():
        path = run_dir / filename
        if path.exists():
            return path, task
    return None


def _infer_backend(run_dir: Path, summary: dict[str, Any]) -> str:
    value = str(
        summary.get("backend")
        or summary.get("vectorstore_backend")
        or summary.get("store")
        or summary.get("retriever")
        or ""
    ).strip().lower()
    if value in {"vespa", "chroma", "chromadb", "pgvector"}:
        return "chroma" if value == "chromadb" else value

    name = f"{run_dir.name} {summary.get('run_name', '')}".lower()
    if "pgvector" in name:
        return "pgvector"
    if "chroma" in name:
        return "chroma"
    if "vespa" in name:
        return "vespa"
    return "unknown"


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric_dict(summary: dict[str, Any], task: str) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "errors": _int(summary.get("rag_errors") if task == "docs" else summary.get("errors")) or 0,
    }

    if task == "docs":
        ragas = summary.get("ragas_summary") if isinstance(summary.get("ragas_summary"), dict) else {}
        for key in (
            "answer_correctness",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_similarity",
        ):
            metrics[key] = _float(ragas.get(key))
        metrics["latency_mean"] = _float(summary.get("latency_mean"))
        metrics["latency_median"] = _float(summary.get("latency_median"))
        metrics["latency_p95"] = _float(summary.get("latency_p95"))
        metrics["latency_max"] = _float(summary.get("latency_max"))
        metrics["primary_score"] = _float(ragas.get("answer_correctness"))
        return metrics

    if task in {"forms", "videos"}:
        raw_metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        for key in ("hit_at_1", "hit_at_k", "precision_at_k", "reciprocal_rank"):
            metrics[key] = _float(raw_metrics.get(key))
        metrics["latency_mean"] = _float(raw_metrics.get("latency_seconds"))
        metrics["primary_score"] = metrics.get("hit_at_1")
        return metrics

    raw_metrics = summary.get("intent_metrics") if isinstance(summary.get("intent_metrics"), dict) else {}
    for key in ("accuracy", "macro_f1"):
        metrics[key] = _float(raw_metrics.get(key))
    metrics["correct"] = _int(raw_metrics.get("correct"))
    metrics["total"] = _int(raw_metrics.get("total"))
    metrics["primary_score"] = metrics.get("accuracy")
    return metrics


def _primary_label(task: str) -> str:
    if task == "docs":
        return "Exactitude"
    if task in {"forms", "videos"}:
        return "Hit@1"
    if task == "intentions":
        return "Accuracy"
    return "Score"


def _normalized_breakdowns(summary: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "docs":
        return {
            "by_org": summary.get("ragas_summary_by_org") or {},
            "by_category": summary.get("ragas_summary_by_category") or {},
            "by_source_type": summary.get("ragas_summary_by_source_type") or {},
        }
    if task in {"forms", "videos"}:
        return {"by_org": summary.get("metrics_by_org") or {}}
    raw_metrics = summary.get("intent_metrics") if isinstance(summary.get("intent_metrics"), dict) else {}
    return {
        "per_label": raw_metrics.get("per_label") or {},
        "confusion_matrix": raw_metrics.get("confusion_matrix") or {},
    }


def normalize_summary(run_dir: Path, summary_path: Path, task: str, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = _metric_dict(summary, task)
    stat = summary_path.stat()
    created_at_label = str(summary.get("created_at") or run_dir.name.split("_", 2)[0] if run_dir.name else "")
    if "_" in run_dir.name and not summary.get("created_at"):
        parts = run_dir.name.split("_")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            created_at_label = f"{parts[0]}_{parts[1]}"

    return {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "run_name": str(summary.get("run_name") or run_dir.name),
        "task": task,
        "task_label": TASK_LABELS.get(task, task),
        "backend": _infer_backend(run_dir, summary),
        "status": "completed",
        "dataset": summary.get("dataset"),
        "dataset_size": _int(summary.get("dataset_size")),
        "branch": summary.get("branch"),
        "commit": summary.get("commit"),
        "created_at": created_at_label,
        "file_updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "metrics": metrics,
        "primary_score": metrics.get("primary_score"),
        "primary_label": _primary_label(task),
        "breakdowns": _normalized_breakdowns(summary, task),
        "summary": summary,
    }


def scan_evaluation_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted((item for item in RUNS_DIR.iterdir() if item.is_dir()), key=lambda path: path.name):
        match = _summary_file(run_dir)
        if match is None:
            continue
        summary_path, task = match
        summary = _read_json(summary_path)
        if not summary:
            continue
        runs.append(normalize_summary(run_dir, summary_path, task, summary))

    return sorted(runs, key=lambda item: str(item.get("created_at") or item.get("run_name") or ""), reverse=True)


async def sync_evaluation_runs(db: AsyncSession) -> list[EvaluationRun]:
    normalized_runs = scan_evaluation_runs()
    if not normalized_runs:
        return []

    result = await db.execute(select(EvaluationRun))
    existing_by_dir = {run.run_dir: run for run in result.scalars().all()}
    now = datetime.utcnow()

    for item in normalized_runs:
        run_dir = str(item["run_dir"])
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        file_updated_at = None
        if item.get("file_updated_at"):
            try:
                file_updated_at = datetime.fromisoformat(str(item["file_updated_at"]))
            except ValueError:
                file_updated_at = None

        run = existing_by_dir.get(run_dir)
        if run is None:
            run = EvaluationRun(run_dir=run_dir, run_name=str(item["run_name"]), task=str(item["task"]))
            db.add(run)

        run.run_name = str(item["run_name"])
        run.task = str(item["task"])
        run.backend = str(item["backend"])
        run.status = str(item.get("status") or "completed")
        run.dataset = str(item["dataset"]) if item.get("dataset") else None
        run.dataset_size = _int(item.get("dataset_size"))
        run.branch = str(item["branch"]) if item.get("branch") else None
        run.commit = str(item["commit"]) if item.get("commit") else None
        run.created_at_label = str(item["created_at"]) if item.get("created_at") else None
        run.file_updated_at = file_updated_at
        run.metrics_json = metrics
        run.summary_json = summary
        run.synced_at = now

    await db.commit()
    result = await db.execute(select(EvaluationRun).order_by(EvaluationRun.created_at_label.desc(), EvaluationRun.id.desc()))
    return list(result.scalars().all())


def evaluation_run_dict(run: EvaluationRun) -> dict[str, Any]:
    metrics = run.metrics_json or {}
    summary = run.summary_json or {}
    task = run.task
    primary_score = metrics.get("primary_score")
    return {
        "id": str(run.id),
        "run_dir": run.run_dir,
        "run_name": run.run_name,
        "task": task,
        "task_label": TASK_LABELS.get(task, task),
        "backend": run.backend,
        "status": run.status,
        "dataset": run.dataset,
        "dataset_size": run.dataset_size,
        "branch": run.branch,
        "commit": run.commit,
        "created_at": run.created_at_label,
        "file_updated_at": run.file_updated_at.isoformat() if run.file_updated_at else None,
        "synced_at": run.synced_at.isoformat() if run.synced_at else None,
        "metrics": metrics,
        "primary_score": primary_score,
        "primary_label": _primary_label(task),
        "breakdowns": _normalized_breakdowns(summary, task),
        "summary": summary,
    }


def evaluation_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {task: 0 for task in TASK_LABELS}
    for run in runs:
        task = str(run.get("task") or "")
        if task in by_task:
            by_task[task] += 1

    docs_scores = [
        _float((run.get("metrics") or {}).get("answer_correctness"))
        for run in runs
        if run.get("task") == "docs"
    ]
    resource_scores = [
        _float((run.get("metrics") or {}).get("hit_at_1"))
        for run in runs
        if run.get("task") in {"forms", "videos"}
    ]
    intent_scores = [
        _float((run.get("metrics") or {}).get("accuracy"))
        for run in runs
        if run.get("task") == "intentions"
    ]
    latencies = [
        _float((run.get("metrics") or {}).get("latency_mean"))
        for run in runs
        if _float((run.get("metrics") or {}).get("latency_mean")) is not None
    ]
    errors = [
        _int((run.get("metrics") or {}).get("errors")) or 0
        for run in runs
    ]

    def best(values: list[float | None]) -> float | None:
        available = [value for value in values if value is not None]
        return max(available) if available else None

    return {
        "total_runs": len(runs),
        "by_task": by_task,
        "latest_run": runs[0]["run_name"] if runs else None,
        "best_docs_score": best(docs_scores),
        "best_resource_hit_at_1": best(resource_scores),
        "best_intent_accuracy": best(intent_scores),
        "avg_latency": mean([value for value in latencies if value is not None]) if latencies else None,
        "total_errors": sum(errors),
    }


def metric_options_for_runs(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    available: set[str] = set()
    for run in runs:
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        for key, value in metrics.items():
            if value is not None:
                available.add(str(key))
    ordered = [
        "primary_score",
        "latency_mean",
        "latency_p95",
        "errors",
        "answer_correctness",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "hit_at_1",
        "hit_at_k",
        "precision_at_k",
        "reciprocal_rank",
        "accuracy",
        "macro_f1",
    ]
    return [
        {"key": key, "label": METRIC_LABELS.get(key, key)}
        for key in ordered
        if key in available
    ]
