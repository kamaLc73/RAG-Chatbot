from __future__ import annotations

import csv
import json
import logging
import math
import os
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger as loguru_logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.logger import setup_logger
from config.settings import LOGS_DIR


load_dotenv(PROJECT_ROOT / ".env")

EVALUATION_DATA_DIR = PROJECT_ROOT / "data" / "evaluation"
RUNS_DIR = EVALUATION_DATA_DIR / "runs"
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


class LoguruForwardHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        loguru_logger.opt(exception=record.exc_info).log(level, record.getMessage())


def configure_logging(source: str = "evaluation") -> None:
    setup_logger(log_dir=LOGS_DIR, source=source)
    logging.basicConfig(
        handlers=[LoguruForwardHandler()],
        level=logging.INFO,
        force=True,
    )


def get_git_metadata() -> dict[str, str]:
    import subprocess

    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={PROJECT_ROOT}", *args],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "branch": run_git(["branch", "--show-current"]),
        "commit": run_git(["rev-parse", "--short", "HEAD"]),
    }


def make_run_dir(output_dir: Path, run_name: str | None, prefix: str) -> tuple[str, str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = run_name or f"{prefix}_{timestamp}"
    run_dir = output_dir / f"{timestamp}_{clean_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return timestamp, clean_name, run_dir


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return float(statistics.mean(clean)) if clean else None


def quantile(values: list[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return clean[lower]
    return clean[lower] + (clean[upper] - clean[lower]) * (pos - lower)


def latency_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in rows if float(row.get("latency_seconds") or 0) > 0]
    return {
        "latency_count": len(latencies),
        "latency_mean": safe_mean(latencies),
        "latency_median": quantile(latencies, 0.5),
        "latency_p95": quantile(latencies, 0.95),
        "latency_max": max(latencies) if latencies else None,
    }


def org_for(row: dict[str, Any]) -> str:
    value = (row.get("org") or row.get("organization") or "").lower()
    if "rcar" in value:
        return "rcar"
    if "cnra" in value:
        return "cnra"
    return "all"


def ranking_metrics(hit_ids: list[str], expected_ids: set[str], k: int) -> dict[str, Any]:
    top_k = hit_ids[:k]
    first_rank = None
    for index, hit_id in enumerate(hit_ids, start=1):
        if hit_id in expected_ids:
            first_rank = index
            break
    matches_at_k = sum(1 for hit_id in top_k if hit_id in expected_ids)
    return {
        "hit_at_1": 1.0 if hit_ids[:1] and hit_ids[0] in expected_ids else 0.0,
        "hit_at_k": 1.0 if matches_at_k > 0 else 0.0,
        "precision_at_k": matches_at_k / max(len(top_k), 1),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "first_match_rank": first_rank or 0,
    }


def summarize_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float | None]:
    return {key: safe_mean([float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]) for key in keys}


def summarize_numeric_by(rows: list[dict[str, Any]], group_key: str, keys: list[str]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get(group_key) or "unknown")
        groups.setdefault(group, []).append(row)
    return {
        group: {
            "count": len(group_rows),
            "metrics": summarize_numeric(group_rows, keys),
        }
        for group, group_rows in sorted(groups.items())
    }


def confusion_counts(rows: list[dict[str, Any]], expected_key: str, predicted_key: str) -> dict[str, dict[str, int]]:
    labels = sorted({str(row.get(expected_key, "")) for row in rows} | {str(row.get(predicted_key, "")) for row in rows})
    matrix = {label: {inner: 0 for inner in labels} for label in labels}
    for row in rows:
        matrix[str(row.get(expected_key, ""))][str(row.get(predicted_key, ""))] += 1
    return matrix


def classification_summary(rows: list[dict[str, Any]], expected_key: str, predicted_key: str) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get(expected_key) == row.get(predicted_key))
    labels = sorted({str(row.get(expected_key, "")) for row in rows})
    per_label = {}
    f1_values = []
    for label in labels:
        tp = sum(1 for row in rows if row.get(expected_key) == label and row.get(predicted_key) == label)
        fp = sum(1 for row in rows if row.get(expected_key) != label and row.get(predicted_key) == label)
        fn = sum(1 for row in rows if row.get(expected_key) == label and row.get(predicted_key) != label)
        support = sum(1 for row in rows if row.get(expected_key) == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": safe_mean(f1_values) or 0.0,
        "per_label": per_label,
        "confusion_matrix": confusion_counts(rows, expected_key, predicted_key),
        "predicted_distribution": dict(Counter(str(row.get(predicted_key, "")) for row in rows)),
        "expected_distribution": dict(Counter(str(row.get(expected_key, "")) for row in rows)),
    }
