from __future__ import annotations

"""
Evaluation du classifier d'intents.

Dataset attendu (JSONL):
  {
    "query": "Je cherche le formulaire de pension",
    "expected_intent": "needs_form",
    "expected_video_gate": false,
    "expected_form_gate": true,
    "source": "manual"
  }

Les champs expected_video_gate / expected_form_gate sont optionnels. S'ils sont
absents, ils sont derives de expected_intent:
  - needs_video -> expected_video_gate=true
  - needs_form  -> expected_form_gate=true
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from eval_common import (
    EVALUATION_DATA_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    classification_summary,
    configure_logging,
    get_git_metadata,
    load_jsonl,
    make_run_dir,
    write_csv,
    write_json,
    write_jsonl,
)


DATA_FILE_DEFAULT = EVALUATION_DATA_DIR / "intent_test_set.jsonl"
INTENTS_DIR_DEFAULT = PROJECT_ROOT / "data" / "intents"
DEFAULT_EXAMPLES_PER_INTENT = 45


def _bool_label(value: bool) -> str:
    return "true" if bool(value) else "false"


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "oui", "y"}:
        return True
    if normalized in {"0", "false", "no", "non", "n"}:
        return False
    return default


def make_starter_dataset(
    path: Path,
    intents_dir: Path,
    examples_per_intent: int,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset existe deja: {path}. Utilise --overwrite pour le remplacer.")
    if not intents_dir.exists():
        raise FileNotFoundError(f"Dossier intents introuvable: {intents_dir}")

    rows: list[dict[str, Any]] = []
    for suggk_file in sorted(intents_dir.glob("*/suggk.txt")):
        intent = suggk_file.parent.name
        examples = [
            line.strip()
            for line in suggk_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for index, query in enumerate(examples[:examples_per_intent], start=1):
            rows.append(
                {
                    "query": query,
                    "expected_intent": intent,
                    "expected_video_gate": intent == "needs_video",
                    "expected_form_gate": intent == "needs_form",
                    "intent_example_index": index,
                    "source": "data/intents examples - replace with holdout before reporting",
                }
            )

    write_jsonl(path, rows)
    logger.info("Starter dataset intents ecrit: {} lignes -> {}", len(rows), path)


def evaluate_intents(args: argparse.Namespace) -> dict[str, Any]:
    from chatbot.intent_classifier import IntentClassifier

    rows = load_jsonl(args.data_file, limit=args.limit)
    classifier = IntentClassifier(intents_dir=args.intents_dir)
    stats = classifier.get_stats()

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        query = row.get("query") or row.get("question") or ""
        expected_intent = str(row.get("expected_intent") or row.get("intent") or "retrieval")
        expected_video_gate = _coerce_bool(row.get("expected_video_gate"), expected_intent == "needs_video")
        expected_form_gate = _coerce_bool(row.get("expected_form_gate"), expected_intent == "needs_form")

        started = time.perf_counter()
        error = ""
        classification: dict[str, Any]
        try:
            classification = classifier.classify(query)
        except Exception as exc:
            error = repr(exc)
            classification = {
                "intent": "error",
                "confidence": 0.0,
                "tier": 0,
                "reasoning": error,
                "loaded": False,
            }
        latency = time.perf_counter() - started

        predicted_intent = str(classification.get("intent") or "retrieval")
        predicted_video_gate = bool(
            classifier.has_video_signal(query)
            and classifier.should_retrieve_videos(classification)
        )
        predicted_form_gate = bool(
            classifier.has_form_signal(query)
            and classifier.should_retrieve_forms(classification)
        )

        result = {
            "index": index,
            "query": query,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": expected_intent == predicted_intent,
            "confidence": float(classification.get("confidence") or 0.0),
            "tier": classification.get("tier"),
            "reasoning": classification.get("reasoning", ""),
            "loaded": bool(classification.get("loaded", False)),
            "expected_video_gate": _bool_label(expected_video_gate),
            "predicted_video_gate": _bool_label(predicted_video_gate),
            "video_gate_correct": expected_video_gate == predicted_video_gate,
            "expected_form_gate": _bool_label(expected_form_gate),
            "predicted_form_gate": _bool_label(predicted_form_gate),
            "form_gate_correct": expected_form_gate == predicted_form_gate,
            "has_video_signal": _bool_label(classifier.has_video_signal(query)),
            "has_form_signal": _bool_label(classifier.has_form_signal(query)),
            "latency_seconds": latency,
            "error": error,
            "source": row.get("source", ""),
        }
        logger.info(
            "[{}/{}] intent={} expected={} ok={} video_gate={} form_gate={} | {}",
            index,
            len(rows),
            predicted_intent,
            expected_intent,
            result["intent_correct"],
            predicted_video_gate,
            predicted_form_gate,
            query[:90],
        )
        results.append(result)

    intent_metrics = classification_summary(results, "expected_intent", "predicted_intent")
    video_gate_metrics = classification_summary(results, "expected_video_gate", "predicted_video_gate")
    form_gate_metrics = classification_summary(results, "expected_form_gate", "predicted_form_gate")

    summary = {
        "task": "intent",
        "dataset": str(args.data_file),
        "dataset_size": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "classifier_stats": stats,
        "intent_metrics": intent_metrics,
        "video_gate_metrics": video_gate_metrics,
        "form_gate_metrics": form_gate_metrics,
        "latency_mean": (
            sum(float(row["latency_seconds"]) for row in results) / len(results)
            if results
            else 0.0
        ),
        **get_git_metadata(),
    }
    return {"summary": summary, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation du classifier d'intents")
    parser.add_argument("--data-file", type=Path, default=DATA_FILE_DEFAULT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--intents-dir", type=Path, default=INTENTS_DIR_DEFAULT)
    parser.add_argument("--make-starter-dataset", action="store_true")
    parser.add_argument("--examples-per-intent", type=int, default=DEFAULT_EXAMPLES_PER_INTENT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging("evaluation")
    if args.make_starter_dataset:
        make_starter_dataset(
            args.data_file,
            args.intents_dir,
            examples_per_intent=args.examples_per_intent,
            overwrite=args.overwrite,
        )
        return 0

    timestamp, run_name, run_dir = make_run_dir(args.output_dir, args.run_name, "intent_eval")
    payload = evaluate_intents(args)
    payload["summary"]["run_name"] = run_name
    payload["summary"]["created_at"] = timestamp
    write_json(run_dir / "intent_summary.json", payload["summary"])
    write_json(run_dir / "intent_results.json", payload["results"])
    write_csv(run_dir / "intent_results.csv", payload["results"])
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Evaluation intents echouee")
        raise
