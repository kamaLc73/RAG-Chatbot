from __future__ import annotations

"""
Evaluation du retrieval des formulaires.

Dataset attendu (JSONL):
  {
    "query": "Je cherche le formulaire demande de pension CRAC",
    "expected_form_ids": ["cnra_i_crac_demande_de_pension_crac"],
    "expected_titles": ["DEMANDE DE PENSION CRAC"],
    "org": "cnra",
    "category": "Imprimes CRAC"
  }

Les IDs peuvent etre les IDs bruts de forms.json ou les IDs Vespa `form::<id>`.
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
    DEFAULT_EMBEDDING_MODEL,
    EVALUATION_DATA_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    configure_logging,
    get_git_metadata,
    load_jsonl,
    make_run_dir,
    org_for,
    ranking_metrics,
    summarize_numeric,
    summarize_numeric_by,
    write_csv,
    write_json,
    write_jsonl,
)
from store.vespa_store import stable_data_id


DATA_FILE_DEFAULT = EVALUATION_DATA_DIR / "forms_test_set_60.jsonl"
FORMS_CATALOG = PROJECT_ROOT / "data" / "forms" / "forms.json"
DEFAULT_TARGET_HITS = 10
DEFAULT_HITS = 5
DEFAULT_THRESHOLD = None


def normalize_form_id(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("form::"):
        return value
    return stable_data_id("form", value)


def make_starter_dataset(path: Path, limit: int, overwrite: bool = False) -> None:
    import json

    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset existe deja: {path}. Utilise --overwrite pour le remplacer.")
    forms = json.loads(FORMS_CATALOG.read_text(encoding="utf-8"))
    forms = [form for form in forms if form.get("form_id") and form.get("title")]

    templates = [
        "Je cherche le formulaire {title}",
        "Où trouver le formulaire {title} ?",
        "Télécharger le formulaire {title}",
        "J'ai besoin du document {title}",
    ]
    ordered_forms = sorted(forms, key=lambda form: ((form.get("org") or "").lower(), form.get("title", "")))
    rows = []
    variant = 0
    while len(rows) < limit and ordered_forms:
        template = templates[variant % len(templates)]
        for form in ordered_forms:
            if len(rows) >= limit:
                break
            title = form.get("title", "").strip()
            org = (form.get("org") or "all").lower()
            rows.append(
                {
                    "query": template.format(title=title),
                    "expected_form_ids": [form["form_id"]],
                    "expected_titles": [title],
                    "org": org if org in {"rcar", "cnra"} else "all",
                    "category": form.get("category", ""),
                    "variant": variant + 1,
                    "source": "data/forms/forms.json",
                }
            )
        variant += 1
    write_jsonl(path, rows)
    logger.info("Starter dataset formulaires ecrit: {} lignes -> {}", len(rows), path)


def evaluate_forms(args: argparse.Namespace) -> dict[str, Any]:
    from chatbot.rag_pipeline import RAGPipeline

    rows = load_jsonl(args.data_file, limit=args.limit)
    rag = RAGPipeline(
        vespa_url=args.vespa_url,
        vespa_port=args.vespa_port,
        embedding_model=args.embedding_model,
        enable_intent_classifier=False,
        enable_video_suggestions=False,
        enable_form_suggestions=True,
    )

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        query = row.get("query") or row.get("question") or ""
        org = org_for(row)
        expected_ids = {normalize_form_id(v) for v in row.get("expected_form_ids", [])}
        started = time.perf_counter()
        error = ""
        hits = []
        candidates = []
        try:
            payload = rag.retrieve_forms(
                query,
                org=org,
                limit=args.hits,
                target_hits=args.target_hits,
                threshold=args.threshold,
                include_debug=True,
            )
            hits = payload["results"]
            candidates = payload["candidates"]
        except Exception as exc:
            error = repr(exc)

        latency = time.perf_counter() - started
        hit_ids = [str(hit.get("form_id", "")) for hit in hits]
        metrics = ranking_metrics(hit_ids, expected_ids, args.hits) if expected_ids else {
            "hit_at_1": 0.0,
            "hit_at_k": 0.0,
            "precision_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "first_match_rank": 0,
        }
        result = {
            "index": index,
            "query": query,
            "org": org,
            "expected_form_ids": sorted(expected_ids),
            "expected_titles": row.get("expected_titles", []),
            "retrieved_form_ids": hit_ids,
            "retrieved_titles": [hit.get("title", "") for hit in hits],
            "retrieved_scores": [hit.get("score", 0.0) for hit in hits],
            "retrieved_vespa_scores": [hit.get("vespa_score", 0.0) for hit in hits],
            "retrieved_rerank_scores": [hit.get("rerank_score", 0.0) for hit in hits],
            "candidate_form_ids": [hit.get("form_id", "") for hit in candidates],
            "candidate_titles": [hit.get("title", "") for hit in candidates],
            "candidate_scores": [hit.get("score", 0.0) for hit in candidates],
            "candidate_vespa_scores": [hit.get("vespa_score", 0.0) for hit in candidates],
            "top_title": hits[0].get("title", "") if hits else "",
            "top_score": hits[0].get("score", 0.0) if hits else 0.0,
            "latency_seconds": latency,
            "error": error,
            **metrics,
        }
        logger.info(
            f"[{index}/{len(rows)}] hit@1={result['hit_at_1']:.0f} "
            f"hit@k={result['hit_at_k']:.0f} rank={result['first_match_rank']} | {query[:90]}"
        )
        results.append(result)

    numeric_keys = ["hit_at_1", "hit_at_k", "precision_at_k", "reciprocal_rank", "latency_seconds"]
    summary = {
        "task": "forms",
        "dataset": str(args.data_file),
        "dataset_size": len(results),
        "embedding_model": args.embedding_model,
        "threshold": args.threshold if args.threshold is not None else "auto",
        "target_hits": args.target_hits,
        "hits": args.hits,
        "errors": sum(1 for row in results if row.get("error")),
        "metrics": summarize_numeric(results, numeric_keys),
        "metrics_by_org": summarize_numeric_by(results, "org", numeric_keys),
        **get_git_metadata(),
    }
    return {"summary": summary, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation du retrieval des formulaires Vespa")
    parser.add_argument("--data-file", type=Path, default=DATA_FILE_DEFAULT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--vespa-url", default=os.getenv("VESPA_URL", "http://localhost"))
    parser.add_argument("--vespa-port", type=int, default=int(os.getenv("VESPA_PORT", "8080")))
    parser.add_argument("--target-hits", type=int, default=DEFAULT_TARGET_HITS)
    parser.add_argument("--hits", type=int, default=DEFAULT_HITS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--vespa-timeout", default="3s")
    parser.add_argument("--make-starter-dataset", action="store_true")
    parser.add_argument("--starter-size", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging("evaluation")
    if args.make_starter_dataset:
        make_starter_dataset(args.data_file, args.starter_size, overwrite=args.overwrite)
        return 0

    timestamp, run_name, run_dir = make_run_dir(args.output_dir, args.run_name, "forms_eval")
    payload = evaluate_forms(args)
    payload["summary"]["run_name"] = run_name
    payload["summary"]["created_at"] = timestamp
    write_json(run_dir / "forms_summary.json", payload["summary"])
    write_json(run_dir / "forms_results.json", payload["results"])
    write_csv(run_dir / "forms_results.csv", payload["results"])
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Evaluation forms echouee")
        raise
