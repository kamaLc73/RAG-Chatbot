from __future__ import annotations

"""
Evaluation du retrieval des videos YouTube.

Dataset attendu (JSONL):
  {
    "query": "Je cherche une video sur Ehtiyati",
    "expected_video_ids": ["abc123"],
    "expected_titles": ["Titre attendu"],
    "org": "rcar"
  }
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


DATA_FILE_DEFAULT = EVALUATION_DATA_DIR / "videos_test_set_45.jsonl"
TRANSCRIPTS_FILE = PROJECT_ROOT / "data" / "youtube" / "transcripts.json"
DEFAULT_TARGET_HITS = 10
DEFAULT_HITS = 5
DEFAULT_THRESHOLD = None
MIN_TRANSCRIPT_CHARS = 80


def infer_video_org(video: dict[str, Any]) -> str:
    text = f"{video.get('title', '')} {video.get('description', '')}".lower()
    if "cnra" in text and "rcar" not in text:
        return "cnra"
    if "rcar" in text and "cnra" not in text:
        return "rcar"
    return "both"


def make_starter_dataset(path: Path, limit: int, overwrite: bool = False) -> None:
    import json

    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset existe deja: {path}. Utilise --overwrite pour le remplacer.")
    videos = json.loads(TRANSCRIPTS_FILE.read_text(encoding="utf-8"))
    videos = [
        video
        for video in videos
        if video.get("video_id") and video.get("title") and len((video.get("transcript") or "").strip()) >= MIN_TRANSCRIPT_CHARS
    ]

    templates = [
        "Je cherche une video sur {title}",
        "Montre-moi une video qui explique {title}",
        "Video YouTube concernant {title}",
    ]
    ordered_videos = sorted(videos, key=lambda video: (infer_video_org(video), video.get("title", "")))
    rows = []
    variant = 0
    while len(rows) < limit and ordered_videos:
        template = templates[variant % len(templates)]
        for video in ordered_videos:
            if len(rows) >= limit:
                break
            title = video.get("title", "").strip()
            expected_org = infer_video_org(video)
            rows.append(
                {
                    "query": template.format(title=title),
                    "expected_video_ids": [video["video_id"]],
                    "expected_titles": [title],
                    "expected_org": expected_org,
                    "org": "all",
                    "variant": variant + 1,
                    "source": "data/youtube/transcripts.json",
                }
            )
        variant += 1
    write_jsonl(path, rows)
    logger.info("Starter dataset videos ecrit: {} lignes -> {}", len(rows), path)


def evaluate_videos(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.data_file, limit=args.limit)
    from chatbot.rag_pipeline import RAGPipeline

    rag = RAGPipeline(
        vespa_url=args.vespa_url,
        vespa_port=args.vespa_port,
        embedding_model=args.embedding_model,
        enable_intent_classifier=False,
        enable_video_suggestions=True,
        enable_form_suggestions=False,
    )

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        query = row.get("query") or row.get("question") or ""
        org = org_for(row) if args.use_org_filter else "all"
        expected_ids = {str(v).strip() for v in row.get("expected_video_ids", []) if str(v).strip()}
        started = time.perf_counter()
        error = ""
        hits = []
        candidates = []
        try:
            payload = rag.retrieve_videos(
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
        hit_ids = [str(hit.get("video_id", "")) for hit in hits]
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
            "expected_video_ids": sorted(expected_ids),
            "expected_titles": row.get("expected_titles", []),
            "retrieved_video_ids": hit_ids,
            "retrieved_titles": [hit.get("title", "") for hit in hits],
            "retrieved_scores": [hit.get("score", 0.0) for hit in hits],
            "retrieved_vespa_scores": [hit.get("vespa_score", 0.0) for hit in hits],
            "retrieved_rerank_scores": [hit.get("rerank_score", 0.0) for hit in hits],
            "retrieved_ranking_modes": [hit.get("ranking_mode", "") for hit in hits],
            "candidate_video_ids": [hit.get("video_id", "") for hit in candidates],
            "candidate_titles": [hit.get("title", "") for hit in candidates],
            "candidate_scores": [hit.get("score", 0.0) for hit in candidates],
            "candidate_vespa_scores": [hit.get("vespa_score", 0.0) for hit in candidates],
            "candidate_rerank_scores": [hit.get("rerank_score", 0.0) for hit in candidates],
            "candidate_ranking_modes": [hit.get("ranking_mode", "") for hit in candidates],
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
        "task": "videos",
        "dataset": str(args.data_file),
        "dataset_size": len(results),
        "embedding_model": args.embedding_model,
        "threshold": args.threshold if args.threshold is not None else "auto",
        "target_hits": args.target_hits,
        "hits": args.hits,
        "org_filter": "dataset" if args.use_org_filter else "all",
        "errors": sum(1 for row in results if row.get("error")),
        "metrics": summarize_numeric(results, numeric_keys),
        "metrics_by_org": summarize_numeric_by(results, "org", numeric_keys),
        **get_git_metadata(),
    }
    return {"summary": summary, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation du retrieval des videos Vespa")
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
    parser.add_argument("--use-org-filter", action="store_true", help="Utiliser org du dataset au lieu de org=all")
    parser.add_argument("--make-starter-dataset", action="store_true")
    parser.add_argument("--starter-size", type=int, default=45)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging("evaluation")
    if args.make_starter_dataset:
        make_starter_dataset(args.data_file, args.starter_size, overwrite=args.overwrite)
        return 0

    timestamp, run_name, run_dir = make_run_dir(args.output_dir, args.run_name, "videos_eval")
    payload = evaluate_videos(args)
    payload["summary"]["run_name"] = run_name
    payload["summary"]["created_at"] = timestamp
    write_json(run_dir / "videos_summary.json", payload["summary"])
    write_json(run_dir / "videos_results.json", payload["results"])
    write_csv(run_dir / "videos_results.csv", payload["results"])
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Evaluation videos echouee")
        raise
