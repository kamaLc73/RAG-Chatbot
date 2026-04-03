"""
src/crawler/main.py
===================
Main entry point to run the crawler.

Usage:
    python src/crawler/main.py --source cnra
    python src/crawler/main.py --source rcar
    python src/crawler/main.py --source all
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ajout du dossier racine dans sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import setup_logger
from config.settings import CRAWLER_CONFIG, LOGS_DIR, PDF_CONFIG, RAW_DIR, SOURCES
from crawler.crawler import Crawler
from loguru import logger


def parse_args() -> argparse.Namespace:
    source_choices = [*SOURCES.keys(), "all", "both"]

    parser = argparse.ArgumentParser(
        description="Crawl multi-sources (FR/AR) pour le projet RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py --source cnra
  python main.py --source all
  python main.py --source all --max-pages 20
  python main.py --source rcar --no-pdfs
        """,
    )
    parser.add_argument(
        "--source",
        choices=source_choices,
        default="all",
        help="Source a crawler (nom de SOURCES) ou all/both pour toutes",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Nombre max de pages par source (0 = illimite)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=0,
        help="Profondeur max de crawl (0 = illimite)",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=0,
        help="Nombre max de PDFs a traiter par source (0 = illimite)",
    )
    parser.add_argument(
        "--no-pdfs",
        action="store_true",
        help="Desactiver le telechargement des PDFs",
    )
    parser.add_argument(
        "--production-mode",
        action="store_true",
        help="Mode production: saute automatiquement les URLs login/inscription/compte",
    )
    parser.add_argument(
        "--parallel-sources",
        dest="parallel_sources",
        action="store_true",
        default=True,
        help="Crawler plusieurs sources en parallele (active par defaut)",
    )
    parser.add_argument(
        "--sequential-sources",
        dest="parallel_sources",
        action="store_false",
        help="Desactiver le parallelisme des sources et executer en sequentiel",
    )
    parser.add_argument(
        "--source-parallelism",
        type=int,
        default=0,
        help="Nombre max de sources crawlees en parallele (0 = valeur config)",
    )
    return parser.parse_args()


def save_report(all_stats: dict, output_dir: Path) -> None:
    report = {
        "crawl_date": datetime.now(timezone.utc).isoformat(),
        "sources": all_stats,
        "totals": {
            "pages_crawled": sum(s["pages_crawled"] for s in all_stats.values()),
            "pages_failed": sum(s["pages_failed"] for s in all_stats.values()),
            "pdfs_found": sum(s["pdfs_found"] for s in all_stats.values()),
            "pdfs_downloaded": sum(s["pdfs_downloaded"] for s in all_stats.values()),
            "pdfs_failed": sum(s["pdfs_failed"] for s in all_stats.values()),
        },
    }

    report_path = output_dir / "crawl_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Rapport sauvegarde : {report_path}")
    logger.info(f"Total pages  : {report['totals']['pages_crawled']}")
    logger.info(f"Total PDFs   : {report['totals']['pdfs_downloaded']}")


async def crawl_one_source(
    source_name: str,
    base_url: str,
    config: dict[str, Any],
    pdf_cfg: dict[str, Any],
    disable_pdfs: bool,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[str, dict[str, Any]]:
    async def _run() -> tuple[str, dict[str, Any]]:
        logger.info(f"Source : {source_name.upper()} -> {base_url}")

        crawl_runner = Crawler(
            source=source_name,
            base_url=base_url,
            raw_dir=RAW_DIR,
            crawler_config=config,
            pdf_config=pdf_cfg if not disable_pdfs else {**pdf_cfg, "max_pdf_size_bytes": 0},
        )

        stats = await crawl_runner.run()

        return source_name, stats

    if semaphore is None:
        return await _run()

    async with semaphore:
        return await _run()


async def run_crawler() -> None:
    args = parse_args()

    setup_logger(log_dir=LOGS_DIR, source="crawl")

    if args.source in {"all", "both"}:
        sources_to_crawl = list(SOURCES.items())
    else:
        sources_to_crawl = [(args.source, SOURCES[args.source])]

    config = CRAWLER_CONFIG.copy()
    if args.max_pages:
        config["max_pages_per_source"] = args.max_pages
        config["max_concurrency"] = 1
    if args.max_depth:
        config["max_depth"] = args.max_depth
    if args.max_pdfs:
        config["max_pdfs_per_source"] = args.max_pdfs
    if args.production_mode:
        config["production_mode"] = True
        logger.info("Mode production active")

    pdf_cfg = PDF_CONFIG.copy()
    if args.no_pdfs:
        logger.info("Mode --no-pdfs active")

    all_stats: dict[str, dict[str, Any]] = {}

    if args.parallel_sources and len(sources_to_crawl) > 1:
        parallelism = args.source_parallelism or int(config.get("source_parallelism", 2))
        parallelism = max(1, parallelism)
        logger.info(f"Mode parallele sources active (max {parallelism})")
        semaphore = asyncio.Semaphore(parallelism)
        tasks = [
            crawl_one_source(
                source_name=source_name,
                base_url=base_url,
                config=config.copy(),
                pdf_cfg=pdf_cfg.copy(),
                disable_pdfs=args.no_pdfs,
                semaphore=semaphore,
            )
            for source_name, base_url in sources_to_crawl
        ]
        results = await asyncio.gather(*tasks)
        for source_name, stats in results:
            all_stats[source_name] = stats
    else:
        for source_name, base_url in sources_to_crawl:
            source_name_result, stats = await crawl_one_source(
                source_name=source_name,
                base_url=base_url,
                config=config.copy(),
                pdf_cfg=pdf_cfg.copy(),
                disable_pdfs=args.no_pdfs,
            )
            all_stats[source_name_result] = stats

            if len(sources_to_crawl) > 1:
                await asyncio.sleep(3)

    save_report(all_stats, RAW_DIR)


def main() -> None:
    asyncio.run(run_crawler())


if __name__ == "__main__":
    main()