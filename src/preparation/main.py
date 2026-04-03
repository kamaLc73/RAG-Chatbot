"""
src/preparation/main.py
=======================
Preparation entrypoint.

Behavior:
- Prepare pages and PDFs from raw data
- Keep OCR queue only in memory
- Run OCR automatically after native extraction
- Write a JSON report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add src directory to path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import setup_logger
from config.settings import DATA_DIR, LOGS_DIR, RAW_DIR, SOURCES
from preparation.pipeline import (
    run_preparation,
    write_minimal_report,
)


def parse_args() -> argparse.Namespace:
    source_choices = [*SOURCES.keys(), "all", "both"]

    parser = argparse.ArgumentParser(
        description="Unified preparation pipeline for RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
            python src/preparation/main.py --source all
            python src/preparation/main.py --source cnra --output-format md
            python src/preparation/main.py --source all --min-native-chars 300
        """,
    )
    parser.add_argument(
        "--source",
        choices=source_choices,
        default="all",
        help="Source to process or all/both for every source.",
    )
    parser.add_argument(
        "--min-native-chars",
        type=int,
        default=250,
        help="Minimum extracted native chars to accept direct text output.",
    )
    parser.add_argument(
        "--output-format",
        choices=["txt", "md"],
        default="md",
        help="Unified text output format for pages and PDFs.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Emit progress logs every N processed items.",
    )
    parser.add_argument(
        "--ocr-languages",
        default="fra+ara",
        help="Tesseract OCR language string (example: fra+ara).",
    )
    parser.add_argument(
        "--ocr-config",
        default="--oem 3 --psm 6",
        help="Tesseract OCR config string.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Rendering DPI used before OCR.",
    )
    parser.add_argument(
        "--max-ocr-pages",
        type=int,
        default=0,
        help="Maximum pages per PDF for OCR (0 = all pages).",
    )
    return parser.parse_args()


def resolve_sources(source_arg: str) -> list[str]:
    if source_arg in {"all", "both"}:
        return list(SOURCES.keys())
    return [source_arg]


def main() -> None:
    args = parse_args()

    setup_logger(log_dir=LOGS_DIR, source="preparation")

    sources = resolve_sources(args.source)
    processed_dir = DATA_DIR / "processed"

    logger.info("Preparation start for sources: {}", ", ".join(s.upper() for s in sources))
    logger.info("Output format: .{}", args.output_format)
    logger.info("Min native chars: {}", max(0, args.min_native_chars))
    logger.info("OCR engine: Siwar Tesseract")
    logger.info("OCR languages: {}", args.ocr_languages)
    logger.info("OCR config: {}", args.ocr_config)
    logger.info("OCR DPI: {}", max(72, args.dpi))
    logger.info("OCR max pages: {}", max(0, args.max_ocr_pages))

    report = run_preparation(
        raw_dir=RAW_DIR,
        processed_dir=processed_dir,
        sources=sources,
        output_format=args.output_format,
        min_native_chars=max(0, args.min_native_chars),
        ocr_languages=args.ocr_languages,
        ocr_config=args.ocr_config,
        dpi=max(72, args.dpi),
        max_ocr_pages=max(0, args.max_ocr_pages),
        log_every=max(1, args.log_every),
    )

    report_path = write_minimal_report(processed_dir=processed_dir, report=report)

    logger.info("Preparation completed")
    logger.info("Page failures total: {}", report["totals"]["page_failures"])
    logger.info("PDF failures total : {}", report["totals"]["pdf_failures"])
    logger.info(
        "OCR totals: queued={} attempted={} succeeded={} failed={}",
        report["totals"]["ocr"]["queued"],
        report["totals"]["ocr"]["attempted"],
        report["totals"]["ocr"]["succeeded"],
        report["totals"]["ocr"]["failed"],
    )
    logger.info("Tracked errors total: {}", report["totals"]["errors"]["count"])
    logger.info("JSON report: {}", report_path)


if __name__ == "__main__":
    main()
