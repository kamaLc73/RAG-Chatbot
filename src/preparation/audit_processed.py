'''src/preparation/audit_processed.py'''

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from test.processed_utils import (
    EXPECTED_LANGUAGES,
    EXPECTED_SOURCES,
    load_processed_document,
    iter_processed_files,
    resolve_project_path,
)


DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT_PATH = DEFAULT_PROCESSED_ROOT / "processed_quality_report.json"

REQUIRED_METADATA = {
    "doc_id",
    "source",
    "kind",
    "language",
    "language_reason",
    "url",
    "source_path",
    "extraction_method",
}
SUSPICIOUS_TEXT_MARKERS = ("Ã", "ï¿½", "\ufffd")
NOISE_MARKERS = ("Modal", "Fermer", "cookie", "Cookies", "javascript:")


@dataclass
class Issue:
    severity: str
    code: str
    path: str
    detail: str


def _source_from_url(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    if "rcar.ma" in host:
        return "rcar"
    if "cnra.ma" in host:
        return "cnra"
    return "unknown"


def _path_parts(path: Path, processed_root: Path) -> tuple[str, str, str]:
    try:
        relative = path.resolve().relative_to(processed_root.resolve())
    except Exception:
        return "unknown", "unknown", "unknown"
    parts = relative.parts
    source = parts[0].lower() if len(parts) > 0 else "unknown"
    kind = parts[1].lower() if len(parts) > 1 else "unknown"
    language = parts[2].lower() if len(parts) > 2 else "unknown"
    return source, kind, language


def _add_issue(
    issues: list[Issue],
    *,
    severity: str,
    code: str,
    path: Path,
    processed_root: Path,
    detail: str,
) -> None:
    try:
        relative = path.resolve().relative_to(processed_root.resolve())
        issue_path = str(relative).replace("\\", "/")
    except Exception:
        issue_path = str(path)
    issues.append(Issue(severity=severity, code=code, path=issue_path, detail=detail))


def audit_processed(
    *,
    processed_root: Path,
    min_chars: int,
    max_issue_samples: int,
) -> dict:
    files = iter_processed_files(processed_root)
    issues: list[Issue] = []
    doc_ids: Counter[str] = Counter()
    urls: Counter[str] = Counter()
    char_counts: list[int] = []
    per_bucket: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    metadata_presence: Counter[str] = Counter()
    extraction_methods: Counter[str] = Counter()
    source_url_mismatches = 0

    for path in files:
        path_source, path_kind, path_language = _path_parts(path, processed_root)
        bucket = f"{path_source}/{path_kind}/{path_language}"
        per_bucket[bucket]["files"] += 1

        try:
            document = load_processed_document(path, processed_root)
        except Exception as exc:
            _add_issue(
                issues,
                severity="error",
                code="read_or_parse_failed",
                path=path,
                processed_root=processed_root,
                detail=str(exc),
            )
            continue

        text = document.text.strip()
        char_count = len(text)
        char_counts.append(char_count)
        doc_ids[document.doc_id] += 1
        if document.metadata.get("url"):
            urls[document.metadata["url"]] += 1
        extraction_methods[document.extraction_method] += 1

        if char_count == 0:
            per_bucket[bucket]["empty"] += 1
            _add_issue(
                issues,
                severity="error",
                code="empty_text",
                path=path,
                processed_root=processed_root,
                detail="document body is empty",
            )
        elif char_count < min_chars:
            per_bucket[bucket]["short"] += 1
            _add_issue(
                issues,
                severity="warning",
                code="short_text",
                path=path,
                processed_root=processed_root,
                detail=f"{char_count} chars < min_chars={min_chars}",
            )

        missing_metadata = sorted(key for key in REQUIRED_METADATA if not document.metadata.get(key))
        if missing_metadata:
            per_bucket[bucket]["missing_metadata"] += 1
            for key in missing_metadata:
                metadata_presence[key] += 1
            _add_issue(
                issues,
                severity="error",
                code="missing_metadata",
                path=path,
                processed_root=processed_root,
                detail=", ".join(missing_metadata),
            )

        if path_source not in EXPECTED_SOURCES or path_language not in EXPECTED_LANGUAGES or path_kind not in {"page", "pdfs"}:
            per_bucket[bucket]["bad_layout"] += 1
            _add_issue(
                issues,
                severity="error",
                code="bad_path_layout",
                path=path,
                processed_root=processed_root,
                detail=f"source={path_source}, kind={path_kind}, language={path_language}",
            )

        expected_kind = "pdf" if path_kind == "pdfs" else path_kind
        if document.source != path_source:
            _add_issue(
                issues,
                severity="error",
                code="source_mismatch",
                path=path,
                processed_root=processed_root,
                detail=f"metadata={document.source}, path={path_source}",
            )
        if document.kind != expected_kind:
            _add_issue(
                issues,
                severity="error",
                code="kind_mismatch",
                path=path,
                processed_root=processed_root,
                detail=f"metadata={document.kind}, path={expected_kind}",
            )
        if document.language != path_language:
            _add_issue(
                issues,
                severity="warning",
                code="language_mismatch",
                path=path,
                processed_root=processed_root,
                detail=f"metadata={document.language}, path={path_language}",
            )

        url_source = _source_from_url(document.metadata.get("url", ""))
        if url_source in EXPECTED_SOURCES and url_source != document.source:
            source_url_mismatches += 1
            _add_issue(
                issues,
                severity="error",
                code="source_url_mismatch",
                path=path,
                processed_root=processed_root,
                detail=f"metadata source={document.source}, url host source={url_source}, url={document.metadata.get('url', '')}",
            )

        if any(marker in text for marker in SUSPICIOUS_TEXT_MARKERS):
            per_bucket[bucket]["encoding_suspect"] += 1
            _add_issue(
                issues,
                severity="warning",
                code="encoding_suspect",
                path=path,
                processed_root=processed_root,
                detail="text contains mojibake/replacement markers",
            )

        found_noise = [marker for marker in NOISE_MARKERS if marker in text]
        if found_noise:
            per_bucket[bucket]["noise_suspect"] += 1
            _add_issue(
                issues,
                severity="warning",
                code="noise_suspect",
                path=path,
                processed_root=processed_root,
                detail=", ".join(found_noise[:5]),
            )

    duplicate_doc_ids = {key: count for key, count in doc_ids.items() if key and count > 1}
    duplicate_urls = {key: count for key, count in urls.items() if key and count > 1}

    severity_counts = Counter(issue.severity for issue in issues)
    code_counts = Counter(issue.code for issue in issues)
    sorted_issues = sorted(
        issues,
        key=lambda issue: ({"error": 0, "warning": 1}.get(issue.severity, 2), issue.code, issue.path),
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed_root": str(processed_root),
        "thresholds": {"min_chars": min_chars},
        "summary": {
            "files": len(files),
            "total_chars": sum(char_counts),
            "avg_chars": round(statistics.mean(char_counts), 2) if char_counts else 0,
            "median_chars": round(statistics.median(char_counts), 2) if char_counts else 0,
            "min_chars": min(char_counts) if char_counts else 0,
            "max_chars": max(char_counts) if char_counts else 0,
            "issues": len(issues),
            "errors": severity_counts.get("error", 0),
            "warnings": severity_counts.get("warning", 0),
            "duplicate_doc_ids": len(duplicate_doc_ids),
            "duplicate_urls": len(duplicate_urls),
            "source_url_mismatches": source_url_mismatches,
        },
        "per_bucket": {bucket: dict(stats) for bucket, stats in sorted(per_bucket.items())},
        "extraction_methods": dict(extraction_methods),
        "missing_metadata_counts": dict(metadata_presence),
        "issue_counts": dict(code_counts),
        "duplicate_doc_ids": duplicate_doc_ids,
        "duplicate_urls": duplicate_urls,
        "issues_sample": [asdict(issue) for issue in sorted_issues[:max_issue_samples]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit quality of data/processed outputs before Vespa indexing.",
    )
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-issue-samples", type=int, default=250)
    parser.add_argument("--fail-on-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = resolve_project_path(args.processed_root, PROJECT_ROOT)
    output_path = resolve_project_path(args.output, PROJECT_ROOT)

    report = audit_processed(
        processed_root=processed_root,
        min_chars=max(1, args.min_chars),
        max_issue_samples=max(1, args.max_issue_samples),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("Processed quality audit")
    print(f"- root: {processed_root}")
    print(f"- files: {summary['files']}")
    print(f"- chars avg/median/min/max: {summary['avg_chars']} / {summary['median_chars']} / {summary['min_chars']} / {summary['max_chars']}")
    print(f"- issues: {summary['issues']} (errors={summary['errors']}, warnings={summary['warnings']})")
    print(f"- duplicate doc_ids: {summary['duplicate_doc_ids']}")
    print(f"- source/url mismatches: {summary['source_url_mismatches']}")
    print(f"- report: {output_path}")

    if args.fail_on_errors and summary["errors"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
