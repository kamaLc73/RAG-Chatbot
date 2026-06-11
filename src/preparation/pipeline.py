"""
src/preparation/pipeline.py
===========================
Simple preparation pipeline for RAG.

Behavior:
- Prepare pages and PDFs from data/raw
- Separate PDF processing in two clear phases:
  1) native text extraction (scanned/native phase)
  2) OCR extraction (OCR phase)
- Keep OCR queue only in memory (no pending PDF copies)
- Write outputs under data/processed/<source>/...
- Write JSON report with failures, OCR stats, and error samples

OCR implementation:
- Mistral OCR for scanned PDFs through the Mistral OCR HTTP API
- Legacy Siwar/Tesseract OCR remains available only when explicitly selected
"""

from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

# Optional dependencies.
try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

try:
    import fitz  # pymupdf

    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

Image = None
ImageEnhance = None
ImageFilter = None
try:
    from PIL import Image, ImageEnhance, ImageFilter

    HAS_PIL = True
except Exception:
    HAS_PIL = False

np = None
try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

pytesseract = None
try:
    import pytesseract

    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

try:
    import requests

    HAS_REQUESTS = True
except Exception as exc:
    requests = None
    HAS_REQUESTS = False


MAX_ERROR_SAMPLES_PER_SOURCE = 50
DEFAULT_MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest").strip() or "mistral-ocr-latest"
DEFAULT_MISTRAL_OCR_PAGES_PER_MINUTE = int(os.getenv("MISTRAL_OCR_PAGES_PER_MINUTE", "550"))
DEFAULT_MISTRAL_OCR_TIMEOUT_SECONDS = int(os.getenv("MISTRAL_OCR_TIMEOUT_SECONDS", "300"))
MISTRAL_PASSWORD_SIGNALS = (
    "password",
    "encrypted",
    "protected",
    "forbidden",
    "401",
    "403",
)

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
AR_PATH_RE = re.compile(r"(^|/)ar(/|$)", re.I)
FR_PATH_RE = re.compile(r"(^|/)fr(/|$)", re.I)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

_TESSERACT_CONFIGURED = False


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_name(value: str) -> str:
    candidate = SAFE_NAME_RE.sub("_", value.strip())
    candidate = candidate.strip("._-")
    return candidate or "doc"


def _push_error_sample(
    bucket: list[dict[str, str]],
    *,
    stage: str,
    path: str,
    reason: str,
    max_samples: int = MAX_ERROR_SAMPLES_PER_SOURCE,
) -> None:
    if len(bucket) >= max_samples:
        return

    compact_reason = (reason or "").strip().replace("\n", " ")
    if len(compact_reason) > 240:
        compact_reason = compact_reason[:237] + "..."

    bucket.append(
        {
            "stage": stage,
            "path": path,
            "reason": compact_reason,
        }
    )


def _normalize_hint(value: str | None) -> str:
    candidate = (value or "").strip().lower()
    if candidate.startswith("ar"):
        return "ar"
    if candidate.startswith("fr"):
        return "fr"
    return "unknown"


def _path_language_hint(url: str) -> str | None:
    path = (urlparse(url).path or "")
    if AR_PATH_RE.search(path):
        return "ar"
    if FR_PATH_RE.search(path):
        return "fr"
    return None


def _detect_script_language(text: str | None) -> str | None:
    sample = (text or "").strip()
    if len(sample) < 30:
        return None

    arabic_count = len(ARABIC_CHAR_RE.findall(sample))
    latin_count = len(LATIN_CHAR_RE.findall(sample))

    if arabic_count >= 18 and arabic_count >= int(latin_count * 1.2):
        return "ar"
    if latin_count >= 18 and latin_count >= int(arabic_count * 1.2):
        return "fr"
    return None


def classify_languages(
    *,
    text: str | None,
    title: str | None,
    url: str,
    html_language_hint: str | None,
    extra_hints: list[str] | None,
) -> tuple[list[str], str]:
    """
    Return one or two language buckets.
    If language is unclear, duplicate into both fr and ar.
    """
    by_script = _detect_script_language(f"{title or ''}\n{text or ''}")
    if by_script in {"fr", "ar"}:
        return [by_script], "content_script"

    html_hint = _normalize_hint(html_language_hint)
    if html_hint in {"fr", "ar"}:
        return [html_hint], "html_lang_hint"

    for hint in (extra_hints or []):
        normalized = _normalize_hint(hint)
        if normalized in {"fr", "ar"}:
            return [normalized], "metadata_hint"

    path_hint = _path_language_hint(url)
    if path_hint in {"fr", "ar"}:
        return [path_hint], "url_path_hint"

    return ["fr", "ar"], "ambiguous_dual_bucket"


def _iter_page_files(raw_dir: Path, source: str) -> list[Path]:
    pages_root = raw_dir / source / "pages"
    if not pages_root.exists():
        return []
    return sorted(pages_root.rglob("page_*.json"))


def _iter_pdf_meta_files(raw_dir: Path, source: str) -> list[Path]:
    pdf_root = raw_dir / source / "pdfs"
    if not pdf_root.exists():
        return []
    return sorted(pdf_root.rglob("*_meta.json"))


def _resolve_pdf_file(meta_path: Path, meta: dict[str, Any], pdf_root: Path) -> Path | None:
    local_path = str(meta.get("local_path", "")).strip()
    if local_path:
        candidate = Path(local_path)
        if candidate.exists():
            return candidate

    filename = str(meta.get("filename", "")).strip()
    if filename:
        in_same_dir = meta_path.parent / filename
        if in_same_dir.exists():
            return in_same_dir

        for candidate in pdf_root.rglob(filename):
            if candidate.exists():
                return candidate

    return None


def _build_output_path(
    processed_dir: Path,
    kind: str,
    source: str,
    language: str,
    doc_id: str,
    output_format: str,
    filename_prefix: str | None = None,
) -> Path:
    ext = "md" if output_format == "md" else "txt"
    safe_doc_id = _safe_name(doc_id)
    safe_prefix = _safe_name(filename_prefix or "").strip("._-")
    file_name = f"{safe_prefix}_{safe_doc_id}.{ext}" if safe_prefix else f"{safe_doc_id}.{ext}"
    normalized_kind = (kind or "").strip().lower()
    kind_dir = "page" if normalized_kind in {"page", "pages"} else "pdfs"

    # Source-first tree:
    # processed/<source>/page/<lang>/...
    # processed/<source>/pdfs/<lang>/...
    return processed_dir / source / kind_dir / language / file_name


def _render_text_document(
    *,
    title: str,
    content: str,
    metadata: dict[str, Any],
    output_format: str,
) -> str:
    safe_title = (title or metadata.get("doc_id") or "document").strip()
    body = (content or "").strip()

    if output_format == "md":
        header_lines = [
            "---",
            f"doc_id: {metadata.get('doc_id', '')}",
            f"source: {metadata.get('source', '')}",
            f"kind: {metadata.get('kind', '')}",
            f"language: {metadata.get('language', '')}",
            f"language_reason: {metadata.get('language_reason', '')}",
            f"url: {metadata.get('url', '')}",
            f"source_path: {metadata.get('source_path', '')}",
            f"extraction_method: {metadata.get('extraction_method', '')}",
            "---",
            "",
            f"# {safe_title}",
            "",
        ]
        return "\n".join(header_lines) + body + "\n"

    header_lines = [
        f"doc_id: {metadata.get('doc_id', '')}",
        f"source: {metadata.get('source', '')}",
        f"kind: {metadata.get('kind', '')}",
        f"language: {metadata.get('language', '')}",
        f"language_reason: {metadata.get('language_reason', '')}",
        f"url: {metadata.get('url', '')}",
        f"source_path: {metadata.get('source_path', '')}",
        f"extraction_method: {metadata.get('extraction_method', '')}",
        "",
        safe_title,
        "=" * len(safe_title),
        "",
    ]
    return "\n".join(header_lines) + body + "\n"


def _extract_native_pdf_text(pdf_path: Path) -> tuple[str, str]:
    if HAS_PDFPLUMBER:
        try:
            texts: list[str] = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text() or "")
            merged = "\n\n".join(texts).strip()
            if merged:
                return merged, "pdfplumber"
        except Exception:
            pass

    if HAS_PYMUPDF:
        try:
            texts = []
            doc = fitz.open(str(pdf_path))
            for page in doc:
                texts.append(page.get_text("text") or "")
            doc.close()
            merged = "\n\n".join(texts).strip()
            if merged:
                return merged, "pymupdf"
        except Exception:
            pass

    return "", "none"


def _configure_tesseract_from_siwar() -> bool:
    """
    Configure pytesseract using:
    1) bundled Tesseract under src/preparation/Tesseract-OCR
    2) legacy Siwar folder location (transition compatibility)
    3) system tesseract from PATH
    """
    global _TESSERACT_CONFIGURED

    if _TESSERACT_CONFIGURED:
        return True
    if not HAS_PYTESSERACT:
        return False

    script_dir = Path(__file__).resolve().parent
    bundled_cmd_primary = script_dir / "Tesseract-OCR" / "tesseract.exe"
    bundled_cmd_legacy = script_dir / "Siwar-Image-Reader" / "Tesseract-OCR" / "tesseract.exe"

    candidates: list[Path] = []
    if bundled_cmd_primary.exists():
        candidates.append(bundled_cmd_primary)
    if bundled_cmd_legacy.exists():
        candidates.append(bundled_cmd_legacy)

    system_cmd = shutil.which("tesseract")
    if system_cmd:
        candidates.append(Path(system_cmd))

    for cmd in candidates:
        if not cmd.exists():
            continue

        try:
            pytesseract.pytesseract.tesseract_cmd = str(cmd)
            tessdata_dir = cmd.parent / "tessdata"
            if tessdata_dir.exists():
                os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)

            _TESSERACT_CONFIGURED = True
            logger.info("Tesseract configured: {}", cmd)
            return True
        except Exception:
            continue

    return False


def _deskew_image(np_image: Any) -> Any:
    """
    Siwar-style deskew based on principal axis of foreground pixels.
    """
    if not HAS_NUMPY or not HAS_PIL:
        return Image.fromarray(np_image)

    try:
        binary = np_image > 0
        coords = np.column_stack(np.where(binary))
        if coords.size == 0:
            return Image.fromarray(np_image)

        cov = np.cov(coords.T)
        evals, evecs = np.linalg.eigh(cov)
        principal_vector = evecs[:, np.argsort(evals)[::-1][0]]
        angle = np.arctan2(principal_vector[1], principal_vector[0]) * (180.0 / np.pi)

        pil_image = Image.fromarray(np_image)
        return pil_image.rotate(-float(angle), expand=True, fillcolor=255)
    except Exception:
        return Image.fromarray(np_image)


def _preprocess_pdf_page(image: Any) -> Any:
    """
    Siwar-style preprocessing:
    grayscale -> contrast -> threshold -> sharpen -> deskew.
    """
    if not (HAS_PIL and HAS_NUMPY):
        return image

    gray = image.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2)
    gray = gray.point(lambda x: 0 if x < 140 else 255, "1")
    gray = gray.filter(ImageFilter.SHARPEN)

    np_image = np.array(gray).astype("uint8") * 255
    return _deskew_image(np_image)


def _correct_orientation(image: Any) -> Any:
    """
    Siwar-style orientation correction using Tesseract OSD.
    """
    if not HAS_PYTESSERACT:
        return image

    try:
        osd = pytesseract.image_to_osd(image)
        rotate = 0
        for line in str(osd).splitlines():
            if "Rotate" in line:
                rotate = int(line.split(":", 1)[1].strip())
                break
        if rotate != 0:
            image = image.rotate(-rotate, expand=True, fillcolor=255)
    except Exception:
        pass
    return image


def _extract_ocr_pdf_text_siwar(
    pdf_path: Path,
    *,
    ocr_languages: str,
    ocr_config: str,
    dpi: int,
    max_pages: int,
) -> tuple[str, str]:
    if not (HAS_PYMUPDF and HAS_PIL and HAS_NUMPY and HAS_PYTESSERACT):
        return "", "ocr_unavailable"

    if not _configure_tesseract_from_siwar():
        return "", "ocr_unavailable"

    try:
        zoom = max(1.0, float(dpi) / 72.0)
        matrix = fitz.Matrix(zoom, zoom)

        doc = fitz.open(str(pdf_path))
        page_limit = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)

        page_texts: list[str] = []
        for page_index in range(page_limit):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            prepared = _preprocess_pdf_page(image)
            prepared = _correct_orientation(prepared)

            text = pytesseract.image_to_string(
                prepared,
                lang=ocr_languages,
                config=ocr_config,
            )
            cleaned = (text or "").strip()
            if cleaned:
                page_texts.append(f"--- Page {page_index + 1} ---\n{cleaned}")

        doc.close()
        merged = "\n\n".join(page_texts).strip()
        return merged, "ocr_tesseract_siwar"

    except Exception as exc:
        logger.warning("Siwar OCR failed for {}: {}", pdf_path, exc)
        return "", "ocr_failed"


def _is_mistral_password_protected_error(error: str | None) -> bool:
    if not error:
        return False
    error_lower = error.lower()
    return any(signal in error_lower for signal in MISTRAL_PASSWORD_SIGNALS)


def _count_pdf_pages(pdf_path: Path) -> int:
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(str(pdf_path))
            page_count = int(doc.page_count)
            doc.close()
            return max(0, page_count)
        except Exception:
            pass

    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return max(0, len(pdf.pages))
        except Exception:
            pass

    return 0


class _MistralOcrPageRateLimiter:
    def __init__(self, pages_per_minute: int) -> None:
        self.pages_per_minute = max(1, int(pages_per_minute))
        self._events: list[tuple[float, int]] = []

    def wait_for(self, page_count: int) -> None:
        pages = max(1, int(page_count or 1))
        if pages > self.pages_per_minute:
            logger.warning(
                "Mistral OCR PDF has {} pages, above the configured {} pages/minute limit. "
                "This single request may still hit the provider limit.",
                pages,
                self.pages_per_minute,
            )

        while True:
            now = time.monotonic()
            self._events = [
                (timestamp, event_pages)
                for timestamp, event_pages in self._events
                if now - timestamp < 60.0
            ]
            used_pages = sum(event_pages for _, event_pages in self._events)
            if used_pages + pages <= self.pages_per_minute or pages > self.pages_per_minute:
                self._events.append((now, pages))
                return

            oldest_timestamp = min(timestamp for timestamp, _ in self._events)
            sleep_seconds = max(0.5, 60.0 - (now - oldest_timestamp) + 0.5)
            logger.info(
                "Mistral OCR page rate limit: used={} incoming={} limit={}/min. Sleeping {:.1f}s.",
                used_pages,
                pages,
                self.pages_per_minute,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)


def _extract_ocr_pdf_text_mistral(
    pdf_path: Path,
    *,
    model: str,
    delay: float,
    output_dir: Path,
    timeout_seconds: int,
) -> tuple[str, str, str | None]:
    if not HAS_REQUESTS or requests is None:
        return "", "ocr_mistral_unavailable", "requests_unavailable"

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        return "", "ocr_mistral_unavailable", "mistral_api_key_missing"

    try:
        pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode()
        payload = {
            "model": model or DEFAULT_MISTRAL_OCR_MODEL,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
            "include_image_base64": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if delay > 0:
            time.sleep(delay)

        response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers=headers,
            json=payload,
            timeout=max(30, int(timeout_seconds)),
        )
        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text
            return "", "ocr_mistral_failed", f"HTTP {response.status_code}: {error_body}"

        data = response.json()
        page_texts: list[str] = []
        for page in data.get("pages", []):
            if not isinstance(page, dict):
                continue
            page_text = str(page.get("markdown") or page.get("text") or "").strip()
            if page_text:
                page_texts.append(page_text)

        text = "\n\n---\n\n".join(page_texts).strip()
        if not text:
            text = str(data.get("text") or "").strip()
        if not text:
            return "", "ocr_mistral_empty", "empty_text"

        return text, "ocr_mistral", None

    except Exception as exc:
        logger.warning("Mistral OCR failed for {}: {}", pdf_path, exc)
        return "", "ocr_mistral_failed", str(exc)


def _write_pdf_text_outputs(
    *,
    processed_dir: Path,
    source: str,
    output_format: str,
    job: dict[str, Any],
    pdf_path: Path,
    text: str,
    extraction_method: str,
    filename_prefix: str,
) -> None:
    languages, reason = classify_languages(
        text=text,
        title=str(job.get("title", "")),
        url=str(job.get("source_url", "")),
        html_language_hint=None,
        extra_hints=[
            str(job.get("source_page_language", "")),
            str(job.get("crawler_pdf_url_language_hint", "")),
        ],
    )

    for language in languages:
        output_path = _build_output_path(
            processed_dir=processed_dir,
            kind="pdfs",
            source=source,
            language=language,
            doc_id=str(job.get("doc_id", "pdf_doc")),
            output_format=output_format,
            filename_prefix=filename_prefix,
        )
        rendered = _render_text_document(
            title=str(job.get("title", "")),
            content=text,
            metadata={
                "doc_id": str(job.get("doc_id", "pdf_doc")),
                "source": source,
                "kind": "pdf",
                "language": language,
                "language_reason": reason,
                "url": str(job.get("source_url", "")),
                "source_path": str(pdf_path),
                "extraction_method": extraction_method,
            },
            output_format=output_format,
        )
        _write_text(output_path, rendered)


def prepare_source_pages(
    *,
    raw_dir: Path,
    processed_dir: Path,
    source: str,
    output_format: str,
    log_every: int,
) -> dict[str, Any]:
    page_failures = 0
    error_samples: list[dict[str, str]] = []
    page_files = _iter_page_files(raw_dir=raw_dir, source=source)
    total = len(page_files)
    logger.info("[{}] pages found: {}", source.upper(), total)

    for index, page_path in enumerate(page_files, start=1):
        try:
            data = _read_json(page_path)
            if not data:
                page_failures += 1
                _push_error_sample(
                    error_samples,
                    stage="page_read",
                    path=str(page_path),
                    reason="invalid_or_unreadable_json",
                )
                continue

            doc_id = str(data.get("id") or f"{source}_page_{hashlib.md5(str(page_path).encode()).hexdigest()[:12]}")
            url = str(data.get("url", ""))
            title = str(data.get("title", ""))
            content = str(data.get("content_markdown") or data.get("content_text") or "").strip()
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}

            if not content:
                page_failures += 1
                _push_error_sample(
                    error_samples,
                    stage="page_content",
                    path=str(page_path),
                    reason="empty_content",
                )
                continue

            languages, reason = classify_languages(
                text=content,
                title=title,
                url=url,
                html_language_hint=str(metadata.get("html_language_hint", "")),
                extra_hints=[
                    str(metadata.get("source_page_language_hint", "")),
                    str(data.get("language", "")),
                ],
            )

            for language in languages:
                output_path = _build_output_path(
                    processed_dir=processed_dir,
                    kind="pages",
                    source=source,
                    language=language,
                    doc_id=doc_id,
                    output_format=output_format,
                    filename_prefix="page",
                )
                rendered = _render_text_document(
                    title=title,
                    content=content,
                    metadata={
                        "doc_id": doc_id,
                        "source": source,
                        "kind": "page",
                        "language": language,
                        "language_reason": reason,
                        "url": url,
                        "source_path": str(page_path),
                        "extraction_method": "html_clean_text",
                    },
                    output_format=output_format,
                )
                _write_text(output_path, rendered)

            if index % max(1, log_every) == 0 or index == total:
                logger.info("[{}] pages progress: {}/{}", source.upper(), index, total)

        except Exception as exc:
            page_failures += 1
            logger.warning("[{}] page preparation failed for {}: {}", source.upper(), page_path, exc)
            _push_error_sample(
                error_samples,
                stage="page_prepare",
                path=str(page_path),
                reason=str(exc),
            )

    logger.info("[{}] pages done, failures={}", source.upper(), page_failures)
    return {
        "total": total,
        "failures": page_failures,
        "error_samples": error_samples,
    }


def prepare_source_scanned_pdfs(
    *,
    raw_dir: Path,
    processed_dir: Path,
    source: str,
    min_native_chars: int,
    output_format: str,
    log_every: int,
) -> dict[str, Any]:
    """
    Phase 1: prepare native/scanned PDFs with direct text extraction.
    Any PDF that cannot provide enough text is queued for OCR phase.
    """
    pdf_failures = 0
    error_samples: list[dict[str, str]] = []
    ocr_jobs: list[dict[str, Any]] = []
    native_success = 0

    pdf_root = raw_dir / source / "pdfs"
    meta_files = _iter_pdf_meta_files(raw_dir=raw_dir, source=source)
    total = len(meta_files)
    logger.info("[{}] pdf meta found: {}", source.upper(), total)

    for index, meta_path in enumerate(meta_files, start=1):
        try:
            meta = _read_json(meta_path)
            if not meta:
                pdf_failures += 1
                _push_error_sample(
                    error_samples,
                    stage="pdf_meta_read",
                    path=str(meta_path),
                    reason="invalid_or_unreadable_json",
                )
                continue

            doc_id = str(meta.get("id") or f"{source}_pdf_{hashlib.md5(str(meta_path).encode()).hexdigest()[:12]}")
            title = str(meta.get("title", ""))
            source_url = str(meta.get("source_url", ""))

            pdf_path = _resolve_pdf_file(meta_path=meta_path, meta=meta, pdf_root=pdf_root)
            if not pdf_path or not pdf_path.exists():
                pdf_failures += 1
                _push_error_sample(
                    error_samples,
                    stage="pdf_file_missing",
                    path=str(meta_path),
                    reason="missing_pdf_file",
                )
                continue

            native_text, native_method = _extract_native_pdf_text(pdf_path)
            if len(native_text.strip()) >= max(0, min_native_chars):
                native_success += 1
                languages, reason = classify_languages(
                    text=native_text,
                    title=title,
                    url=source_url,
                    html_language_hint=None,
                    extra_hints=[
                        str(meta.get("source_page_language", "")),
                        str(meta.get("crawler_pdf_url_language_hint", "")),
                    ],
                )

                for language in languages:
                    output_path = _build_output_path(
                        processed_dir=processed_dir,
                        kind="pdfs",
                        source=source,
                        language=language,
                        doc_id=doc_id,
                        output_format=output_format,
                        filename_prefix="pdf_native",
                    )
                    rendered = _render_text_document(
                        title=title,
                        content=native_text,
                        metadata={
                            "doc_id": doc_id,
                            "source": source,
                            "kind": "pdf",
                            "language": language,
                            "language_reason": reason,
                            "url": source_url,
                            "source_path": str(pdf_path),
                            "extraction_method": native_method,
                        },
                        output_format=output_format,
                    )
                    _write_text(output_path, rendered)
            else:
                ocr_jobs.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "source_url": source_url,
                        "pdf_path": str(pdf_path),
                        "source_page_language": str(meta.get("source_page_language", "")),
                        "crawler_pdf_url_language_hint": str(meta.get("crawler_pdf_url_language_hint", "")),
                    }
                )

            if index % max(1, log_every) == 0 or index == total:
                logger.info(
                    "[{}] pdf native progress: {}/{} (ocr_queue={})",
                    source.upper(),
                    index,
                    total,
                    len(ocr_jobs),
                )

        except Exception as exc:
            pdf_failures += 1
            logger.warning("[{}] pdf native preparation failed for {}: {}", source.upper(), meta_path, exc)
            _push_error_sample(
                error_samples,
                stage="pdf_native_prepare",
                path=str(meta_path),
                reason=str(exc),
            )

    logger.info("[{}] native pdf phase done, failures={}", source.upper(), pdf_failures)
    return {
        "total": total,
        "native_success": native_success,
        "failures": pdf_failures,
        "ocr_jobs": ocr_jobs,
        "error_samples": error_samples,
    }


def prepare_source_ocr_pdfs(
    *,
    processed_dir: Path,
    source: str,
    output_format: str,
    ocr_jobs: list[dict[str, Any]],
    ocr_engine: str = "mistral",
    mistral_model: str = DEFAULT_MISTRAL_OCR_MODEL,
    ocr_delay: float = 2.0,
    ocr_max_retries: int = 2,
    mistral_ocr_pages_per_minute: int = DEFAULT_MISTRAL_OCR_PAGES_PER_MINUTE,
    mistral_ocr_timeout_seconds: int = DEFAULT_MISTRAL_OCR_TIMEOUT_SECONDS,
    ocr_artifacts_dir: Path | None = None,
    ocr_languages: str = "fra+ara",
    ocr_config: str = "--oem 3 --psm 6",
    dpi: int = 250,
    max_ocr_pages: int = 0,
    log_every: int = 20,
) -> dict[str, Any]:
    """
    Phase 2: prepare OCR PDFs from in-memory queue.
    """
    pdf_failures = 0
    error_samples: list[dict[str, str]] = []
    engine = (ocr_engine or "mistral").strip().lower()
    if engine in {"mistral-ocr", "mistral_ocr"}:
        engine = "mistral"
    if engine not in {"mistral", "tesseract"}:
        engine = "mistral"

    ocr_stats = {
        "engine": "mistral_ocr" if engine == "mistral" else "siwar_tesseract",
        "queued": len(ocr_jobs),
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
    }

    if not ocr_jobs:
        logger.info("[{}] no OCR jobs", source.upper())
        return {
            "failures": 0,
            "ocr_stats": ocr_stats,
            "error_samples": error_samples,
        }

    if engine == "mistral":
        api_key_available = bool(os.getenv("MISTRAL_API_KEY", "").strip())
        if not HAS_REQUESTS or not api_key_available:
            pdf_failures = len(ocr_jobs)
            ocr_stats["attempted"] = len(ocr_jobs)
            ocr_stats["failed"] = len(ocr_jobs)
            reason = (
                "requests_unavailable"
                if not HAS_REQUESTS
                else "mistral_api_key_missing"
            )
            _push_error_sample(
                error_samples,
                stage="ocr_setup",
                path=source,
                reason=reason,
            )
            logger.warning(
                "[{}] Mistral OCR unavailable. Marked {} OCR jobs as failures.",
                source.upper(),
                len(ocr_jobs),
            )
            return {
                "failures": pdf_failures,
                "ocr_stats": ocr_stats,
                "error_samples": error_samples,
            }

        artifacts_dir = ocr_artifacts_dir or (processed_dir / "_ocr_artifacts" / source)
        page_limiter = _MistralOcrPageRateLimiter(
            pages_per_minute=max(1, mistral_ocr_pages_per_minute)
        )
        logger.info("[{}] Mistral OCR memory queue size: {}", source.upper(), len(ocr_jobs))
        logger.info(
            "[{}] Mistral OCR page limit: {}/min, timeout={}s",
            source.upper(),
            max(1, mistral_ocr_pages_per_minute),
            max(30, mistral_ocr_timeout_seconds),
        )

        for index, job in enumerate(ocr_jobs, start=1):
            ocr_stats["attempted"] += 1
            pdf_path = Path(str(job.get("pdf_path", "")))
            estimated_pages = _count_pdf_pages(pdf_path) or 1
            success = False
            last_error = ""

            for attempt in range(1, max(1, ocr_max_retries) + 1):
                retry_delay = 0.0 if attempt == 1 else max(0.0, ocr_delay) * (2 ** (attempt - 2))
                page_limiter.wait_for(estimated_pages)
                ocr_text, ocr_method, error = _extract_ocr_pdf_text_mistral(
                    pdf_path=pdf_path,
                    model=mistral_model or DEFAULT_MISTRAL_OCR_MODEL,
                    delay=retry_delay,
                    output_dir=artifacts_dir,
                    timeout_seconds=max(30, mistral_ocr_timeout_seconds),
                )

                if error and _is_mistral_password_protected_error(error):
                    last_error = f"password_protected:{error}"
                    break

                if error:
                    last_error = error
                    if attempt < max(1, ocr_max_retries):
                        logger.warning(
                            "[{}] Mistral OCR retry {}/{} for {}: {}",
                            source.upper(),
                            attempt,
                            max(1, ocr_max_retries),
                            pdf_path,
                            error,
                        )
                        continue

                if not ocr_text.strip():
                    last_error = error or f"empty_text:{ocr_method}"
                    if attempt < max(1, ocr_max_retries):
                        continue
                    break

                _write_pdf_text_outputs(
                    processed_dir=processed_dir,
                    source=source,
                    output_format=output_format,
                    job=job,
                    pdf_path=pdf_path,
                    text=ocr_text,
                    extraction_method=ocr_method,
                    filename_prefix="pdf_ocr",
                )
                ocr_stats["succeeded"] += 1
                success = True
                break

            if not success:
                pdf_failures += 1
                ocr_stats["failed"] += 1
                _push_error_sample(
                    error_samples,
                    stage="ocr_extract",
                    path=str(pdf_path),
                    reason=last_error or "mistral_ocr_failed",
                )

            if index % max(1, log_every) == 0 or index == len(ocr_jobs):
                logger.info("[{}] pdf ocr progress: {}/{}", source.upper(), index, len(ocr_jobs))

        logger.info("[{}] Mistral OCR phase done, failures={}", source.upper(), pdf_failures)
        return {
            "failures": pdf_failures,
            "ocr_stats": ocr_stats,
            "error_samples": error_samples,
        }

    prerequisites_ok = HAS_PYMUPDF and HAS_PIL and HAS_NUMPY and HAS_PYTESSERACT and _configure_tesseract_from_siwar()
    if not prerequisites_ok:
        pdf_failures = len(ocr_jobs)
        ocr_stats["attempted"] = len(ocr_jobs)
        ocr_stats["failed"] = len(ocr_jobs)
        _push_error_sample(
            error_samples,
            stage="ocr_setup",
            path=source,
            reason="tesseract_or_dependencies_unavailable",
        )
        logger.warning(
            "[{}] OCR unavailable. Marked {} OCR jobs as failures.",
            source.upper(),
            len(ocr_jobs),
        )
        return {
            "failures": pdf_failures,
            "ocr_stats": ocr_stats,
            "error_samples": error_samples,
        }

    logger.info("[{}] OCR memory queue size: {}", source.upper(), len(ocr_jobs))

    for index, job in enumerate(ocr_jobs, start=1):
        ocr_stats["attempted"] += 1
        pdf_path = Path(str(job.get("pdf_path", "")))

        try:
            ocr_text, ocr_method = _extract_ocr_pdf_text_siwar(
                pdf_path=pdf_path,
                ocr_languages=ocr_languages,
                ocr_config=ocr_config,
                dpi=max(72, dpi),
                max_pages=max(0, max_ocr_pages),
            )

            if not ocr_text.strip():
                pdf_failures += 1
                ocr_stats["failed"] += 1
                _push_error_sample(
                    error_samples,
                    stage="ocr_extract",
                    path=str(pdf_path),
                    reason=f"empty_text:{ocr_method}",
                )
                continue

            ocr_stats["succeeded"] += 1

            languages, reason = classify_languages(
                text=ocr_text,
                title=str(job.get("title", "")),
                url=str(job.get("source_url", "")),
                html_language_hint=None,
                extra_hints=[
                    str(job.get("source_page_language", "")),
                    str(job.get("crawler_pdf_url_language_hint", "")),
                ],
            )

            for language in languages:
                output_path = _build_output_path(
                    processed_dir=processed_dir,
                    kind="pdfs",
                    source=source,
                    language=language,
                    doc_id=str(job.get("doc_id", "pdf_doc")),
                    output_format=output_format,
                    filename_prefix="pdf_ocr",
                )
                rendered = _render_text_document(
                    title=str(job.get("title", "")),
                    content=ocr_text,
                    metadata={
                        "doc_id": str(job.get("doc_id", "pdf_doc")),
                        "source": source,
                        "kind": "pdf",
                        "language": language,
                        "language_reason": reason,
                        "url": str(job.get("source_url", "")),
                        "source_path": str(pdf_path),
                        "extraction_method": ocr_method,
                    },
                    output_format=output_format,
                )
                _write_text(output_path, rendered)

            if index % max(1, log_every) == 0 or index == len(ocr_jobs):
                logger.info("[{}] pdf ocr progress: {}/{}", source.upper(), index, len(ocr_jobs))

        except Exception as exc:
            pdf_failures += 1
            ocr_stats["failed"] += 1
            logger.warning("[{}] OCR failed for {}: {}", source.upper(), pdf_path, exc)
            _push_error_sample(
                error_samples,
                stage="ocr_runtime",
                path=str(pdf_path),
                reason=str(exc),
            )

    logger.info("[{}] OCR phase done, failures={}", source.upper(), pdf_failures)
    return {
        "failures": pdf_failures,
        "ocr_stats": ocr_stats,
        "error_samples": error_samples,
    }


def run_preparation(
    *,
    raw_dir: Path,
    processed_dir: Path,
    sources: list[str],
    output_format: str,
    min_native_chars: int,
    ocr_languages: str,
    ocr_config: str,
    dpi: int,
    max_ocr_pages: int,
    log_every: int,
    ocr_engine: str = "mistral",
    mistral_model: str = DEFAULT_MISTRAL_OCR_MODEL,
    ocr_delay: float = 2.0,
    ocr_max_retries: int = 2,
    mistral_ocr_pages_per_minute: int = DEFAULT_MISTRAL_OCR_PAGES_PER_MINUTE,
    mistral_ocr_timeout_seconds: int = DEFAULT_MISTRAL_OCR_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    source_stats: dict[str, dict[str, Any]] = {}

    for source in sources:
        logger.info("------------------------------")
        logger.info("Preparation start for {}", source.upper())

        page_result = prepare_source_pages(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            source=source,
            output_format=output_format,
            log_every=max(1, log_every),
        )

        native_pdf_result = prepare_source_scanned_pdfs(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            source=source,
            min_native_chars=max(0, min_native_chars),
            output_format=output_format,
            log_every=max(1, log_every),
        )

        ocr_pdf_result = prepare_source_ocr_pdfs(
            processed_dir=processed_dir,
            source=source,
            output_format=output_format,
            ocr_jobs=list(native_pdf_result.get("ocr_jobs", [])),
            ocr_engine=ocr_engine,
            mistral_model=mistral_model,
            ocr_delay=max(0.0, ocr_delay),
            ocr_max_retries=max(1, ocr_max_retries),
            mistral_ocr_pages_per_minute=max(1, mistral_ocr_pages_per_minute),
            mistral_ocr_timeout_seconds=max(30, mistral_ocr_timeout_seconds),
            ocr_artifacts_dir=processed_dir / "_ocr_artifacts" / source,
            ocr_languages=ocr_languages,
            ocr_config=ocr_config,
            dpi=max(72, dpi),
            max_ocr_pages=max(0, max_ocr_pages),
            log_every=max(1, log_every),
        )

        page_failures = int(page_result.get("failures", 0))
        pdf_failures = int(native_pdf_result.get("failures", 0)) + int(ocr_pdf_result.get("failures", 0))

        ocr_stats = ocr_pdf_result.get("ocr_stats", {})
        error_samples = list(page_result.get("error_samples", []))
        error_samples.extend(list(native_pdf_result.get("error_samples", [])))
        error_samples.extend(list(ocr_pdf_result.get("error_samples", [])))

        source_stats[source] = {
            "page_failures": page_failures,
            "pdf_failures": pdf_failures,
            "ocr_stats": {
                "engine": str(ocr_stats.get("engine", ocr_engine)),
                "queued": int(ocr_stats.get("queued", 0)),
                "attempted": int(ocr_stats.get("attempted", 0)),
                "succeeded": int(ocr_stats.get("succeeded", 0)),
                "failed": int(ocr_stats.get("failed", 0)),
            },
            "errors": {
                "count": len(error_samples),
                "samples": error_samples,
            },
        }

        logger.info(
            "Preparation done for {}: page_failures={}, pdf_failures={}",
            source.upper(),
            page_failures,
            pdf_failures,
        )
        logger.info(
            "{} OCR: queued={} attempted={} succeeded={} failed={}",
            source.upper(),
            source_stats[source]["ocr_stats"]["queued"],
            source_stats[source]["ocr_stats"]["attempted"],
            source_stats[source]["ocr_stats"]["succeeded"],
            source_stats[source]["ocr_stats"]["failed"],
        )
        logger.info("{} errors tracked: {}", source.upper(), source_stats[source]["errors"]["count"])

    totals = {
        "page_failures": sum(v["page_failures"] for v in source_stats.values()),
        "pdf_failures": sum(v["pdf_failures"] for v in source_stats.values()),
        "ocr": {
            "queued": sum(v["ocr_stats"]["queued"] for v in source_stats.values()),
            "attempted": sum(v["ocr_stats"]["attempted"] for v in source_stats.values()),
            "succeeded": sum(v["ocr_stats"]["succeeded"] for v in source_stats.values()),
            "failed": sum(v["ocr_stats"]["failed"] for v in source_stats.values()),
        },
        "errors": {
            "count": sum(v["errors"]["count"] for v in source_stats.values()),
        },
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": source_stats,
        "totals": totals,
    }


def write_minimal_report(processed_dir: Path, report: dict[str, Any]) -> Path:
    report_path = processed_dir / "preparation_report.json"
    _write_json(report_path, report)
    return report_path
