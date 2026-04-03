"""
src/crawler/data_handler.py
===========================
Unified data utilities for pages and PDFs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiofiles
import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from loguru import logger

from config.logger import log_failed_url


ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
AR_PATH_RE = re.compile(r"(^|/)ar(/|$)", re.I)
FR_PATH_RE = re.compile(r"(^|/)fr(/|$)", re.I)


def read_json_file(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_json_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def unique_target_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_moved{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def path_language_hint(url: str) -> str | None:
    path = (urlparse(url).path or "")
    if AR_PATH_RE.search(path):
        return "ar"
    if FR_PATH_RE.search(path):
        return "fr"
    return None


def detect_script_language(text: str | None) -> str | None:
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


def html_to_clean_text(soup: BeautifulSoup, exclude_selectors: list[str]) -> str:
    """
    Remove navigation/footer/scripts and convert remaining content
    into clean text with simple markdown headings.
    """
    soup_copy = BeautifulSoup(str(soup), "html.parser")

    for selector in exclude_selectors:
        for element in soup_copy.select(selector):
            element.decompose()

    main_content = (
        soup_copy.find("main")
        or soup_copy.find("article")
        or soup_copy.find(id=re.compile(r"content|main|article", re.I))
        or soup_copy.find(class_=re.compile(r"content|main|article", re.I))
        or soup_copy.find("body")
        or soup_copy
    )

    lines: list[str] = []
    _extract_text_recursive(main_content, lines)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _extract_text_recursive(element: Tag | NavigableString, lines: list[str]) -> None:
    if isinstance(element, NavigableString):
        text = str(element).strip()
        if text:
            lines.append(text)
        return

    tag_name = element.name if element.name else ""

    heading_map = {
        "h1": "# ",
        "h2": "## ",
        "h3": "### ",
        "h4": "#### ",
        "h5": "##### ",
        "h6": "###### ",
    }
    if tag_name in heading_map:
        text = element.get_text(separator=" ", strip=True)
        if text:
            lines.append(f"\n{heading_map[tag_name]}{text}\n")
        return

    if tag_name == "p":
        text = element.get_text(separator=" ", strip=True)
        if text:
            lines.append(text)
            lines.append("")
        return

    if tag_name in ("ul", "ol"):
        for i, li in enumerate(element.find_all("li", recursive=False), 1):
            text = li.get_text(separator=" ", strip=True)
            prefix = f"{i}. " if tag_name == "ol" else "- "
            if text:
                lines.append(f"{prefix}{text}")
        lines.append("")
        return

    if tag_name == "table":
        for row in element.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if cells:
                lines.append(" | ".join(cells))
        lines.append("")
        return

    for child in element.children:
        _extract_text_recursive(child, lines)


def extract_page_metadata(soup: BeautifulSoup) -> dict:
    meta: dict = {}

    title_tag = soup.find("title")
    meta["title"] = title_tag.get_text(strip=True) if title_tag else ""

    for attr in ["description", "keywords"]:
        tag = soup.find("meta", attrs={"name": attr})
        if tag and tag.get("content"):
            meta[attr] = tag["content"].strip()

    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        meta["language"] = html_tag["lang"].strip()[:5]
    else:
        meta["language"] = "unknown"

    return meta


def infer_simple_language(
    url: str,
    text: str | None,
    title: str | None = None,
    html_language: str | None = None,
    fallback: str = "unknown",
) -> tuple[str, str]:
    combined_text = f"{title or ''}\n{text or ''}"
    by_script = detect_script_language(combined_text)
    if by_script:
        return by_script, "title_content_script"

    lang_hint = (html_language or "").strip().lower()
    if lang_hint.startswith("ar"):
        return "ar", "html_lang"
    if lang_hint.startswith("fr"):
        return "fr", "html_lang"

    path_hint = path_language_hint(url)
    if path_hint:
        return path_hint, "url_path"

    if fallback in {"fr", "ar", "unknown"}:
        return fallback, "fallback"
    return "unknown", "fallback"


def normalize_language_hint(value: str | None) -> str:
    candidate = (value or "").strip().lower()
    if candidate.startswith("ar"):
        return "ar"
    if candidate.startswith("fr"):
        return "fr"
    return "unknown"


def save_page(
    url: str,
    source: str,
    metadata: dict,
    content: str,
    pdf_links: list[str],
    depth: int,
    language_hint: str,
    pages_dir: Path,
) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    json_path = pages_dir / f"page_{url_hash}.json"

    data = {
        "id": f"{source}_page_{url_hash}",
        "url": url,
        "source": source,
        "title": metadata.get("title", ""),
        "language": "unknown",
        "language_reason": "deferred_to_preparation",
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "depth": depth,
        "content_markdown": content,
        "char_count": len(content),
        "pdf_links": pdf_links,
        "metadata": {
            "description": metadata.get("description", ""),
            "keywords": metadata.get("keywords", ""),
            "html_language_hint": metadata.get("html_language_raw", ""),
            "source_page_language_hint": normalize_language_hint(language_hint),
            "language_reason": "deferred_to_preparation",
            "crawler_stage": "raw_collection",
        },
    }

    write_json_file(json_path, data)

    logger.debug(f"Page sauvegardee : {json_path.name} ({len(content)} chars)")


def find_pdf_links(html_content: str | BeautifulSoup, base_url: str) -> list[str]:
    soup = html_content if isinstance(html_content, BeautifulSoup) else BeautifulSoup(html_content, "html.parser")
    pdf_urls: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"].strip()

        if href.startswith(("#", "javascript:", "mailto:")):
            continue

        absolute_url = urljoin(base_url, href)

        parsed = urlparse(absolute_url)
        path_lower = parsed.path.lower()

        if path_lower.endswith(".pdf") or "pdf" in parsed.query.lower():
            pdf_urls.add(absolute_url)

    return list(pdf_urls)


async def download_pdf(
    client: httpx.AsyncClient,
    pdf_url: str,
    dest_dir: Path,
    max_size_bytes: int,
    download_timeout: int,
    source: str,
    source_page_url: str,
    max_retries: int,
    retry_backoff_base: float,
) -> Path | None:
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:10]
    original_name = Path(urlparse(pdf_url).path).name or "document.pdf"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in original_name)
    filename = f"{url_hash}_{safe_name}"
    dest_path = dest_dir / filename

    if dest_path.exists():
        logger.debug(f"PDF deja telecharge : {filename}")
        return dest_path

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Telechargement PDF tentative {attempt}/{max_retries} : {pdf_url}")
            async with client.stream(
                "GET",
                pdf_url,
                timeout=download_timeout,
                follow_redirects=True,
            ) as response:
                response.raise_for_status()

                content_length = int(response.headers.get("content-length", 0))
                if content_length > max_size_bytes:
                    logger.warning(f"PDF trop volumineux ({content_length} bytes) : {pdf_url}")
                    log_failed_url(
                        url=pdf_url,
                        reason=f"PDF trop volumineux ({content_length} bytes)",
                        source=source,
                        stage="pdf_download",
                        context_url=source_page_url,
                    )
                    return None

                total_bytes = 0
                async with aiofiles.open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        total_bytes += len(chunk)
                        if total_bytes > max_size_bytes:
                            logger.warning(
                                f"PDF depasse la taille max en cours de telechargement : {pdf_url}"
                            )
                            log_failed_url(
                                url=pdf_url,
                                reason="PDF depasse la taille maximale pendant le telechargement",
                                source=source,
                                stage="pdf_download",
                                context_url=source_page_url,
                            )
                            dest_path.unlink(missing_ok=True)
                            return None
                        await f.write(chunk)

            logger.success(f"PDF sauvegarde : {filename} ({total_bytes / 1024:.1f} KB)")
            return dest_path

        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP {e.response.status_code} pour PDF : {pdf_url}")
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                log_failed_url(
                    url=pdf_url,
                    reason=f"HTTP {e.response.status_code}",
                    source=source,
                    stage="pdf_download",
                    context_url=source_page_url,
                )
                dest_path.unlink(missing_ok=True)
                return None
        except httpx.TimeoutException:
            logger.error(f"Timeout telechargement PDF : {pdf_url}")
        except Exception as e:
            logger.error(f"Erreur inattendue PDF ({pdf_url}) : {e}")

        dest_path.unlink(missing_ok=True)
        if attempt < max_retries:
            await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))

    log_failed_url(
        url=pdf_url,
        reason=f"Echec telechargement apres {max_retries} tentatives",
        source=source,
        stage="pdf_download",
        context_url=source_page_url,
    )
    return None


def save_pdf_metadata(
    pdf_path: Path,
    pdf_url: str,
    source_page_url: str,
    source: str,
    source_page_language_hint: str,
    pdf_url_language_hint: str,
    output_dir: Path,
) -> None:
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:10]
    json_path = output_dir / f"{url_hash}_meta.json"

    metadata = {
        "id": f"{source}_pdf_{url_hash}",
        "source_url": pdf_url,
        "source_page_url": source_page_url,
        "source_page_language": normalize_language_hint(source_page_language_hint),
        "crawler_pdf_url_language_hint": normalize_language_hint(pdf_url_language_hint),
        "source": source,
        "language": "unknown",
        "language_reason": "deferred_to_preparation",
        "filename": pdf_path.name,
        "local_path": str(pdf_path),
        "title": Path(urlparse(pdf_url).path).stem.replace("_", " ").replace("-", " "),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "crawler_stage": "raw_collection",
    }

    write_json_file(json_path, metadata)

    logger.debug(f"Metadonnees PDF sauvegardees : {json_path.name}")


async def process_pdf(
    client: httpx.AsyncClient,
    pdf_url: str,
    source_page_url: str,
    source: str,
    source_page_language_hint: str,
    pdf_url_language_hint: str,
    pdf_dir: Path,
    pdf_config: dict,
) -> bool:
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = await download_pdf(
        client=client,
        pdf_url=pdf_url,
        dest_dir=pdf_dir,
        max_size_bytes=pdf_config["max_pdf_size_bytes"],
        download_timeout=pdf_config["download_timeout"],
        source=source,
        source_page_url=source_page_url,
        max_retries=int(pdf_config.get("download_max_retries", 3)),
        retry_backoff_base=float(pdf_config.get("download_retry_backoff_base", 1.0)),
    )

    if pdf_path is None:
        return False

    save_pdf_metadata(
        pdf_path=pdf_path,
        pdf_url=pdf_url,
        source_page_url=source_page_url,
        source=source,
        source_page_language_hint=source_page_language_hint,
        pdf_url_language_hint=pdf_url_language_hint,
        output_dir=pdf_path.parent,
    )

    return True