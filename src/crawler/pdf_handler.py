"""
src/crawler/pdf_handler.py
===============================
Gestion complète des PDFs :
  1. Détection des liens PDF dans une page HTML
  2. Téléchargement asynchrone
  3. Extraction du texte (pdfplumber → pymupdf en fallback)
  4. Détection des PDFs scannés (peu de texte extractible)
  5. Sauvegarde JSON des métadonnées
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiofiles
import httpx
from bs4 import BeautifulSoup
from loguru import logger

from config.logger import log_failed_url


ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
AR_PATH_RE = re.compile(r"(^|/)ar(/|$)", re.I)
FR_PATH_RE = re.compile(r"(^|/)fr(/|$)", re.I)

# Import conditionnel pour les deux librairies d'extraction PDF
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# ─────────────────────────────────────────────
# DÉTECTION DES LIENS PDF DANS UNE PAGE
# ─────────────────────────────────────────────

def find_pdf_links(html_content: str, base_url: str) -> list[str]:
    """
    Parcourt le HTML d'une page et retourne tous les liens pointant vers un PDF.

    Args:
        html_content: Contenu HTML brut de la page.
        base_url:     URL de base pour résoudre les liens relatifs.

    Returns:
        Liste d'URLs absolues vers des fichiers PDF (dédoublonnée).
    """
    soup = BeautifulSoup(html_content, "html.parser")
    pdf_urls: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"].strip()

        # Ignorer les ancres, javascript, mailto
        if href.startswith(("#", "javascript:", "mailto:")):
            continue

        # Résoudre l'URL relative en absolue
        absolute_url = urljoin(base_url, href)

        # Vérifier si l'URL pointe vers un PDF (extension ou paramètre)
        parsed = urlparse(absolute_url)
        path_lower = parsed.path.lower()

        if path_lower.endswith(".pdf") or "pdf" in parsed.query.lower():
            pdf_urls.add(absolute_url)

    return list(pdf_urls)


# ─────────────────────────────────────────────
# TÉLÉCHARGEMENT D'UN PDF
# ─────────────────────────────────────────────

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
    """
    Télécharge un PDF de manière asynchrone.

    Args:
        client:           Client HTTP httpx partagé.
        pdf_url:          URL du PDF à télécharger.
        dest_dir:         Dossier de destination.
        max_size_bytes:   Taille maximale acceptée (en bytes).
        download_timeout: Timeout en secondes.
        source:           Identifiant de la source (cnra/rcar).
        source_page_url:  URL de la page contenant le lien PDF.
        max_retries:      Nombre maximum de tentatives de téléchargement.
        retry_backoff_base: Backoff initial entre retries (exponentiel).

    Returns:
        Chemin local du fichier téléchargé, ou None si échec.
    """
    # Génère un nom de fichier stable à partir de l'URL (hash MD5)
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:10]
    original_name = Path(urlparse(pdf_url).path).name or "document.pdf"
    # Nettoie le nom : supprime caractères problématiques
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in original_name)
    filename = f"{url_hash}_{safe_name}"
    dest_path = dest_dir / filename

    # Si le fichier existe déjà, on ne re-télécharge pas
    if dest_path.exists():
        logger.debug(f"PDF déjà téléchargé : {filename}")
        return dest_path

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"⬇ Téléchargement PDF (tentative {attempt}/{max_retries}) : {pdf_url}")
            async with client.stream(
                "GET",
                pdf_url,
                timeout=download_timeout,
                follow_redirects=True,
            ) as response:
                response.raise_for_status()

                # Vérifie la taille si l'en-tête Content-Length est présent
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

                # Téléchargement par chunks pour ne pas saturer la mémoire
                total_bytes = 0
                async with aiofiles.open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        total_bytes += len(chunk)
                        if total_bytes > max_size_bytes:
                            logger.warning(f"PDF dépasse la taille max en cours de téléchargement : {pdf_url}")
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

            logger.success(f"✓ PDF sauvegardé : {filename} ({total_bytes / 1024:.1f} KB)")
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
            logger.error(f"Timeout téléchargement PDF : {pdf_url}")
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


# ─────────────────────────────────────────────
# EXTRACTION DU TEXTE D'UN PDF
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path, min_chars: int) -> dict:
    """
    Extrait le texte d'un PDF en essayant pdfplumber en premier,
    puis pymupdf (fitz) en fallback.

    Args:
        pdf_path:  Chemin vers le fichier PDF local.
        min_chars: Seuil minimum de caractères pour considérer le PDF comme "texte natif".

    Returns:
        Dictionnaire avec : text, page_count, extraction_method, is_scanned.
    """
    result = {
        "content_text": "",
        "page_count": 0,
        "extraction_method": "none",
        "is_scanned": False,
    }

    # ── Tentative 1 : pdfplumber (meilleur pour tableaux et layouts complexes)
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                result["page_count"] = len(pdf.pages)
                texts = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    texts.append(page_text)
                full_text = "\n\n".join(texts).strip()

                if len(full_text) >= min_chars:
                    result["content_text"] = full_text
                    result["extraction_method"] = "pdfplumber"
                    result["is_scanned"] = False
                    logger.debug(f"pdfplumber OK : {len(full_text)} chars, {result['page_count']} pages")
                    return result
                else:
                    logger.debug(f"pdfplumber : peu de texte ({len(full_text)} chars) → fallback pymupdf")

        except Exception as e:
            logger.warning(f"pdfplumber a échoué sur {pdf_path.name} : {e}")

    # ── Tentative 2 : pymupdf / fitz (meilleur pour PDFs complexes/protégés)
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(str(pdf_path))
            result["page_count"] = doc.page_count
            texts = []
            for page in doc:
                page_text = page.get_text("text") or ""
                texts.append(page_text)
            doc.close()

            full_text = "\n\n".join(texts).strip()

            if len(full_text) >= min_chars:
                result["content_text"] = full_text
                result["extraction_method"] = "pymupdf"
                result["is_scanned"] = False
                logger.debug(f"pymupdf OK : {len(full_text)} chars")
            else:
                # Très peu de texte → probablement un PDF scanné (image)
                result["content_text"] = full_text
                result["extraction_method"] = "pymupdf"
                result["is_scanned"] = True
                logger.warning(f"PDF probablement scanné (image) : {pdf_path.name} ({len(full_text)} chars)")

            return result

        except Exception as e:
            logger.error(f"pymupdf a échoué sur {pdf_path.name} : {e}")

    # Aucun outil n'a fonctionné
    logger.error(f"Impossible d'extraire le texte de : {pdf_path.name}")
    return result


# ─────────────────────────────────────────────
# SAUVEGARDE JSON D'UN PDF TRAITÉ
# ─────────────────────────────────────────────

def save_pdf_metadata(
    pdf_path: Path,
    pdf_url: str,
    source_page_url: str,
    source: str,
    language: str,
    source_page_language: str,
    language_reason: str,
    extraction: dict,
    output_dir: Path,
) -> None:
    """
    Sauvegarde les métadonnées et le texte extrait d'un PDF en JSON.

    Args:
        pdf_path:        Chemin local du PDF téléchargé.
        pdf_url:         URL d'origine du PDF.
        source_page_url: URL de la page où le lien PDF a été trouvé.
        source:          Identifiant de la source ("cnra" ou "rcar").
        language:        Langue catégorisée ("fr" ou "ar").
        extraction:      Résultat de extract_text_from_pdf().
        output_dir:      Dossier où sauvegarder le JSON.
    """
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:10]
    json_path = output_dir / f"{url_hash}_meta.json"

    metadata = {
        "id": f"{source}_pdf_{url_hash}",
        "source_url": pdf_url,
        "source_page_url": source_page_url,
        "source_page_language": source_page_language,
        "source": source,
        "language": language,
        "language_reason": language_reason,
        "filename": pdf_path.name,
        "local_path": str(pdf_path),
        "title": Path(urlparse(pdf_url).path).stem.replace("_", " ").replace("-", " "),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "page_count": extraction["page_count"],
        "content_text": extraction["content_text"],
        "extraction_method": extraction["extraction_method"],
        "is_scanned": extraction["is_scanned"],
        "char_count": len(extraction["content_text"]),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.debug(f"Métadonnées PDF sauvegardées : {json_path.name}")


# ─────────────────────────────────────────────
# PIPELINE COMPLET : TÉLÉCHARGER + EXTRAIRE + SAUVEGARDER
# ─────────────────────────────────────────────


def _detect_script_language(text: str | None) -> str | None:
    sample = (text or "").strip()
    if len(sample) < 30:
        return None

    arabic_count = len(ARABIC_CHAR_RE.findall(sample))
    latin_count = len(LATIN_CHAR_RE.findall(sample))

    if arabic_count >= 18 and arabic_count > latin_count:
        return "ar"
    if latin_count >= 18 and latin_count > arabic_count:
        return "fr"
    return None


def infer_pdf_language(
    pdf_url: str,
    title: str,
    content_text: str,
    source_page_language: str,
    fallback: str = "fr",
) -> tuple[str, str]:
    """
    Classification PDF avec priorité au contenu réel.

    Ordre:
      1) script dominant dans title + content_text
      2) langue de la page source
      3) URL (/ar/ ou /fr/)
      4) fallback
    """
    by_script = _detect_script_language(f"{title}\n{content_text}")
    if by_script:
        return by_script, "title_content_script"

    if source_page_language in {"fr", "ar"}:
        return source_page_language, "source_page_language"

    path = (urlparse(pdf_url).path or "")
    if AR_PATH_RE.search(path):
        return "ar", "pdf_url_path"
    if FR_PATH_RE.search(path):
        return "fr", "pdf_url_path"

    return ("ar" if fallback == "ar" else "fr"), "fallback"

async def process_pdf(
    client: httpx.AsyncClient,
    pdf_url: str,
    source_page_url: str,
    source: str,
    language: str,
    source_page_language: str,
    pdf_dir: Path,
    pdf_config: dict,
) -> bool:
    """
    Pipeline complet pour un PDF : téléchargement → extraction → sauvegarde JSON.

    Returns:
        True si le PDF a été traité avec succès, False sinon.
    """
    language_safe = language if language in {"fr", "ar"} else (
        source_page_language if source_page_language in {"fr", "ar"} else "fr"
    )
    language_pdf_dir = pdf_dir / language_safe
    language_pdf_dir.mkdir(parents=True, exist_ok=True)

    # 1. Télécharger
    pdf_path = await download_pdf(
        client=client,
        pdf_url=pdf_url,
        dest_dir=language_pdf_dir,
        max_size_bytes=pdf_config["max_pdf_size_bytes"],
        download_timeout=pdf_config["download_timeout"],
        source=source,
        source_page_url=source_page_url,
        max_retries=int(pdf_config.get("download_max_retries", 3)),
        retry_backoff_base=float(pdf_config.get("download_retry_backoff_base", 1.0)),
    )

    if pdf_path is None:
        return False

    # 2. Extraire le texte
    extraction = extract_text_from_pdf(
        pdf_path=pdf_path,
        min_chars=pdf_config["min_text_chars_threshold"],
    )

    if extraction["extraction_method"] == "none":
        log_failed_url(
            url=pdf_url,
            reason="Echec extraction texte (pdfplumber et pymupdf)",
            source=source,
            stage="pdf_extraction",
            context_url=source_page_url,
        )

    title = Path(urlparse(pdf_url).path).stem.replace("_", " ").replace("-", " ")
    detected_language, language_reason = infer_pdf_language(
        pdf_url=pdf_url,
        title=title,
        content_text=extraction.get("content_text", ""),
        source_page_language=source_page_language,
        fallback=language_safe,
    )

    if detected_language != language_safe:
        target_dir = pdf_dir / detected_language
        target_dir.mkdir(parents=True, exist_ok=True)
        target_pdf_path = target_dir / pdf_path.name

        if target_pdf_path != pdf_path:
            if target_pdf_path.exists():
                target_pdf_path.unlink(missing_ok=True)
            pdf_path.replace(target_pdf_path)
            pdf_path = target_pdf_path
        language_safe = detected_language

    # 3. Sauvegarder les métadonnées en JSON
    save_pdf_metadata(
        pdf_path=pdf_path,
        pdf_url=pdf_url,
        source_page_url=source_page_url,
        source=source,
        language=language_safe,
        source_page_language=source_page_language,
        language_reason=language_reason,
        extraction=extraction,
        output_dir=pdf_path.parent,
    )

    return True
