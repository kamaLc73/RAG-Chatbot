"""
src/crawler/crawler.py
===========================
Crawler principal utilisant Crawlee (BeautifulSoupCrawler).
Pour chaque page :
  - Extrait le contenu textuel propre (Markdown simplifié)
  - Détecte les liens PDF et les traite via pdf_handler
  - Enqueue les liens internes du même domaine
  - Sauvegarde les données en JSON
"""

import asyncio
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx
from crawlee import ConcurrencySettings, Request
from crawlee.http_clients import HttpxHttpClient
from bs4 import BeautifulSoup, NavigableString, Tag
from crawlee.crawlers import BeautifulSoupCrawler, BeautifulSoupCrawlingContext
from loguru import logger

# Ajout du parent dans le path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import HTML_EXCLUDE_SELECTORS, PDF_CONFIG, CRAWLER_CONFIG
from config.logger import log_failed_url
from crawler.pdf_handler import find_pdf_links


TRACKING_OR_SESSION_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "session",
    "sessionid",
    "sid",
    "phpsessid",
    "jsessionid",
    "token",
}

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
AR_PATH_RE = re.compile(r"(^|/)ar(/|$)", re.I)
FR_PATH_RE = re.compile(r"(^|/)fr(/|$)", re.I)


def _path_language_hint(url: str) -> str | None:
    path = (urlparse(url).path or "")
    if AR_PATH_RE.search(path):
        return "ar"
    if FR_PATH_RE.search(path):
        return "fr"
    return None


def _read_json_file(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _unique_target_path(path: Path) -> Path:
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


def _detect_script_language(text: str | None) -> str | None:
    """Détecte une langue dominante à partir du script utilisé dans le texte."""
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

# ─────────────────────────────────────────────
# CONVERSION HTML → TEXTE MARKDOWN SIMPLIFIÉ
# ─────────────────────────────────────────────

def html_to_clean_text(soup: BeautifulSoup, exclude_selectors: list[str]) -> str:
    """
    Supprime les éléments de navigation/footer/scripts puis
    convertit le contenu restant en texte propre (avec titres Markdown).

    Args:
        soup:              Objet BeautifulSoup de la page complète.
        exclude_selectors: Sélecteurs CSS des éléments à supprimer.

    Returns:
        Texte nettoyé (format Markdown simplifié).
    """
    # Copie pour ne pas modifier l'objet original
    soup_copy = BeautifulSoup(str(soup), "html.parser")

    # Supprimer les éléments indésirables (nav, footer, scripts…)
    for selector in exclude_selectors:
        for element in soup_copy.select(selector):
            element.decompose()

    # Chercher la zone de contenu principal
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

    # Nettoyage final : supprimer les lignes vides consécutives
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _extract_text_recursive(element: Tag | NavigableString, lines: list[str]) -> None:
    """
    Parcourt récursivement le DOM et construit les lignes de texte.
    Convertit les balises h1-h6 en titres Markdown (#, ##, etc.).
    """
    if isinstance(element, NavigableString):
        text = str(element).strip()
        if text:
            lines.append(text)
        return

    tag_name = element.name if element.name else ""

    # Titres → Markdown
    heading_map = {"h1": "# ", "h2": "## ", "h3": "### ",
                   "h4": "#### ", "h5": "##### ", "h6": "###### "}
    if tag_name in heading_map:
        text = element.get_text(separator=" ", strip=True)
        if text:
            lines.append(f"\n{heading_map[tag_name]}{text}\n")
        return

    # Paragraphes → ligne séparée
    if tag_name == "p":
        text = element.get_text(separator=" ", strip=True)
        if text:
            lines.append(text)
            lines.append("")  # Ligne vide après chaque paragraphe
        return

    # Listes
    if tag_name in ("ul", "ol"):
        for i, li in enumerate(element.find_all("li", recursive=False), 1):
            text = li.get_text(separator=" ", strip=True)
            prefix = f"{i}. " if tag_name == "ol" else "- "
            if text:
                lines.append(f"{prefix}{text}")
        lines.append("")
        return

    # Tableaux → texte tabulé simple
    if tag_name == "table":
        for row in element.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if cells:
                lines.append(" | ".join(cells))
        lines.append("")
        return

    # Récursion pour les autres éléments
    for child in element.children:
        _extract_text_recursive(child, lines)


# ─────────────────────────────────────────────
# EXTRACTION DES MÉTADONNÉES DE LA PAGE
# ─────────────────────────────────────────────

def extract_page_metadata(soup: BeautifulSoup) -> dict:
    """Extrait le titre, la description et les mots-clés depuis les balises meta."""
    meta: dict = {}

    # Titre
    title_tag = soup.find("title")
    meta["title"] = title_tag.get_text(strip=True) if title_tag else ""

    # Description et mots-clés
    for attr in ["description", "keywords"]:
        tag = soup.find("meta", attrs={"name": attr})
        if tag and tag.get("content"):
            meta[attr] = tag["content"].strip()

    # Langue
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        meta["language"] = html_tag["lang"].strip()[:5]  # ex: "fr", "ar", "fr-MA"
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
    """
    Classification simple FR/AR:
      1) script dominant dans le titre + contenu (priorité)
      2) balise html lang
      3) URL /ar/ ou /fr/
      4) fallback
    """
    combined_text = f"{title or ''}\n{text or ''}"
    by_script = _detect_script_language(combined_text)
    if by_script:
        return by_script, "title_content_script"

    lang_hint = (html_language or "").strip().lower()
    if lang_hint.startswith("ar"):
        return "ar", "html_lang"
    if lang_hint.startswith("fr"):
        return "fr", "html_lang"

    path_hint = _path_language_hint(url)
    if path_hint:
        return path_hint, "url_path"

    if fallback in {"fr", "ar", "unknown"}:
        return fallback, "fallback"
    return "unknown", "fallback"


def infer_pdf_language_hint(pdf_url: str, page_language: str) -> tuple[str, str]:
    """Langue PDF initiale: priorité à la langue de la page, URL en dernier recours."""
    if page_language in {"fr", "ar"}:
        return page_language, "source_page_language"

    path_hint = _path_language_hint(pdf_url)
    if path_hint:
        return path_hint, "pdf_url_path"

    return "unknown", "fallback"


# ─────────────────────────────────────────────
# SAUVEGARDE D'UNE PAGE EN JSON
# ─────────────────────────────────────────────

def save_page(
    url: str,
    source: str,
    metadata: dict,
    content: str,
    pdf_links: list[str],
    depth: int,
    language: str,
    pages_dir: Path,
) -> None:
    """Sauvegarde les données d'une page crawlée dans un fichier JSON."""
    language_dir = pages_dir / language
    language_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    json_path = language_dir / f"page_{url_hash}.json"

    data = {
        "id": f"{source}_page_{url_hash}",
        "url": url,
        "source": source,
        "title": metadata.get("title", ""),
        "language": language,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "depth": depth,
        "content_markdown": content,
        "char_count": len(content),
        "pdf_links": pdf_links,
        "metadata": {
            "description": metadata.get("description", ""),
            "keywords": metadata.get("keywords", ""),
            "html_language_hint": metadata.get("html_language_raw", ""),
            "language_reason": metadata.get("language_reason", ""),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.debug(f"Page sauvegardée : {json_path.name} ({len(content)} chars)")


# ─────────────────────────────────────────────
# CLASSE PRINCIPALE DU CRAWLER
# ─────────────────────────────────────────────

class crawler:
    """
    Crawler principal pour les sites RCAR et CNRA.
    Gère le crawling des pages ET le téléchargement des PDFs.
    """

    def __init__(
        self,
        source: str,
        base_url: str,
        raw_dir: Path,
        crawler_config: dict,
        pdf_config: dict,
    ) -> None:
        """
        Args:
            source:         "cnra" ou "rcar"
            base_url:       URL racine du site (ex: "https://www.cnra.ma")
            raw_dir:        Répertoire data/raw/
            crawler_config: Paramètres du crawler (depuis settings.py)
            pdf_config:     Paramètres des PDFs (depuis settings.py)
        """
        self.source = source
        self.base_url = base_url.rstrip("/")
        self.domain = urlparse(base_url).netloc  # ex: "www.cnra.ma"
        self.seed_urls = [self.base_url, f"{self.base_url}/ar/"]

        # Dossiers de sortie
        self.pages_dir = raw_dir / source / "pages"
        self.pdfs_dir  = raw_dir / source / "pdfs"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)

        self.crawler_config = crawler_config
        self.pdf_config = pdf_config
        self.production_mode = bool(self.crawler_config.get("production_mode", False))
        self._production_skip_patterns = [
            re.compile(p, re.I)
            for p in self.crawler_config.get("production_skip_url_patterns", [])
        ]

        # État interne
        self._visited_urls: set[str] = set()
        self._enqueued_urls: set[str] = {self._normalize_url(url) for url in self.seed_urls}
        self._pdf_queue: list[tuple[str, str, str, str]] = []
        # (pdf_url, source_page_url, pdf_language_hint, source_page_language)
        self._stats = {
            "pages_crawled": 0,
            "pages_failed": 0,
            "pdfs_found": 0,
            "pdfs_downloaded": 0,
            "pdfs_failed": 0,
        }

    def _is_internal_url(self, url: str) -> bool:
        """Vérifie que l'URL appartient bien au même domaine que la source."""
        try:
            parsed = urlparse(url)
            # Accepter le même domaine (avec ou sans www)
            return parsed.netloc == self.domain or parsed.netloc == ""
        except Exception:
            return False

    def _normalize_url(self, url: str) -> str:
        """
        Normalise une URL : supprime les fragments (#ancre) et les
        paramètres de session qui créent des doublons inutiles.
        """
        parsed = urlparse(url)

        # Supprimer paramètres de tracking/session tout en conservant les paramètres métier.
        kept_params = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            if key.lower() in TRACKING_OR_SESSION_PARAMS:
                continue
            kept_params.append((key, value))

        # Tri stable pour obtenir des URLs canoniques.
        normalized_query = urlencode(sorted(kept_params), doseq=True)

        # Reconstruire sans fragment.
        normalized = parsed._replace(fragment="", query=normalized_query).geturl()
        return normalized.rstrip("/")

    def _is_same_source_domain(self, url: str) -> bool:
        """Vérifie qu'une URL appartient au domaine de la source en cours."""
        parsed = urlparse(url)
        if not parsed.netloc:
            return True
        return parsed.netloc == self.domain

    async def run(self) -> dict:
        """
        Lance le crawl complet : pages + PDFs.

        Returns:
            Dictionnaire de statistiques finales.
        """
        logger.info(f"{'='*60}")
        logger.info(f"Démarrage du crawl : {self.source.upper()} ({self.base_url})")
        logger.info(f"{'='*60}")

        # ── Phase A : Crawler les pages ──
        await self._crawl_pages()

        # ── Phase B : Télécharger les PDFs collectés ──
        if self.pdf_config.get("max_pdf_size_bytes", 1) <= 0:
            logger.info("Téléchargement PDF désactivé par configuration.")
        elif self._pdf_queue:
            logger.info(f"\n{'─'*60}")
            logger.info(f"📄 {len(self._pdf_queue)} PDFs détectés → début téléchargement")
            logger.info(f"{'─'*60}")
            await self._download_pdfs()

        # Normalisation finale pour corriger tout mauvais classement FR/AR résiduel.
        self._postprocess_language_outputs()

        # Rapport final
        self._log_stats()
        return self._stats

    async def _crawl_pages(self) -> None:
        """Configure et lance le BeautifulSoupCrawler de Crawlee."""

        # Référence vers self pour l'utiliser dans le handler (closure)
        crawler_self = self
        default_headers = {
            "User-Agent": self.crawler_config["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,ar;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
        }

        # Configuration Crawlee
        max_concurrency = max(1, int(self.crawler_config.get("max_concurrency", 3)))
        crawler = BeautifulSoupCrawler(
            parser="html.parser",
            http_client=HttpxHttpClient(headers=default_headers, follow_redirects=True, http2=False),
            max_requests_per_crawl=(
                self.crawler_config["max_pages_per_source"] or None
            ),
            max_crawl_depth=self.crawler_config["max_depth"] or None,
            max_request_retries=self.crawler_config.get("max_retries", 3),
            respect_robots_txt_file=self.crawler_config.get("respect_robots_txt", True),
            navigation_timeout=timedelta(seconds=self.crawler_config.get("request_timeout", 30)),
            concurrency_settings=ConcurrencySettings(
                min_concurrency=1,
                desired_concurrency=max_concurrency,
                max_concurrency=max_concurrency,
            ),
        )

        @crawler.router.default_handler
        async def handle_page(context: BeautifulSoupCrawlingContext) -> None:
            """Handler appelé pour chaque page visitée."""
            url = context.request.url
            soup = context.soup

            # Delai de politesse variable entre requetes.
            await asyncio.sleep(
                random.uniform(
                    float(crawler_self.crawler_config.get("request_delay_min", 1.0)),
                    float(crawler_self.crawler_config.get("request_delay_max", 2.5)),
                )
            )

            # Normaliser et vérifier si déjà visitée
            norm_url = crawler_self._normalize_url(url)
            if norm_url in crawler_self._visited_urls:
                logger.debug(f"Déjà visitée, ignorée : {norm_url}")
                return

            crawler_self._visited_urls.add(norm_url)
            crawler_self._stats["pages_crawled"] += 1

            logger.info(
                f"[{crawler_self._stats['pages_crawled']}] "
                f"Crawl : {url[:80]}{'...' if len(url) > 80 else ''}"
            )

            try:
                # 1. Extraire les métadonnées
                metadata = extract_page_metadata(soup)
                depth = context.request.user_data.get("depth", 0)

                # En mode production FR/AR, ignorer explicitement les pages EN.
                html_lang_raw = str(metadata.get("language", "")).lower()
                if crawler_self.production_mode and html_lang_raw.startswith("en"):
                    logger.debug(f"Page EN ignorée en mode production : {url}")
                    return

                # 2. Nettoyer et extraire le texte
                content = html_to_clean_text(soup, HTML_EXCLUDE_SELECTORS)

                page_language, page_language_reason = infer_simple_language(
                    url=url,
                    text=content,
                    title=metadata.get("title", ""),
                    html_language=html_lang_raw,
                    fallback="fr",
                )

                metadata["html_language_raw"] = html_lang_raw
                metadata["language_reason"] = page_language_reason
                metadata["language"] = page_language

                # 3. Détecter les liens PDF
                html_content = str(soup)
                pdf_links = find_pdf_links(html_content, url)
                filtered_pdf_links: list[str] = []

                if pdf_links:
                    # Garder uniquement les PDFs du même domaine source.
                    filtered_pdf_links = [
                        link for link in pdf_links if crawler_self._is_same_source_domain(link)
                    ]

                    logger.info(f"  └─ 📎 {len(filtered_pdf_links)} PDF(s) trouvé(s)")
                    crawler_self._stats["pdfs_found"] += len(filtered_pdf_links)

                    max_pdfs = int(crawler_self.crawler_config.get("max_pdfs_per_source", 0) or 0)
                    max_pdfs_reached = False
                    for pdf_url in filtered_pdf_links:
                        pdf_language, _pdf_reason = infer_pdf_language_hint(pdf_url, page_language)

                        # Éviter les doublons dans la file
                        if max_pdfs and len(crawler_self._pdf_queue) >= max_pdfs:
                            max_pdfs_reached = True
                            break
                        if (pdf_url, pdf_language, page_language) not in [
                            (u, lang, src_lang) for u, _, lang, src_lang in crawler_self._pdf_queue
                        ]:
                            crawler_self._pdf_queue.append((pdf_url, url, pdf_language, page_language))

                    if max_pdfs_reached:
                        logger.warning(
                            f"  └─ Limite max PDFs atteinte ({max_pdfs}) pour {crawler_self.source.upper()}"
                        )

                # 4. Sauvegarder la page en JSON
                save_page(
                    url=url,
                    source=crawler_self.source,
                    metadata=metadata,
                    content=content,
                    pdf_links=filtered_pdf_links,
                    depth=depth,
                    language=page_language,
                    pages_dir=crawler_self.pages_dir,
                )

                # Limiter l'exploration si une profondeur max explicite est configurée.
                max_depth = crawler_self.crawler_config.get("max_depth", 0)
                if max_depth and depth >= max_depth:
                    return

                def transform_request(req: dict) -> dict:
                    """Normalise l'URL et propage la profondeur pour les pages enfilees."""
                    raw_url = req.get("url", "")
                    if not raw_url:
                        return "skip"

                    parsed = urlparse(raw_url)
                    if parsed.scheme not in ("http", "https"):
                        return "skip"

                    # Eviter les URL artificielles de type /ar/ar/ar/... qui gonflent inutilement le crawl.
                    if re.search(r"/(?:ar|fr)(?:/(?:ar|fr)){2,}(?:/|$)", parsed.path, re.I):
                        return "skip"

                    normalized_url = crawler_self._normalize_url(raw_url)

                    if crawler_self.production_mode and any(
                        pattern.search(normalized_url)
                        for pattern in crawler_self._production_skip_patterns
                    ):
                        return "skip"

                    # Eviter de re-enqueue la meme page (anti-boucle).
                    if normalized_url in crawler_self._enqueued_urls:
                        return "skip"

                    crawler_self._enqueued_urls.add(normalized_url)
                    req["url"] = normalized_url
                    req["unique_key"] = normalized_url

                    existing_headers = req.get("headers") or {}
                    req["headers"] = {**existing_headers, **default_headers}

                    user_data = req.get("user_data") or {}
                    user_data["depth"] = depth + 1
                    req["user_data"] = user_data
                    return req

                # 5. Enqueue les liens internes (même domaine, pas PDF)
                await context.enqueue_links(
                    strategy="same-domain",
                    transform_request_function=transform_request,
                    exclude=[
                        re.compile(r".*\.pdf(?:$|[?#])", re.I),
                        re.compile(r".*#.*"),
                        re.compile(r".*\.(?:xlsx|xls|doc|docx|ppt|pptx|odt|ods|odp|rtf|csv)(?:$|[?#])", re.I),
                        re.compile(
                            r".*\.(?:jpg|jpeg|png|gif|svg|webp|ico|css|js|zip|rar|7z|mp4|mp3|woff|woff2|ttf|eot)(?:$|[?#])",
                            re.I,
                        ),
                    ],
                )

            except Exception as e:
                crawler_self._stats["pages_failed"] += 1
                logger.error(f"Erreur traitement page {url} : {e}")
                log_failed_url(
                    url=url,
                    reason=str(e),
                    source=crawler_self.source,
                    stage="page_crawl",
                )

        # Lancer le crawl avec la racine + seed arabe pour couvrir les deux langues.
        seed_requests = [
            Request.from_url(
                url,
                headers=default_headers,
                user_data={"depth": 0},
            )
            for url in self.seed_urls
        ]
        await crawler.run(seed_requests)

    async def _download_pdfs(self) -> None:
        """
        Télécharge tous les PDFs de la file en parallèle,
        avec un semaphore pour limiter la concurrence.
        """
        semaphore = asyncio.Semaphore(self.pdf_config["download_concurrency"])
        headers = {"User-Agent": self.crawler_config["user_agent"]}

        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            tasks = [
                self._download_one_pdf(
                    client,
                    pdf_url,
                    page_url,
                    pdf_language,
                    source_page_language,
                    semaphore,
                )
                for pdf_url, page_url, pdf_language, source_page_language in self._pdf_queue
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compter les succès/échecs
        for (pdf_url, page_url, _pdf_language, _source_page_language), result in zip(self._pdf_queue, results):
            if isinstance(result, Exception):
                self._stats["pdfs_failed"] += 1
                log_failed_url(
                    url=pdf_url,
                    reason=f"Exception pipeline PDF: {result}",
                    source=self.source,
                    stage="pdf_pipeline",
                    context_url=page_url,
                )
            elif result:
                self._stats["pdfs_downloaded"] += 1
            else:
                self._stats["pdfs_failed"] += 1
                log_failed_url(
                    url=pdf_url,
                    reason="Traitement PDF echoue",
                    source=self.source,
                    stage="pdf_pipeline",
                    context_url=page_url,
                )

    async def _download_one_pdf(
        self,
        client: httpx.AsyncClient,
        pdf_url: str,
        page_url: str,
        pdf_language: str,
        source_page_language: str,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """Télécharge + extrait + sauvegarde un seul PDF (avec semaphore)."""
        async with semaphore:
            # Délai de politesse entre téléchargements
            await asyncio.sleep(self.crawler_config["request_delay_min"])

            from crawler.pdf_handler import process_pdf
            try:
                timeout_seconds = int(self.pdf_config.get("per_pdf_task_timeout_seconds", 240))
                return await asyncio.wait_for(
                    process_pdf(
                        client=client,
                        pdf_url=pdf_url,
                        source_page_url=page_url,
                        source=self.source,
                        language=pdf_language,
                        source_page_language=source_page_language,
                        pdf_dir=self.pdfs_dir,
                        pdf_config=self.pdf_config,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                log_failed_url(
                    url=pdf_url,
                    reason=f"Timeout pipeline PDF (> {self.pdf_config.get('per_pdf_task_timeout_seconds', 240)}s)",
                    source=self.source,
                    stage="pdf_pipeline_timeout",
                    context_url=page_url,
                )
                logger.error(f"Timeout pipeline PDF: {pdf_url}")
                return False

    def _postprocess_language_outputs(self) -> None:
        """
        Reclassement final simple et robuste:
          - Pages: basé sur title + content_markdown
          - PDFs: basé sur title + content_text depuis le JSON metadata

        Objectif: corriger les cas restants où URL/hints ont mal classé FR/AR.
        """
        pages_checked = 0
        pages_moved = 0
        pdf_meta_checked = 0
        pdf_meta_moved = 0
        pdf_files_moved = 0

        # Pages
        for lang_dir in ("fr", "ar", "unknown"):
            current_dir = self.pages_dir / lang_dir
            if not current_dir.exists():
                continue

            for page_json in current_dir.glob("page_*.json"):
                pages_checked += 1
                data = _read_json_file(page_json)
                if not data:
                    continue

                metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
                text_blob = "\n".join(
                    [
                        str(data.get("content_markdown", "")),
                        str(metadata.get("description", "")),
                        str(metadata.get("keywords", "")),
                    ]
                )

                detected, reason = infer_simple_language(
                    url=str(data.get("url", "")),
                    text=text_blob,
                    title=str(data.get("title", "")),
                    html_language=str(metadata.get("html_language_hint", "")) or None,
                    fallback="unknown",
                )

                if detected == lang_dir:
                    continue

                if detected not in {"fr", "ar", "unknown"}:
                    detected = "unknown"

                target_dir = self.pages_dir / detected
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = _unique_target_path(target_dir / page_json.name)

                data["language"] = detected
                meta_out = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
                meta_out["language_reason"] = f"postprocess:{reason}"
                data["metadata"] = meta_out

                _write_json_file(target_path, data)
                page_json.unlink(missing_ok=True)
                pages_moved += 1

        # PDFs + metadata
        for lang_dir in ("fr", "ar", "unknown"):
            current_dir = self.pdfs_dir / lang_dir
            if not current_dir.exists():
                continue

            for meta_json in current_dir.glob("*_meta.json"):
                pdf_meta_checked += 1
                data = _read_json_file(meta_json)
                if not data:
                    continue

                source_page_lang_raw = str(data.get("source_page_language", "")).lower().strip()
                source_page_lang = "unknown"
                if source_page_lang_raw.startswith("ar"):
                    source_page_lang = "ar"
                elif source_page_lang_raw.startswith("fr"):
                    source_page_lang = "fr"

                detected, reason = infer_simple_language(
                    url=str(data.get("source_url", "")),
                    text=str(data.get("content_text", "")),
                    title=str(data.get("title", "")),
                    html_language=None,
                    fallback="unknown",
                )

                if detected == "unknown" and source_page_lang in {"fr", "ar"}:
                    detected = source_page_lang
                    reason = "source_page_language"

                if detected == "unknown":
                    path_hint = _path_language_hint(str(data.get("source_url", "")))
                    if path_hint:
                        detected = path_hint
                        reason = "pdf_url_path"

                if detected not in {"fr", "ar", "unknown"}:
                    detected = "unknown"

                if detected == lang_dir:
                    continue

                target_dir = self.pdfs_dir / detected
                target_dir.mkdir(parents=True, exist_ok=True)

                filename = str(data.get("filename", ""))
                if filename:
                    pdf_file = current_dir / filename
                    if not pdf_file.exists():
                        local_path = str(data.get("local_path", ""))
                        if local_path:
                            candidate = Path(local_path)
                            if candidate.exists():
                                pdf_file = candidate

                    if pdf_file.exists():
                        target_pdf = _unique_target_path(target_dir / pdf_file.name)
                        pdf_file.replace(target_pdf)
                        data["filename"] = target_pdf.name
                        data["local_path"] = str(target_pdf)
                        pdf_files_moved += 1

                data["language"] = detected
                data["language_reason"] = f"postprocess:{reason}"

                target_meta = _unique_target_path(target_dir / meta_json.name)
                _write_json_file(target_meta, data)
                meta_json.unlink(missing_ok=True)
                pdf_meta_moved += 1

        logger.info(
            "Post-traitement langue ({}): pages déplacées {}/{}, pdf_meta déplacées {}/{}, pdf déplacés {}",
            self.source.upper(),
            pages_moved,
            pages_checked,
            pdf_meta_moved,
            pdf_meta_checked,
            pdf_files_moved,
        )

    def _log_stats(self) -> None:
        """Affiche les statistiques finales du crawl."""
        s = self._stats
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Crawl terminé : {self.source.upper()}")
        logger.info(f"   Pages crawlées   : {s['pages_crawled']}")
        logger.info(f"   Pages en erreur  : {s['pages_failed']}")
        logger.info(f"   PDFs détectés    : {s['pdfs_found']}")
        logger.info(f"   PDFs téléchargés : {s['pdfs_downloaded']}")
        logger.info(f"   PDFs en erreur   : {s['pdfs_failed']}")
        logger.info(f"{'='*60}\n")
