"""
src/crawler/crawler.py
===========================
Crawler principal utilisant Crawlee (BeautifulSoupCrawler).
Pour chaque page :
  - Extrait le contenu textuel propre (Markdown simplifié)
    - Détecte les liens PDF et les traite via data_handler
  - Enqueue les liens internes du même domaine
  - Sauvegarde les données en JSON
"""

import asyncio
import random
import re
import sys
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx
from crawlee import ConcurrencySettings, Request
from crawlee.http_clients import HttpxHttpClient
from crawlee.crawlers import BeautifulSoupCrawler, BeautifulSoupCrawlingContext
from loguru import logger

# Ajout du parent dans le path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import HTML_EXCLUDE_SELECTORS
from config.logger import log_failed_url
from crawler.data_handler import (
    extract_page_metadata,
    find_pdf_links,
    html_to_clean_text,
    path_language_hint,
    process_pdf,
    save_page,
)


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

ARTIFICIAL_LANG_LOOP_RE = re.compile(r"/(?:ar|fr)(?:/(?:ar|fr)){2,}(?:/|$)", re.I)
ENQUEUE_EXCLUDE_PATTERNS = [
    re.compile(r".*\.pdf(?:$|[?#])", re.I),
    re.compile(r".*#.*"),
    re.compile(r".*\.(?:xlsx|xls|doc|docx|ppt|pptx|odt|ods|odp|rtf|csv)(?:$|[?#])", re.I),
    re.compile(
        r".*\.(?:jpg|jpeg|png|gif|svg|webp|ico|css|js|zip|rar|7z|mp4|mp3|woff|woff2|ttf|eot)(?:$|[?#])",
        re.I,
    ),
]

# ─────────────────────────────────────────────
# CLASSE PRINCIPALE DU CRAWLER
# ─────────────────────────────────────────────

class Crawler:
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
        self._pdf_queue_seen: set[str] = set()
        # (pdf_url, source_page_url, source_page_language_hint, pdf_url_language_hint)
        self._stats = {
            "pages_crawled": 0,
            "pages_failed": 0,
            "pdfs_found": 0,
            "pdfs_downloaded": 0,
            "pdfs_failed": 0,
        }

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

        # Rapport final
        self._log_stats()
        return self._stats

    async def _crawl_pages(self) -> None:
        """Configure et lance le BeautifulSoupCrawler de Crawlee."""

        # Référence vers self pour l'utiliser dans le handler (closure)
        crawler_self = self
        request_delay_min = float(self.crawler_config.get("request_delay_min", 1.0))
        request_delay_max = float(self.crawler_config.get("request_delay_max", 2.5))
        max_depth = int(self.crawler_config.get("max_depth", 0) or 0)
        default_headers = {
            "User-Agent": self.crawler_config["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,ar;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
        }

        # Configuration Crawlee
        max_concurrency = max(1, int(self.crawler_config.get("max_concurrency", 3)))
        soup_crawler = BeautifulSoupCrawler(
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

        @soup_crawler.router.default_handler
        async def handle_page(context: BeautifulSoupCrawlingContext) -> None:
            """Handler appelé pour chaque page visitée."""
            url = context.request.url
            soup = context.soup

            # Delai de politesse variable entre requetes.
            await asyncio.sleep(
                random.uniform(
                    request_delay_min,
                    request_delay_max,
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

                source_page_language_hint = "unknown"
                if html_lang_raw.startswith("ar"):
                    source_page_language_hint = "ar"
                elif html_lang_raw.startswith("fr"):
                    source_page_language_hint = "fr"
                else:
                    url_lang_hint = path_language_hint(url)
                    if url_lang_hint in {"fr", "ar"}:
                        source_page_language_hint = url_lang_hint

                metadata["html_language_raw"] = html_lang_raw
                metadata["source_page_language_hint"] = source_page_language_hint
                metadata["crawler_stage"] = "raw_collection"

                # 3. Détecter les liens PDF
                pdf_links = find_pdf_links(soup, url)
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
                        queue_key = pdf_url

                        # Éviter les doublons dans la file
                        if max_pdfs and len(crawler_self._pdf_queue) >= max_pdfs:
                            max_pdfs_reached = True
                            break
                        if queue_key not in crawler_self._pdf_queue_seen:
                            pdf_url_language_hint = path_language_hint(pdf_url) or "unknown"
                            crawler_self._pdf_queue.append(
                                (pdf_url, url, source_page_language_hint, pdf_url_language_hint)
                            )
                            crawler_self._pdf_queue_seen.add(queue_key)

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
                    language_hint=source_page_language_hint,
                    pages_dir=crawler_self.pages_dir,
                )

                # Limiter l'exploration si une profondeur max explicite est configurée.
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
                    if ARTIFICIAL_LANG_LOOP_RE.search(parsed.path):
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
                    exclude=ENQUEUE_EXCLUDE_PATTERNS,
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
        await soup_crawler.run(seed_requests)

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
                    source_page_language_hint,
                    pdf_url_language_hint,
                    semaphore,
                )
                for pdf_url, page_url, source_page_language_hint, pdf_url_language_hint in self._pdf_queue
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compter les succès/échecs
        for (pdf_url, page_url, _source_page_language_hint, _pdf_url_language_hint), result in zip(self._pdf_queue, results):
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
        source_page_language_hint: str,
        pdf_url_language_hint: str,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """Telecharge et sauvegarde un seul PDF (avec semaphore)."""
        async with semaphore:
            # Délai de politesse entre téléchargements
            await asyncio.sleep(self.crawler_config["request_delay_min"])
            try:
                timeout_seconds = int(self.pdf_config.get("per_pdf_task_timeout_seconds", 240))
                return await asyncio.wait_for(
                    process_pdf(
                        client=client,
                        pdf_url=pdf_url,
                        source_page_url=page_url,
                        source=self.source,
                        source_page_language_hint=source_page_language_hint,
                        pdf_url_language_hint=pdf_url_language_hint,
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
