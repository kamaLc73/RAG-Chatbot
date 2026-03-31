"""
config/logger.py
================
Configuration centralisee du logging pour la phase 1.

Ce module fournit :
  - setup_logger() : configuration Loguru (console + fichier)
  - log_failed_url() : ecriture robuste des URLs en echec dans failed_urls.json
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


# Chemin partage du fichier des URLs en echec
_FAILED_URLS_PATH: Path | None = None
_FAILED_URLS_LOCK = threading.Lock()


def setup_logger(log_dir: Path, source: str = "crawl") -> None:
    """
    Initialise Loguru pour tout le projet (console + fichier).

    Args:
        log_dir: Dossier des logs.
        source: Prefixe du fichier principal de log.
    """
    global _FAILED_URLS_PATH

    log_dir.mkdir(parents=True, exist_ok=True)
    _FAILED_URLS_PATH = log_dir / "failed_urls.json"

    # Reinitialiser les handlers pour eviter les doublons si setup est rappele.
    logger.remove()

    # Sortie console lisible pendant l'execution du crawl.
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - "
        "<level>{message}</level>",
    )

    # Fichier detaille pour audit/debug (rotation + retention).
    logger.add(
        sink=log_dir / f"{source}.log",
        level="DEBUG",
        encoding="utf-8",
        rotation="10 MB",
        retention="30 days",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} - {message}",
    )

    logger.info(f"Logger initialise. Dossier logs: {log_dir}")
    logger.info(f"Fichier des URLs echouees: {_FAILED_URLS_PATH}")


def log_failed_url(
    url: str,
    reason: str,
    source: str,
    stage: str,
    context_url: str | None = None,
) -> None:
    """
    Ajoute une entree dans failed_urls.json sans interrompre le pipeline.

    Args:
        url: URL en echec.
        reason: Motif d'echec (exception/message).
        source: Source metier (cnra/rcar).
        stage: Etape concernee (page_crawl, pdf_download, ...).
        context_url: URL contexte (ex: page contenant le lien PDF).
    """
    failed_path = _FAILED_URLS_PATH or Path("logs") / "failed_urls.json"
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    item = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "stage": stage,
        "url": url,
        "context_url": context_url,
        "reason": reason,
    }

    try:
        with _FAILED_URLS_LOCK:
            existing: list[dict]
            if failed_path.exists():
                with open(failed_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    existing = json.loads(content) if content else []
                    if not isinstance(existing, list):
                        existing = []
            else:
                existing = []

            existing.append(item)

            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

    except Exception as exc:
        # Ne jamais faire echouer le crawl a cause du logger secondaire.
        logger.error(f"Impossible d'ecrire failed_urls.json pour {url}: {exc}")
