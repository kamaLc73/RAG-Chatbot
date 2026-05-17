from pathlib import Path
import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CHEMINS RACINE
# ─────────────────────────────────────────────
# settings.py est dans src/config, donc la racine projet est 3 niveaux au-dessus.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
LOGS_DIR = BASE_DIR / "logs"

# ─────────────────────────────────────────────
# VESPA
# ─────────────────────────────────────────────
VESPA_URL = os.getenv("VESPA_URL", "http://localhost")
VESPA_PORT = int(os.getenv("VESPA_PORT", "8080"))
VESPA_CONTENT_CLUSTER = os.getenv("VESPA_CONTENT_CLUSTER", "rcar_cnra")

# ─────────────────────────────────────────────
# SOURCES À CRAWLER
# ─────────────────────────────────────────────
# Pour ajouter une nouvelle source plus tard, ajoute simplement une entrée ici.
SOURCES: dict[str, str] = {
    "cnra": "https://www.cnra.ma",
    "rcar": "https://www.rcar.ma",
}

# ─────────────────────────────────────────────
# URLS À EXCLURE EN MODE PRODUCTION
# (authentification, formulaires d'inscription, espaces compte)
# ─────────────────────────────────────────────
PRODUCTION_SKIP_URL_PATTERNS: list[str] = [
    r"(?:^|/)login(?:/|$)",
    r"(?:^|/)connexion(?:/|$)",
    r"(?:^|/)auth(?:/|$)",
    r"(?:^|/)signup(?:/|$)",
    r"(?:^|/)register(?:/|$)",
    r"(?:^|/)inscription(?:/|$)",
    r"(?:^|/)compte(?:/|$)",
    r"(?:^|/)mon-compte(?:/|$)",
    r"(?:^|/)espace-client(?:/|$)",
    r"(?:^|/)espace-personnel(?:/|$)",
    # Le périmètre métier visé est FR/AR: ignorer sections anglaises.
    r"(?:^|/)en(?:/|$)",
    r"/index\.php/en(?:/|$)",
    r"[?&](?:lang|locale)=en(?:$|[&#])",
    # Sous-domaines non documentaires.
    r"https?://ehtiyati\.rcar\.ma(?:/|$)",
]

# ─────────────────────────────────────────────
# PARAMÈTRES DU CRAWLER
# ─────────────────────────────────────────────
CRAWLER_CONFIG = {
    # Délai minimum entre deux requêtes (secondes) — respect des serveurs
    "request_delay_min": 1.0,
    "request_delay_max": 2.5,

    # Timeout d'une requête (secondes)
    "request_timeout": 30,

    # Nombre de retries en cas d'échec
    "max_retries": 3,

    # Profondeur maximale de crawl (1 = page racine uniquement, 0 = illimité)
    "max_depth": 0,

    # Nombre max de pages crawlées par source (0 = illimité)
    "max_pages_per_source": 0,

    # Nombre max de PDFs à traiter par source (0 = illimité)
    "max_pdfs_per_source": 0,

    # Nombre de workers parallèles (concurrent requests)
    "max_concurrency": 3,

    # User-Agent présenté aux serveurs
    "user_agent": "ENSAM-RAG-Bot/1.0 (Projet PFE; contact: kamal_dehbi@um5.ac.ma)",

    # Respecter robots.txt
    "respect_robots_txt": True,

    # Si True, les contenus ambigus sont classés en "unknown" (au lieu de forcer fr/ar).
    "strict_language_classification": True,

    # Mode production : exclusion automatique des URLs non utiles (login/inscription/...)
    "production_mode": False,
    "production_skip_url_patterns": PRODUCTION_SKIP_URL_PATTERNS,

    # Nombre de sources crawlées en parallèle (CNRA/RCAR/...)
    "source_parallelism": 2,
}

# ─────────────────────────────────────────────
# PARAMÈTRES DE TÉLÉCHARGEMENT DES PDFs
# ─────────────────────────────────────────────
PDF_CONFIG = {
    # Timeout téléchargement PDF (secondes)
    "download_timeout": 60,

    # Taille max d'un PDF accepté (bytes) — 50 MB
    "max_pdf_size_bytes": 50 * 1024 * 1024,

    # Nombre de workers parallèles pour le téléchargement
    "download_concurrency": 3,

    # Timeout max pour le pipeline de téléchargement d'un PDF
    "per_pdf_task_timeout_seconds": 240,

    # Retries automatiques pour le téléchargement des PDFs
    "download_max_retries": 3,

    # Backoff initial entre retries (en secondes)
    "download_retry_backoff_base": 1.0,
}

# ─────────────────────────────────────────────
# SÉLECTEURS HTML À EXCLURE DU CONTENU
# (menus, footer, navigation, cookies…)
# ─────────────────────────────────────────────
HTML_EXCLUDE_SELECTORS: list[str] = [
    "nav", "header", "footer",
    ".navbar", ".nav", ".menu", ".sidebar",
    ".breadcrumb", ".pagination",
    ".cookie", ".cookies", "#cookie",
    ".social", ".share",
    "script", "style", "noscript",
    "[class*='cookie']", "[class*='banner']",
    "[id*='cookie']",
]

# ─────────────────────────────────────────────
# EXTENSIONS DE FICHIERS CONSIDÉRÉES COMME PDFs
# ─────────────────────────────────────────────
PDF_EXTENSIONS: set[str] = {".pdf", ".PDF"}
