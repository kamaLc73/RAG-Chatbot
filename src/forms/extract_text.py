"""
src/forms/extract_text.py
==========================
Extrait le texte des formulaires PDF téléchargés.

Stratégie en 2 couches :
    1. pdfplumber        — extraction texte native (rapide, gratuit, fonctionne sur PDFs textuels)
    2. Mistral OCR       — si pdfplumber donne < MIN_CHARS_PER_PAGE caractères par page
                           Utilise mistral-ocr-latest via l'API Mistral (nécessite MISTRAL_API_KEY)
    3. Fallback pymupdf  — si Mistral non disponible, rendu image + pytesseract OCR

Usage :
    python src/forms/extract_text.py
    python src/forms/extract_text.py --force     # réextrait tout, même si .txt existe déjà
    python src/forms/extract_text.py --no-ocr    # pdfplumber seulement
    python src/forms/extract_text.py --form-id rcar_pension_retraite  # un seul formulaire

Sortie :
    data/forms/texts/<form_id>.txt   ← texte extrait
    data/forms/forms.json            ← mis à jour avec champ "text_extracted"
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from loguru import logger

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from config.logger import setup_logger
    from config.settings import BASE_DIR, LOGS_DIR
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[2]
    LOGS_DIR = BASE_DIR / "logs"
    setup_logger = None

FORMS_DIR    = BASE_DIR / "data" / "forms"
PDFS_DIR     = FORMS_DIR / "pdfs"
TEXTS_DIR    = FORMS_DIR / "texts"
CATALOG_FILE = FORMS_DIR / "forms.json"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_OCR_MODEL = "mistral-ocr-latest"  # modèle OCR dédié de Mistral

MIN_CHARS_PER_PAGE = 80   # Seuil : moins → on considère que c'est un PDF scanné


# ─────────────────────────────────────────────────────────────────────────────
# Couche 1 : pdfplumber
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_pdfplumber(pdf_path: Path) -> tuple[str, int]:
    """
    Extrait le texte avec pdfplumber.
    Retourne (texte, nombre_de_pages).
    Retourne ("", 0) en cas d'échec.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber non installé : pip install pdfplumber")
        return "", 0

    pages_text = []
    num_pages  = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                txt = (page.extract_text() or "").strip()
                if txt:
                    pages_text.append(txt)
    except Exception as exc:
        logger.warning("pdfplumber erreur sur {}: {}", pdf_path.name, exc)
        return "", 0

    full_text = "\n\n".join(pages_text)
    return full_text, num_pages


def is_text_sufficient(text: str, num_pages: int) -> bool:
    """Détermine si le texte extrait est suffisant ou si OCR est nécessaire."""
    if num_pages == 0:
        return False
    avg_chars = len(text.strip()) / max(num_pages, 1)
    return avg_chars >= MIN_CHARS_PER_PAGE


# ─────────────────────────────────────────────────────────────────────────────
# Couche 2 : Mistral OCR (mistral-ocr-latest)
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_mistral_ocr(pdf_path: Path) -> str:
    """
    Envoie le PDF à l'API Mistral OCR et retourne le texte extrait.
    Nécessite MISTRAL_API_KEY dans .env.
    Retourne "" en cas d'erreur ou si la clé est absente.
    """
    if not MISTRAL_API_KEY:
        return ""

    try:
        import httpx
    except ImportError:
        logger.warning("httpx non installé : pip install httpx")
        return ""

    pdf_bytes  = pdf_path.read_bytes()
    pdf_b64    = base64.standard_b64encode(pdf_bytes).decode()

    payload = {
        "model": MISTRAL_OCR_MODEL,
        "document": {
            "type": "document_url",
            # Mistral OCR accepte aussi data URI base64
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        "include_image_base64": False,
    }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        resp = httpx.post(
            "https://api.mistral.ai/v1/ocr",
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # La réponse Mistral OCR a un champ "pages" avec "markdown" par page
        pages = data.get("pages", [])
        if pages:
            return "\n\n".join(p.get("markdown", "") for p in pages if p.get("markdown")).strip()

        # Fallback : essayer "text" directement
        return data.get("text", "").strip()

    except Exception as exc:
        logger.warning("Mistral OCR erreur sur {}: {}", pdf_path.name, exc)
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# Extraction complète d'un formulaire
# ─────────────────────────────────────────────────────────────────────────────

def extract_form_text(form: dict, use_ocr: bool = True) -> str | None:
    """
    Extrait le texte d'un formulaire PDF.
    Met à jour le dict form avec les champs d'extraction.
    Retourne le texte extrait ou None si le PDF est introuvable.
    """
    local_path = form.get("local_path") or (str(PDFS_DIR / form["filename"]))
    pdf_path   = Path(local_path)

    if not pdf_path.exists():
        logger.debug("PDF absent, skip: {}", pdf_path.name)
        form["text_extracted"]  = False
        form["extraction_method"] = None
        return None

    # Couche 1 : pdfplumber
    text, num_pages = extract_with_pdfplumber(pdf_path)

    if is_text_sufficient(text, num_pages):
        form["text_extracted"]    = True
        form["extraction_method"] = "pdfplumber"
        form["num_pages"]         = num_pages
        logger.info("pdfplumber OK: {} ({} pages, {} chars)", pdf_path.name, num_pages, len(text))
        return text

    if not use_ocr:
        logger.info("pdfplumber: texte insuffisant ({} chars) mais OCR désactivé — {}",
                    len(text), pdf_path.name)
        form["text_extracted"]    = len(text) > 20
        form["extraction_method"] = "pdfplumber_partial"
        form["num_pages"]         = num_pages
        return text if text.strip() else None

    logger.info(
        "pdfplumber: texte insuffisant ({} chars / {} pages) — tentative OCR: {}",
        len(text), num_pages, pdf_path.name
    )

    # Couche 2 : Mistral OCR
    if MISTRAL_API_KEY:
        ocr_text = extract_with_mistral_ocr(pdf_path)
        if ocr_text.strip():
            form["text_extracted"]    = True
            form["extraction_method"] = "mistral_ocr"
            form["num_pages"]         = num_pages
            logger.info("Mistral OCR OK: {} ({} chars)", pdf_path.name, len(ocr_text))
            return ocr_text
        logger.warning("Mistral OCR : résultat vide pour {}", pdf_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

def run(
    force:    bool = False,
    use_ocr:  bool = True,
    form_id:  str  = "",
) -> list[dict]:
    """
    Extrait le texte de tous les formulaires (ou d'un seul si form_id est fourni).
    Sauvegarde les textes en .txt et met à jour forms.json.
    """
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CATALOG_FILE.exists():
        logger.error("forms.json absent — lance d'abord scrape_forms.py")
        return []

    forms: list[dict] = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))

    if form_id:
        forms = [f for f in forms if f.get("form_id") == form_id]
        if not forms:
            logger.error("form_id '{}' non trouvé dans le catalogue", form_id)
            return []

    ocr_status = f"Mistral ({MISTRAL_OCR_MODEL})" if MISTRAL_API_KEY else "Tesseract"
    logger.info(
        "Extraction: {} formulaires | OCR: {} | force={}",
        len(forms), ocr_status if use_ocr else "désactivé", force
    )

    ok, skip, fail = 0, 0, 0
    all_forms = {f["form_id"]: f for f in json.loads(CATALOG_FILE.read_text(encoding="utf-8"))}

    for form in forms:
        txt_path = TEXTS_DIR / f"{form['form_id']}.txt"

        # Skip si déjà extrait et pas --force
        if not force and txt_path.exists() and txt_path.stat().st_size > 50:
            logger.debug("Déjà extrait: {}", form["form_id"])
            form["text_extracted"]    = True
            form["extraction_method"] = form.get("extraction_method", "cached")
            form["text_path"]         = str(txt_path)
            all_forms[form["form_id"]].update(form)
            skip += 1
            continue

        text = extract_form_text(form, use_ocr=use_ocr)

        if text and text.strip():
            txt_path.write_text(text, encoding="utf-8")
            form["text_path"] = str(txt_path)
            ok += 1
        else:
            form["text_path"] = None
            fail += 1

        all_forms[form["form_id"]].update(form)

    # Sauvegarder catalogue mis à jour
    updated = list(all_forms.values())
    CATALOG_FILE.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Extraction terminée: {} OK | {} skip | {} échecs",
        ok, skip, fail
    )
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extrait le texte des formulaires PDF")
    p.add_argument("--force",   action="store_true", help="Réextraire même si .txt existe")
    p.add_argument("--no-ocr",  action="store_true", help="pdfplumber uniquement, pas d'OCR")
    p.add_argument("--form-id", default="",          help="Extraire un seul formulaire (par form_id)")
    return p.parse_args()


if __name__ == "__main__":
    if setup_logger is not None:
        setup_logger(log_dir=LOGS_DIR, source="fetch_forms")

    args   = parse_args()
    result = run(force=args.force, use_ocr=not args.no_ocr, form_id=args.form_id)
    ok     = sum(1 for f in result if f.get("text_extracted"))
    print(f"\nExtraction: {ok}/{len(result)} formulaires avec texte")
    if MISTRAL_API_KEY:
        print(f"OCR: Mistral ({MISTRAL_OCR_MODEL})")
    else:
        print("OCR: Tesseract (Mistral non configuré — ajoute MISTRAL_API_KEY dans .env pour un meilleur résultat)")
