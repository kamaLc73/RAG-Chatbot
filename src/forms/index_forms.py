"""
src/forms/index_forms.py
=========================
Indexe les formulaires CNRA/RCAR dans une collection ChromaDB dédiée.

Chaque Document LangChain représente UN formulaire (pas de chunking fin) :
    page_content = titre + description + texte_extrait_du_PDF (tronqué)
    metadata     = form_id, org, category, page_url, pdf_url, filename

Pourquoi pas de chunking fin ?
    Les formulaires PDF sont courts (1-3 pages). Garder chaque formulaire
    en un seul chunk permet au retriever de renvoyer la fiche complète
    sans recoller des fragments.

Usage :
    python src/forms/index_forms.py
    python src/forms/index_forms.py --reset
    python src/forms/index_forms.py --no-reset    # ajouter sans supprimer
"""

import argparse
import json
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

import os

FORMS_DIR       = BASE_DIR / "data" / "forms"
CATALOG_FILE    = FORMS_DIR / "forms.json"
TEXTS_DIR       = FORMS_DIR / "texts"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma_db_unified"
COLLECTION_NAME = "rcar_cnra_unified"
EMBEDDING_MODEL = "BAAI/bge-m3"

MAX_TEXT_CHARS = 2000   # Tronquer les textes trop longs (formulaires exceptionnellement longs)


# ─────────────────────────────────────────────────────────────────────────────
# Construction des Documents LangChain
# ─────────────────────────────────────────────────────────────────────────────

def build_documents(forms: list[dict]) -> list:
    """
    Crée un Document LangChain par formulaire.
    Le page_content combine titre + catégorie + description + texte PDF.
    """
    from langchain_core.documents import Document

    docs  = []
    skipped = 0

    for form in forms:
        form_id  = form.get("form_id", "")
        title    = form.get("title",    "Formulaire CNRA/RCAR")
        category = form.get("category", "")
        org      = form.get("org",      "")
        desc     = form.get("description", "")

        # Texte PDF : charger depuis le .txt si disponible
        pdf_text = ""
        txt_path_str = form.get("text_path")
        if txt_path_str:
            txt_path = Path(txt_path_str)
            if txt_path.exists():
                pdf_text = txt_path.read_text(encoding="utf-8").strip()[:MAX_TEXT_CHARS]

        # Construire le contenu indexé
        # Le titre et la catégorie sont répétés pour booster leur poids sémantique
        parts = [
            f"Organisation : {org}",
            f"Catégorie : {category}",
            f"Formulaire : {title}",
        ]
        if desc and desc != title:
            parts.append(f"Description : {desc}")
        if pdf_text:
            parts.append(f"\nContenu du formulaire :\n{pdf_text}")

        content = "\n".join(parts)

        if len(content.strip()) < 30:
            logger.debug("Contenu trop court, skip: {}", form_id)
            skipped += 1
            continue

        docs.append(Document(
            page_content=content,
            metadata={
                "source":        "forms",
                "type":          "form",
                "form_id":       form_id,
                "org":           org,
                "category":      category,
                "category_slug": form.get("category_slug", ""),
                "title":         title,
                "description":   desc,
                "page_url":      form.get("page_url", ""),
                "pdf_url":       form.get("pdf_url",  ""),
                "filename":      form.get("filename", ""),
                "has_pdf_text":  bool(pdf_text),
                "extraction_method": form.get("extraction_method", "none"),
            },
        ))

    logger.info("Documents créés: {} ({} skippés)", len(docs), skipped)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Indexation ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def index_forms(
    catalog_file:    Path = CATALOG_FILE,
    vectorstore_dir: Path = VECTORSTORE_DIR,
    collection_name: str  = COLLECTION_NAME,
    embedding_model: str  = EMBEDDING_MODEL,
) -> dict:
    """
    Pipeline complet : catalog → documents → ChromaDB.
    Retourne un résumé de l'indexation.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    if not catalog_file.exists():
        raise FileNotFoundError(
            f"forms.json introuvable: {catalog_file}\n"
            "Lance d'abord: python src/forms/scrape_forms.py"
        )

    forms = json.loads(catalog_file.read_text(encoding="utf-8"))
    logger.info("Catalogue: {} formulaires", len(forms))

    documents = build_documents(forms)
    if not documents:
        raise ValueError("Aucun document valide. Vérifie que scrape_forms.py et extract_text.py ont tourné.")

    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    hf_token = os.getenv("HF_TOKEN", "").strip()
    model_kwargs: dict = {"device": device}
    if hf_token:
        model_kwargs["token"] = hf_token

    logger.info("Device: {} | Modèle: {}", device, embedding_model)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    # Supprimer uniquement les chunks type="form" dans la collection unifiée
    vectorstore = Chroma(
        persist_directory=str(vectorstore_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    try:
        vectorstore._collection.delete(where={"type": "form"})
        logger.info("Chunks type=form supprimés de la collection unifiée avant réindexation")
    except Exception as exc:
        logger.warning("Impossible de supprimer les chunks form existants: {}", exc)

    logger.info("Indexation de {} formulaires...", len(documents))
    vectorstore.add_documents(documents)

    summary = {
        "vectorstore_dir":  str(vectorstore_dir),
        "collection_name":  collection_name,
        "embedding_model":  embedding_model,
        "forms_total":      len(forms),
        "forms_indexed":    len(documents),
        "has_pdf_text":     sum(1 for d in documents if d.metadata.get("has_pdf_text")),
    }
    logger.info("Indexation terminée: {}", summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Indexe les formulaires dans ChromaDB")
    p.add_argument("--reset",    action="store_true", default=True,  help="Supprimer la collection existante (défaut)")
    p.add_argument("--no-reset", action="store_true",                help="Ne pas supprimer la collection existante")
    p.add_argument("--collection-name", default=COLLECTION_NAME)
    p.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    return p.parse_args()


if __name__ == "__main__":
    if setup_logger is not None:
        setup_logger(log_dir=LOGS_DIR, source="index_forms")

    args   = parse_args()
    result = index_forms(
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
    )
    print("\nIndexation terminée:")
    for k, v in result.items():
        print(f"  {k}: {v}")