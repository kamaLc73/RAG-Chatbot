from __future__ import annotations

import argparse
import json
import logging
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

from store.vespa_store import delete_all_docs, feed_documents, make_vespa_app, stable_data_id

logging.getLogger("vespa").setLevel(logging.WARNING)
logging.getLogger("vespa.application").setLevel(logging.WARNING)

FORMS_DIR = BASE_DIR / "data" / "forms"
CATALOG_FILE = FORMS_DIR / "forms.json"
TEXTS_DIR = FORMS_DIR / "texts"
VESPA_URL = os.getenv("VESPA_URL", "http://localhost")
VESPA_PORT = int(os.getenv("VESPA_PORT", "8080"))
VESPA_CONTENT_CLUSTER = os.getenv("VESPA_CONTENT_CLUSTER", "rcar_cnra")
EMBEDDING_MODEL = "BAAI/bge-m3"

MAX_TEXT_CHARS = 2000


def build_documents(forms: list[dict]) -> list:
    from langchain_core.documents import Document

    docs = []
    skipped = 0

    for form in forms:
        form_id = form.get("form_id", "")
        title = form.get("title", "Formulaire CNRA/RCAR")
        category = form.get("category", "")
        org = form.get("org", "").strip().lower() or "both"
        desc = form.get("description", "")

        pdf_text = ""
        txt_path_str = form.get("text_path")
        if txt_path_str:
            txt_path = Path(txt_path_str)
            if txt_path.exists():
                pdf_text = txt_path.read_text(encoding="utf-8").strip()[:MAX_TEXT_CHARS]

        parts = [
            f"Organisation : {org}",
            f"Categorie : {category}",
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

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "forms",
                    "type": "form",
                    "form_id": form_id,
                    "org": org,
                    "category": category,
                    "category_slug": form.get("category_slug", ""),
                    "title": title,
                    "description": desc,
                    "page_url": form.get("page_url", ""),
                    "pdf_url": form.get("pdf_url", ""),
                    "filename": form.get("filename", ""),
                    "has_pdf_text": bool(pdf_text),
                    "extraction_method": form.get("extraction_method", "none"),
                },
            )
        )

    logger.info("Documents crees: {} ({} skippes)", len(docs), skipped)
    return docs


def _resolve_embedding_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def index_forms(
    catalog_file: Path = CATALOG_FILE,
    vespa_url: str = VESPA_URL,
    vespa_port: int = VESPA_PORT,
    content_cluster: str = VESPA_CONTENT_CLUSTER,
    embedding_model: str = EMBEDDING_MODEL,
    reset: bool = True,
) -> dict:
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
        raise ValueError("Aucun document valide. Verifie que scrape_forms.py et extract_text.py ont tourne.")

    device = _resolve_embedding_device()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    model_kwargs: dict = {"device": device}
    if hf_token:
        model_kwargs["token"] = hf_token

    logger.info("Device: {} | Modele: {}", device, embedding_model)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    app = make_vespa_app(vespa_url, vespa_port)
    if reset:
        logger.info("Suppression des documents Vespa schema=form avant reindexation")
        delete_all_docs(app, "form", content_cluster)

    logger.info("Calcul embeddings pour {} formulaires...", len(documents))
    vectors = embeddings.embed_documents([doc.page_content for doc in documents])
    vespa_documents = []
    for doc, vector in zip(documents, vectors):
        meta = doc.metadata
        form_id = stable_data_id("form", meta.get("form_id") or meta.get("title"))
        vespa_documents.append(
            {
                "id": form_id,
                "fields": {
                    "form_id": form_id,
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "content": doc.page_content,
                    "org": meta.get("org", "both"),
                    "pdf_url": meta.get("pdf_url", ""),
                    "page_url": meta.get("page_url", ""),
                    "embedding": vector,
                },
            }
        )

    logger.info("Feed Vespa schema=form: {} documents...", len(vespa_documents))
    fed = feed_documents(app, schema="form", documents=vespa_documents)

    summary = {
        "vespa_url": vespa_url,
        "vespa_port": vespa_port,
        "schema": "form",
        "embedding_model": embedding_model,
        "forms_total": len(forms),
        "forms_indexed": fed,
        "has_pdf_text": sum(1 for d in documents if d.metadata.get("has_pdf_text")),
    }
    logger.info("Indexation terminee: {}", summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Indexe les formulaires dans Vespa")
    p.add_argument("--no-reset", action="store_true", help="Ne pas supprimer les formulaires existants")
    p.add_argument("--vespa-url", default=VESPA_URL)
    p.add_argument("--vespa-port", type=int, default=VESPA_PORT)
    p.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    return p.parse_args()


if __name__ == "__main__":
    if setup_logger is not None:
        setup_logger(log_dir=LOGS_DIR, source="index_forms")

    args = parse_args()
    result = index_forms(
        vespa_url=args.vespa_url,
        vespa_port=args.vespa_port,
        embedding_model=args.embedding_model,
        reset=not args.no_reset,
    )
    print("\nIndexation terminee:")
    for k, v in result.items():
        print(f"  {k}: {v}")
