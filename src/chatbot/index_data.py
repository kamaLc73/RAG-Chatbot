"""
src/indexing/index_data.py
===========================
Construit (ou met à jour) le vectorstore ChromaDB à partir des fichiers
FAQ RCAR/CNRA stockés dans data/supportstagerag.

Stratégie de mise à jour :
    - La collection est PARTAGÉE avec les vidéos et les formulaires.
    - Avant réindexation, SEULS les chunks type="doc" sont supprimés
      (les chunks type="video" et type="form" sont préservés).
    - La suppression est VÉRIFIÉE : si elle échoue ou est incomplète,
      l'indexation est annulée pour éviter les doublons.
"""
# ── Compatibilité Python 3.8+ pour les annotations de type ───────────────────
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "index_data.log"

TARGET_SITES = ("rcar", "cnra")
TARGET_SOURCE_TYPES = ("faq", "web", "bibliotheque")
TARGET_LANGUAGE = "fr"
SUPPORTED_EXTENSIONS = (".md", ".txt")
SOURCE_TYPE_PRIORITY = {
    "faq": 3,
    "web": 2,
    "bibliotheque": 1,
}

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "supportstagerag"
DEFAULT_VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore" / "chroma_db_unified"

DEFAULT_CHUNK_SIZE    = 1200  # Nombre de caractères par chunk
DEFAULT_CHUNK_OVERLAP = 180
INDEX_BATCH_SIZE      = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    force=True,
)


def _resolve_project_path(path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _is_supported_document(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def _list_supported_documents(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if _is_supported_document(path))


def _resolve_embedding_device():
    """Prefer CUDA when available, otherwise fallback to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def analyze_processed_data(
    processed_root=DEFAULT_PROCESSED_ROOT,
    sites=TARGET_SITES,
    source_types=TARGET_SOURCE_TYPES,
    language=TARGET_LANGUAGE,
):
    """Analyze FAQ file availability for the selected sites and source types."""
    root = _resolve_project_path(processed_root)
    summary = {}

    for site in sites:
        for source_type in source_types:
            folder = root / site / source_type
            key = f"{site}/{source_type}"

            if not folder.exists():
                summary[key] = {
                    "exists": False,
                    "file_count": 0,
                    "total_bytes": 0,
                    "avg_bytes": 0,
                    "min_bytes": 0,
                    "max_bytes": 0,
                    "extensions": {},
                }
                continue

            files = _list_supported_documents(folder)
            sizes = [path.stat().st_size for path in files]
            total_bytes = sum(sizes)
            file_count = len(files)
            extension_counts = {
                ext: sum(1 for file in files if file.suffix.lower() == ext)
                for ext in SUPPORTED_EXTENSIONS
            }

            summary[key] = {
                "exists": True,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "avg_bytes": round(total_bytes / file_count, 2) if file_count else 0,
                "min_bytes": min(sizes) if sizes else 0,
                "max_bytes": max(sizes) if sizes else 0,
                "extensions": extension_counts,
            }

    return summary


def collect_files(
    processed_root=DEFAULT_PROCESSED_ROOT,
    sites=TARGET_SITES,
    source_types=TARGET_SOURCE_TYPES,
    language=TARGET_LANGUAGE,
):
    """Collect prepared documents recursively for RCAR and CNRA."""
    root = _resolve_project_path(processed_root)
    collected = []

    for site in sites:
        for source_type in source_types:
            folder = root / site / source_type
            if not folder.exists():
                logging.warning("Dossier absent: %s", folder)
                continue

            files = _list_supported_documents(folder)
            collected.extend(files)
            md_count  = sum(1 for file in files if file.suffix.lower() == ".md")
            txt_count = sum(1 for file in files if file.suffix.lower() == ".txt")
            logging.info(
                "%s: %s fichiers %s (md=%s, txt=%s)",
                folder, len(files), source_type, md_count, txt_count,
            )

    return collected


def load_documents(file_paths, processed_root=DEFAULT_PROCESSED_ROOT):
    """Load prepared files into LangChain Document objects with metadata."""
    root = _resolve_project_path(processed_root)
    documents = []

    for file_path in file_paths:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as exc:
            logging.warning("Lecture ignoree pour %s: %s", file_path, exc)
            continue

        if not text:
            continue

        try:
            relative = file_path.resolve().relative_to(root)
            parts = relative.parts
            site        = parts[0] if len(parts) > 0 else "unknown"
            source_type = parts[1] if len(parts) > 1 else "unknown"
            doc_scope   = parts[2] if len(parts) > 2 else "unknown"
            topic       = "/".join(parts[3:-1]) if len(parts) > 4 else ""
            language    = TARGET_LANGUAGE
        except Exception:
            relative    = file_path
            site        = "unknown"
            source_type = "unknown"
            doc_scope   = "unknown"
            topic       = ""
            language    = TARGET_LANGUAGE

        source_priority = SOURCE_TYPE_PRIORITY.get(source_type, 0)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source":          str(file_path.as_posix()),
                    "relative_source": str(relative).replace("\\", "/"),
                    "org":             site if site in ("cnra", "rcar") else "both",
                    "type":            "doc",
                    "site":            site,
                    "source_type":     source_type,
                    "source_priority": source_priority,
                    "faq_scope":       doc_scope if source_type == "faq" else "",
                    "doc_scope":       doc_scope,
                    "faq_topic":       topic if source_type == "faq" else "",
                    "doc_topic":       topic,
                    "language":        language,
                    "file_name":       file_path.name,
                    "file_extension":  file_path.suffix.lower(),
                },
            )
        )

    return documents


def build_chunker(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Create a chunk splitter tuned for FAQ markdown/text files."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\nQ:",
            "\nQ",
            "\n**Q:",
            "\nR:",
            "\n**R:",
            "\n================================================================================",
            "\n---",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def chunk_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    splitter = build_chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Ajoute un index de chunk stable par fichier source.
    per_source_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("relative_source", chunk.metadata.get("source", "unknown"))
        current_idx = per_source_index.get(source, 0)
        chunk.metadata["chunk_index"] = current_idx
        per_source_index[source] = current_idx + 1

    return chunks


def _delete_doc_chunks(vectorstore: Chroma) -> None:
    """
    Supprime TOUS les chunks de type='doc' de la collection partagée,
    puis vérifie que la suppression est complète.

    La vérification en deux temps (count avant → delete → count après)
    est essentielle car Chroma ne lève pas toujours d'exception en cas
    d'échec partiel. Sans vérification, on risque d'accumuler des doublons
    à chaque réindexation.

    Raises:
        RuntimeError: si la suppression est incomplète après l'appel.
        Exception:    si une erreur inattendue survient lors du delete.
    """
    # Comptage des entrées existantes — include=[] pour éviter de rapatrier
    # tout le contenu (on n'a besoin que des IDs).
    result_before = vectorstore._collection.get(where={"type": "doc"}, include=[])
    n_before = len(result_before.get("ids", []))

    if n_before == 0:
        logging.info(
            "Aucun chunk type=doc dans la collection — première indexation ou collection vierge."
        )
        return

    logging.info("Suppression de %d chunks type=doc avant réindexation...", n_before)
    vectorstore._collection.delete(where={"type": "doc"})

    # Vérification : s'assurer que plus aucun chunk doc ne subsiste
    result_after = vectorstore._collection.get(where={"type": "doc"}, include=[])
    n_after = len(result_after.get("ids", []))

    if n_after > 0:
        raise RuntimeError(
            f"Suppression incomplète : {n_after}/{n_before} chunks type=doc toujours présents. "
            f"Réindexation annulée pour éviter les doublons. "
            f"Vérifiez l'état de la collection ChromaDB."
        )

    logging.info("Chunks type=doc supprimés avec succès : %d entrées retirées.", n_before)


def index_data(
    processed_root=DEFAULT_PROCESSED_ROOT,
    persist_directory=DEFAULT_VECTORSTORE_DIR,
    collection_name="rcar_cnra_unified",
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """
    Construit (ou met à jour) le vectorstore ChromaDB à partir des FAQ RCAR/CNRA.

    La collection est partagée avec les vidéos et formulaires — on ne supprime
    que les chunks type='doc' avant de réindexer.
    """
    analysis = analyze_processed_data(processed_root=processed_root)
    for bucket, stats in analysis.items():
        logging.info(
            "%s | exists=%s | files=%s | total=%s | avg=%s | min=%s | max=%s | ext=%s",
            bucket,
            stats["exists"],
            stats["file_count"],
            stats["total_bytes"],
            stats["avg_bytes"],
            stats["min_bytes"],
            stats["max_bytes"],
            stats["extensions"],
        )

    file_paths = collect_files(processed_root=processed_root)
    if not file_paths:
        raise ValueError(
            "Aucun fichier FAQ (.md/.txt) trouve pour RCAR/CNRA dans data/supportstagerag."
        )

    documents = load_documents(file_paths, processed_root=processed_root)
    if not documents:
        raise ValueError("Aucun document exploitable trouve apres lecture des fichiers FAQ.")

    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Aucun chunk genere. Verifiez les contenus FAQ md/txt.")

    persist_path = _resolve_project_path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    device = _resolve_embedding_device()
    hf_token = os.getenv("HF_TOKEN")
    logging.info("Embedding device utilise pour l'indexation: %s", device)
    if not hf_token:
        logging.warning(
            "HF_TOKEN non configure: telechargement Hugging Face en mode non authentifie "
            "(limites plus strictes)."
        )

    model_kwargs: dict = {"device": device}
    if hf_token:
        model_kwargs["token"] = hf_token

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Suppression sécurisée des anciens chunks doc ───────────────────────────
    # On ouvre la collection AVANT d'ajouter les nouveaux chunks pour pouvoir
    # vérifier et supprimer les anciens. Si la suppression échoue (partielle ou
    # totale), on lève une exception et on n'indexe rien — pas de doublons.
    vectorstore = Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    try:
        _delete_doc_chunks(vectorstore)
    except Exception as exc:
        logging.error(
            "Suppression des chunks doc échouée : %s — réindexation annulée.", exc
        )
        raise

    # ── Indexation ────────────────────────────────────────────────────────────
    for start in range(0, len(chunks), INDEX_BATCH_SIZE):
        batch = chunks[start:start + INDEX_BATCH_SIZE]
        vectorstore.add_documents(batch)
        logging.info(
            "Batch docs %s/%s indexe",
            min(start + INDEX_BATCH_SIZE, len(chunks)),
            len(chunks),
        )

    logging.info("ChromaDB sauvegarde dans %s", persist_path)
    logging.info("Collection: %s", collection_name)
    logging.info("Documents charges: %s", len(documents))
    logging.info("Chunks indexes: %s", len(chunks))
    logging.info("Chunk size=%s | overlap=%s", chunk_size, chunk_overlap)

    return {
        "persist_directory": str(persist_path),
        "collection_name":   collection_name,
        "documents":         len(documents),
        "chunks":            len(chunks),
        "chunk_size":        chunk_size,
        "chunk_overlap":     chunk_overlap,
    }


if __name__ == "__main__":
    result = index_data()
    logging.info("Indexation terminee: %s", result)
