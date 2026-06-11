from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from store.vespa_store import delete_all_docs, feed_documents, make_vespa_app, stable_data_id
from config.embedding_cache import make_huggingface_embeddings, resolve_embedding_device

load_dotenv(PROJECT_ROOT / ".env")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "index_data.log"

TARGET_SITES = ("rcar", "cnra")
TARGET_SOURCE_TYPES: tuple[str, ...] = ()
TARGET_LANGUAGE = "fr"
SUPPORTED_EXTENSIONS = (".md", ".txt")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "supportstagerag"
VESPA_URL = os.getenv("VESPA_URL", "http://localhost")
VESPA_PORT = int(os.getenv("VESPA_PORT", "8080"))
VESPA_CONTENT_CLUSTER = os.getenv("VESPA_CONTENT_CLUSTER", "rcar_cnra")

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    force=True,
)
logging.getLogger("vespa").setLevel(logging.WARNING)
logging.getLogger("vespa.application").setLevel(logging.WARNING)


def _resolve_project_path(path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _is_supported_document(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def _list_supported_documents(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if _is_supported_document(path))


def analyze_processed_data(
    processed_root=DEFAULT_PROCESSED_ROOT,
    sites=TARGET_SITES,
    source_types=TARGET_SOURCE_TYPES,
    language=TARGET_LANGUAGE,
):
    root = _resolve_project_path(processed_root)
    summary = {}

    for site in sites:
        selected_source_types = source_types or _discover_source_types(root, site)
        for source_type in selected_source_types:
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
    root = _resolve_project_path(processed_root)
    collected = []

    for site in sites:
        selected_source_types = source_types or _discover_source_types(root, site)
        for source_type in selected_source_types:
            folder = root / site / source_type
            if not folder.exists():
                logging.warning("Dossier absent: %s", folder)
                continue

            files = _list_supported_documents(folder)
            collected.extend(files)
            md_count = sum(1 for file in files if file.suffix.lower() == ".md")
            txt_count = sum(1 for file in files if file.suffix.lower() == ".txt")
            logging.info(
                "%s: %s fichiers support RAG (md=%s, txt=%s)",
                folder,
                len(files),
                md_count,
                txt_count,
            )

    return collected


def _discover_source_types(root: Path, site: str) -> tuple[str, ...]:
    site_dir = root / site
    if not site_dir.exists():
        return ()
    return tuple(sorted(path.name for path in site_dir.iterdir() if path.is_dir()))


def load_documents(file_paths, processed_root=DEFAULT_PROCESSED_ROOT):
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
            site = parts[0] if len(parts) > 0 else "unknown"
            source_type = parts[1] if len(parts) > 1 else "unknown"
            faq_scope = parts[2] if len(parts) > 2 else "unknown"
            topic = "/".join(parts[3:-1]) if len(parts) > 4 else ""
            language = TARGET_LANGUAGE
        except Exception:
            relative = file_path
            site = "unknown"
            source_type = "unknown"
            faq_scope = "unknown"
            topic = ""
            language = TARGET_LANGUAGE

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path.as_posix()),
                    "relative_source": str(relative).replace("\\", "/"),
                    "org": site if site in ("cnra", "rcar") else "both",
                    "type": "doc",
                    "site": site,
                    "source_type": source_type,
                    "faq_scope": faq_scope,
                    "faq_topic": topic,
                    "language": language,
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix.lower(),
                },
            )
        )

    return documents


def build_chunker(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
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

    per_source_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("relative_source", chunk.metadata.get("source", "unknown"))
        current_idx = per_source_index.get(source, 0)
        chunk.metadata["chunk_index"] = current_idx
        per_source_index[source] = current_idx + 1

    return chunks


def _chunk_to_vespa_document(chunk: Document, embedding: list[float]) -> dict:
    metadata = chunk.metadata
    doc_id = stable_data_id(
        "doc",
        metadata.get("relative_source", metadata.get("source", "unknown")),
        metadata.get("chunk_index", 0),
    )
    return {
        "id": doc_id,
        "fields": {
            "doc_id": doc_id,
            "text": chunk.page_content,
            "org": metadata.get("org", "both"),
            "source_type": metadata.get("source_type", "faq"),
            "faq_scope": metadata.get("faq_scope", "unknown"),
            "relative_source": metadata.get("relative_source", metadata.get("source", "")),
            "chunk_index": int(metadata.get("chunk_index", 0)),
            "embedding": embedding,
        },
    }


def index_data(
    processed_root=DEFAULT_PROCESSED_ROOT,
    vespa_url=VESPA_URL,
    vespa_port=VESPA_PORT,
    content_cluster=VESPA_CONTENT_CLUSTER,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    reset: bool = True,
):
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
        raise ValueError("Aucun fichier (.md/.txt) trouve pour RCAR/CNRA dans data/supportstagerag.")

    documents = load_documents(file_paths, processed_root=processed_root)
    if not documents:
        raise ValueError("Aucun document exploitable trouve apres lecture des fichiers FAQ.")

    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Aucun chunk genere. Verifiez les contenus FAQ md/txt.")

    device = resolve_embedding_device()
    logging.info("Embedding device utilise pour l'indexation: %s", device)
    embeddings = make_huggingface_embeddings(
        model_name=EMBEDDING_MODEL,
        device=device,
        encode_kwargs={"normalize_embeddings": True},
    )

    app = make_vespa_app(vespa_url, vespa_port)
    if reset:
        logging.info("Suppression des documents Vespa schema=doc avant reindexation...")
        delete_all_docs(app, "doc", content_cluster)

    logging.info("Calcul embeddings pour %s chunks support RAG...", len(chunks))
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    vespa_documents = [
        _chunk_to_vespa_document(chunk, vector)
        for chunk, vector in zip(chunks, vectors)
    ]

    logging.info("Feed Vespa schema=doc: %s documents...", len(vespa_documents))
    fed = feed_documents(app, schema="doc", documents=vespa_documents)

    logging.info("Vespa endpoint: %s:%s", vespa_url, vespa_port)
    logging.info("Documents charges: %s", len(documents))
    logging.info("Chunks indexes: %s", fed)
    logging.info("Chunk size=%s | overlap=%s", chunk_size, chunk_overlap)

    return {
        "vespa_url": vespa_url,
        "vespa_port": vespa_port,
        "schema": "doc",
        "documents": len(documents),
        "chunks": fed,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


if __name__ == "__main__":
    result = index_data()
    logging.info("Indexation terminee: %s", result)
