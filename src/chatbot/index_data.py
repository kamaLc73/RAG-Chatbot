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
TARGET_SOURCE_TYPES = ("faq",)
TARGET_LANGUAGE = "fr"
SUPPORTED_EXTENSIONS = (".md", ".txt")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "supportstagerag"
DEFAULT_VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore" / "chroma_db"

DEFAULT_CHUNK_SIZE = 1200 # Nombre de caractères par chunk
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
    """Collect FAQ documents recursively for RCAR and CNRA."""
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
            md_count = sum(1 for file in files if file.suffix.lower() == ".md")
            txt_count = sum(1 for file in files if file.suffix.lower() == ".txt")
            logging.info(
                "%s: %s fichiers FAQ (md=%s, txt=%s)",
                folder,
                len(files),
                md_count,
                txt_count,
            )

    return collected


def load_documents(file_paths, processed_root=DEFAULT_PROCESSED_ROOT):
    """Load FAQ files into LangChain Document objects with metadata."""
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

    # Add stable chunk index per source file.
    per_source_index = {}
    for chunk in chunks:
        source = chunk.metadata.get("relative_source", chunk.metadata.get("source", "unknown"))
        current_idx = per_source_index.get(source, 0)
        chunk.metadata["chunk_index"] = current_idx
        per_source_index[source] = current_idx + 1

    return chunks


def index_data(
    processed_root=DEFAULT_PROCESSED_ROOT,
    persist_directory=DEFAULT_VECTORSTORE_DIR,
    collection_name="rcar_cnra_fr",
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    reset_collection=True,
):
    """
    Build a ChromaDB vector store from RCAR/CNRA FAQ files stored in supportstagerag.
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

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError("Aucun chunk genere. Verifiez les contenus FAQ md/txt.")

    persist_path = _resolve_project_path(persist_directory)
    if reset_collection and persist_path.exists():
        shutil.rmtree(persist_path)

    persist_path.mkdir(parents=True, exist_ok=True)

    device = _resolve_embedding_device()
    hf_token = os.getenv("HF_TOKEN")
    logging.info("Embedding device utilise pour l'indexation: %s", device)
    if not hf_token:
        logging.warning(
            "HF_TOKEN non configure: telechargement Hugging Face en mode non authentifie "
            "(limites plus strictes)."
        )

    model_kwargs = {"device": device}
    if hf_token:
        model_kwargs["token"] = hf_token

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=collection_name,
    )

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    logging.info("ChromaDB sauvegarde dans %s", persist_path)
    logging.info("Collection: %s", collection_name)
    logging.info("Documents charges: %s", len(documents))
    logging.info("Chunks indexes: %s", len(chunks))
    logging.info("Chunk size=%s | overlap=%s", chunk_size, chunk_overlap)

    return {
        "persist_directory": str(persist_path),
        "collection_name": collection_name,
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


if __name__ == "__main__":
    result = index_data()
    logging.info("Indexation terminee: %s", result)