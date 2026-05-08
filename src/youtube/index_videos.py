"""
src/youtube/index_videos.py
============================
Indexe les transcripts YouTube dans une collection ChromaDB dédiée.
Utilise le même modèle d'embedding que le RAG principal (BAAI/bge-m3)
pour garantir la cohérence sémantique lors du reranking croisé.

Usage:
    python src/youtube/index_videos.py
    python src/youtube/index_videos.py --reset          # force réindexation
    python src/youtube/index_videos.py --min-chars 100  # seuil transcripts courts

Prérequis:
    Lancer fetch_transcripts.py d'abord pour générer data/youtube/transcripts.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from config.settings import BASE_DIR, LOGS_DIR
    from config.logger import setup_logger
    setup_logger(LOGS_DIR, source="index_youtube")
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[2]
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOGS_DIR / "index_youtube.log", level="INFO")
    
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")

TRANSCRIPTS_FILE = BASE_DIR / "data" / "youtube" / "transcripts.json"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma_db_unified"
COLLECTION_NAME = "rcar_cnra_unified"
EMBEDDING_MODEL = "BAAI/bge-m3"

MIN_TRANSCRIPT_CHARS = 80  # Ignorer les transcripts trop courts (erreurs de sous-titres)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_transcripts(path: Path = TRANSCRIPTS_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {path}\n"
            "Lance d'abord: python src/youtube/fetch_transcripts.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Chargé {} enregistrements depuis {}", len(data), path)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Création des documents LangChain
# ─────────────────────────────────────────────────────────────────────────────

def build_documents(videos: list[dict], min_chars: int = MIN_TRANSCRIPT_CHARS) -> list:
    """
    Convertit les enregistrements vidéo en Documents LangChain.
    Le contenu = titre + transcript, les métadonnées gardent le lien YouTube.
    """
    from langchain_core.documents import Document

    docs = []
    skipped = 0

    for video in videos:
        transcript = (video.get("transcript") or "").strip()
        if len(transcript) < min_chars:
            logger.debug(
                "Skip (transcript trop court: {} chars): {}",
                len(transcript), video.get("title", "?")[:60]
            )
            skipped += 1
            continue

        title = video.get("title", "Vidéo CNRA/RCAR")

        # Contenu indexé : titre + transcript
        # Le titre est répété au début pour améliorer la pertinence sémantique
        content = f"Titre : {title}\n\n{transcript}"

        # Détecter l'organisme depuis le titre/description
        text_lower = (title + " " + video.get("description", "")).lower()
        if "cnra" in text_lower and "rcar" not in text_lower:
            org = "cnra"
        elif "rcar" in text_lower and "cnra" not in text_lower:
            org = "rcar"
        else:
            org = "both"

        docs.append(Document(
            page_content=content,
            metadata={
                "source": "youtube",
                "type": "video",
                "org": org,
                "video_id": video["video_id"],
                "title": title,
                "url": video["url"],
                "upload_date": video.get("upload_date", ""),
                "duration": video.get("duration"),
                "description": video.get("description", ""),
                "transcript_language": video.get("transcript_language", "unknown"),
                "transcript_translated": video.get("transcript_translated", False),
                "transcript_original_lang": video.get("transcript_original_lang", ""),
                # Thumbnail générée directement depuis video_id (pas besoin d'API)
                "thumbnail_url": f"https://img.youtube.com/vi/{video['video_id']}/mqdefault.jpg",
            },
        ))

    logger.info(
        "Documents créés: {} (ignorés: {} sans transcript suffisant)",
        len(docs), skipped
    )
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list) -> list:
    """
    Conserve la vidéo entière comme un unique chunk.
    Aucun découpage et aucun overlap n'est appliqué car les vidéos ne sont pas liées.
    """
    for doc in documents:
        doc.metadata["chunk_index"] = 0

    logger.info("Vidéos conservées complètes: 1 chunk par vidéo. Total: {}", len(documents))
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Indexation ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def index_videos(
    transcripts_file: Path = TRANSCRIPTS_FILE,
    vectorstore_dir: Path = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    min_transcript_chars: int = MIN_TRANSCRIPT_CHARS,
    reset: bool = True,
) -> dict:
    """
    Pipeline complet : transcripts → documents entiers → ChromaDB.
    Retourne un résumé de l'indexation.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # Chargement
    videos = load_transcripts(transcripts_file)
    documents = build_documents(videos, min_chars=min_transcript_chars)

    if not documents:
        raise ValueError(
            "Aucun document valide à indexer. "
            "Vérifiez que fetch_transcripts.py a récupéré des transcripts."
        )

    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("Aucun chunk généré.")

    # Reset si demandé
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    # Embedding device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    logger.info("Device embedding: {}", device)

    hf_token = os.getenv("HF_TOKEN")
    model_kwargs: dict = {"device": device}
    if hf_token:
        model_kwargs["token"] = hf_token

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    # Supprimer uniquement les chunks type="video" dans la collection unifiée
    vectorstore = Chroma(
        persist_directory=str(vectorstore_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    try:
        vectorstore._collection.delete(where={"type": "video"})
        logger.info("Chunks type=video supprimés de la collection unifiée avant réindexation")
    except Exception as exc:
        logger.warning("Impossible de supprimer les chunks video existants: {}", exc)

    logger.info("Indexation de {} chunks en cours...", len(chunks))

    # Insertion par batch pour les grandes collections
    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        logger.info("Batch {}/{} indexé", min(i + BATCH_SIZE, len(chunks)), len(chunks))

    summary = {
        "vectorstore_dir": str(vectorstore_dir),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "videos_total": len(videos),
        "videos_indexed": len(documents),
        "chunks": len(chunks),
    }

    logger.info("Indexation terminée: {}", summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Indexe les transcripts YouTube dans ChromaDB (1 vidéo = 1 chunk complet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python src/youtube/index_videos.py
  python src/youtube/index_videos.py --reset
        """,
    )
    parser.add_argument(
        "--transcripts-file", default=str(TRANSCRIPTS_FILE),
        help="Chemin vers transcripts.json"
    )
    parser.add_argument(
        "--vectorstore-dir", default=str(VECTORSTORE_DIR),
        help="Dossier ChromaDB de sortie"
    )
    parser.add_argument(
        "--collection-name", default=COLLECTION_NAME,
        help="Nom de la collection ChromaDB"
    )
    parser.add_argument(
        "--embedding-model", default=EMBEDDING_MODEL,
        help="Modèle HuggingFace pour les embeddings"
    )
    parser.add_argument(
        "--min-chars", type=int, default=MIN_TRANSCRIPT_CHARS,
        help="Longueur minimale d'un transcript pour être indexé"
    )
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Ne pas supprimer la collection existante"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = index_videos(
        transcripts_file=Path(args.transcripts_file),
        vectorstore_dir=Path(args.vectorstore_dir),
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        min_transcript_chars=args.min_chars,
        reset=not args.no_reset,
    )
    print("\nIndexation terminée:")
    for k, v in result.items():
        print(f"  {k}: {v}")