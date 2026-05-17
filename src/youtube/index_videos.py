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
    from config.settings import BASE_DIR, LOGS_DIR
    from config.logger import setup_logger
    from dotenv import load_dotenv

    setup_logger(LOGS_DIR, source="index_youtube")
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[2]
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOGS_DIR / "index_youtube.log", level="INFO")

    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")

from store.vespa_store import delete_all_docs, feed_documents, make_vespa_app, stable_data_id

logging.getLogger("vespa").setLevel(logging.WARNING)
logging.getLogger("vespa.application").setLevel(logging.WARNING)

TRANSCRIPTS_FILE = BASE_DIR / "data" / "youtube" / "transcripts.json"
VESPA_URL = os.getenv("VESPA_URL", "http://localhost")
VESPA_PORT = int(os.getenv("VESPA_PORT", "8080"))
VESPA_CONTENT_CLUSTER = os.getenv("VESPA_CONTENT_CLUSTER", "rcar_cnra")
EMBEDDING_MODEL = "BAAI/bge-m3"

MIN_TRANSCRIPT_CHARS = 80


def load_transcripts(path: Path = TRANSCRIPTS_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {path}\n"
            "Lance d'abord: python src/youtube/fetch_transcripts.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Charge {} enregistrements depuis {}", len(data), path)
    return data


def build_documents(videos: list[dict], min_chars: int = MIN_TRANSCRIPT_CHARS) -> list:
    from langchain_core.documents import Document

    docs = []
    skipped = 0

    for video in videos:
        transcript = (video.get("transcript") or "").strip()
        if len(transcript) < min_chars:
            logger.debug(
                "Skip (transcript trop court: {} chars): {}",
                len(transcript),
                video.get("title", "?")[:60],
            )
            skipped += 1
            continue

        title = video.get("title", "Video CNRA/RCAR")
        content = f"Titre : {title}\n\n{transcript}"

        text_lower = (title + " " + video.get("description", "")).lower()
        if "cnra" in text_lower and "rcar" not in text_lower:
            org = "cnra"
        elif "rcar" in text_lower and "cnra" not in text_lower:
            org = "rcar"
        else:
            org = "both"

        docs.append(
            Document(
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
                    "thumbnail_url": f"https://img.youtube.com/vi/{video['video_id']}/mqdefault.jpg",
                },
            )
        )

    logger.info("Documents crees: {} (ignores: {} sans transcript suffisant)", len(docs), skipped)
    return docs


def chunk_documents(documents: list) -> list:
    for doc in documents:
        doc.metadata["chunk_index"] = 0
    logger.info("Videos conservees completes: 1 document par video. Total: {}", len(documents))
    return documents


def _resolve_embedding_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def index_videos(
    transcripts_file: Path = TRANSCRIPTS_FILE,
    vespa_url: str = VESPA_URL,
    vespa_port: int = VESPA_PORT,
    content_cluster: str = VESPA_CONTENT_CLUSTER,
    embedding_model: str = EMBEDDING_MODEL,
    min_transcript_chars: int = MIN_TRANSCRIPT_CHARS,
    reset: bool = True,
) -> dict:
    from langchain_huggingface import HuggingFaceEmbeddings

    videos = load_transcripts(transcripts_file)
    documents = build_documents(videos, min_chars=min_transcript_chars)

    if not documents:
        raise ValueError(
            "Aucun document valide a indexer. "
            "Verifiez que fetch_transcripts.py a recupere des transcripts."
        )

    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("Aucun document genere.")

    device = _resolve_embedding_device()
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

    app = make_vespa_app(vespa_url, vespa_port)
    if reset:
        logger.info("Suppression des documents Vespa schema=video avant reindexation")
        delete_all_docs(app, "video", content_cluster)

    logger.info("Calcul embeddings pour {} videos...", len(chunks))
    vectors = embeddings.embed_documents([doc.page_content for doc in chunks])
    vespa_documents = []
    for doc, vector in zip(chunks, vectors):
        meta = doc.metadata
        raw_video_id = meta.get("video_id", "")
        vespa_doc_id = stable_data_id("video", raw_video_id)
        vespa_documents.append(
            {
                "id": vespa_doc_id,
                "fields": {
                    "video_id": raw_video_id,
                    "title": meta.get("title", ""),
                    "transcript": doc.page_content,
                    "org": meta.get("org", "both"),
                    "url": meta.get("url", ""),
                    "thumbnail_url": meta.get("thumbnail_url", ""),
                    "upload_date": str(meta.get("upload_date", "")),
                    "embedding": vector,
                },
            }
        )

    logger.info("Feed Vespa schema=video: {} documents...", len(vespa_documents))
    fed = feed_documents(app, schema="video", documents=vespa_documents)

    summary = {
        "vespa_url": vespa_url,
        "vespa_port": vespa_port,
        "schema": "video",
        "embedding_model": embedding_model,
        "videos_total": len(videos),
        "videos_indexed": len(documents),
        "chunks": fed,
    }

    logger.info("Indexation terminee: {}", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Indexe les transcripts YouTube dans Vespa (1 video = 1 document)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python src/youtube/index_videos.py
  python src/youtube/index_videos.py --no-reset
        """,
    )
    parser.add_argument("--transcripts-file", default=str(TRANSCRIPTS_FILE), help="Chemin vers transcripts.json")
    parser.add_argument("--vespa-url", default=VESPA_URL, help="URL Vespa")
    parser.add_argument("--vespa-port", type=int, default=VESPA_PORT, help="Port Vespa")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Modele HuggingFace pour les embeddings")
    parser.add_argument("--min-chars", type=int, default=MIN_TRANSCRIPT_CHARS, help="Longueur minimale d'un transcript")
    parser.add_argument("--no-reset", action="store_true", help="Ne pas supprimer les documents video existants")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = index_videos(
        transcripts_file=Path(args.transcripts_file),
        vespa_url=args.vespa_url,
        vespa_port=args.vespa_port,
        embedding_model=args.embedding_model,
        min_transcript_chars=args.min_chars,
        reset=not args.no_reset,
    )
    print("\nIndexation terminee:")
    for k, v in result.items():
        print(f"  {k}: {v}")
