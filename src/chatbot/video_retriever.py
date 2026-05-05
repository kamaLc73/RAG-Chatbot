"""
src/chatbot/video_retriever.py
================================
Récupère les vidéos YouTube pertinentes pour une requête.

Pipeline:
    1. Recherche vectorielle dans chroma_db_youtube (top-K candidats)
    2. Reranking cross-encoder (BAAI/bge-reranker-v2-m3, multilingue)
    3. Déduplication par video_id (garder le meilleur score par vidéo)
    4. Filtrage par seuil de pertinence
    5. Retourner top-N vidéos avec métadonnées

Le VideoRetriever est optionnel : si le vectorstore n'existe pas,
il retourne silencieusement une liste vide sans bloquer le RAG principal.
"""

import os
import sys
from pathlib import Path

from loguru import logger

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from config.logger import is_logger_initialized, setup_logger
    from config.settings import BASE_DIR, LOGS_DIR
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[2]
    LOGS_DIR = BASE_DIR / "logs"
    is_logger_initialized = lambda: False
    setup_logger = None

YOUTUBE_VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma_db_youtube"
COLLECTION_NAME = "youtube_videos"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Reranker multilingue — cohérent avec le pipeline du projet principal (shipping)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Paramètres de retrieval
TOP_K_RETRIEVE = 10      # Candidats récupérés avant reranking
TOP_K_FINAL = 2          # Vidéos retournées après reranking
RELEVANCE_THRESHOLD = 0.1  # Seuil minimum de score reranker (à ajuster)


class VideoRetriever:
    """
    Retriever de vidéos YouTube avec reranking cross-encoder.

    Conçu pour être composable avec RAGPipeline :
        retriever = VideoRetriever()
        videos = retriever.retrieve("comment calculer ma retraite")

    Chaque vidéo retournée est un dict:
        {
            "video_id": "abc123",
            "title": "Titre de la vidéo",
            "url": "https://www.youtube.com/watch?v=abc123",
            "thumbnail_url": "https://img.youtube.com/vi/abc123/mqdefault.jpg",
            "score": 0.82,
            "excerpt": "Début du passage pertinent...",
        }
    """

    def __init__(
        self,
        vectorstore_dir: Path = YOUTUBE_VECTORSTORE_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_final: int = TOP_K_FINAL,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ):
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final
        self.relevance_threshold = relevance_threshold
        self._available = False
        self.vectorstore = None
        self.reranker = None

        if setup_logger is not None and not is_logger_initialized():
            setup_logger(log_dir=LOGS_DIR, source="youtube_fetch")

        # Vérification silencieuse : pas de crash si pas encore indexé
        if not vectorstore_dir.exists():
            logger.info(
                "VideoRetriever: vectorstore absent ({})."
                " Lance index_videos.py pour activer les suggestions vidéo.",
                vectorstore_dir,
            )
            return

        try:
            self._load_vectorstore(vectorstore_dir, collection_name, embedding_model)
        except Exception as exc:
            logger.warning("VideoRetriever: impossible de charger le vectorstore: {}", exc)
            return

        try:
            self._load_reranker(reranker_model)
        except Exception as exc:
            logger.warning(
                "VideoRetriever: reranker indisponible ({}). Utilisation scores vectoriels uniquement.", exc
            )

        self._available = True
        logger.info(
            "VideoRetriever prêt (reranker={}, k_retrieve={}, k_final={}, seuil={})",
            self.reranker is not None,
            self.top_k_retrieve,
            self.top_k_final,
            self.relevance_threshold,
        )

    # ─── Chargement ───────────────────────────────────────────────────────────

    def _load_vectorstore(self, directory: Path, collection_name: str, embedding_model: str) -> None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        hf_token = os.getenv("HF_TOKEN")
        model_kwargs: dict = {"device": device}
        if hf_token:
            model_kwargs["token"] = hf_token

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            persist_directory=str(directory),
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        count = self.vectorstore._collection.count()
        logger.info("YouTube vectorstore: {} chunks", count)

    def _load_reranker(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(
            model_name,
            max_length=512,
            # Pas de device explicite : CrossEncoder auto-détecte CUDA
        )
        logger.info("Reranker chargé: {}", model_name)

    # ─── API publique ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    def retrieve(self, query: str) -> list[dict]:
        """
        Retourne les vidéos pertinentes pour une requête.
        Retourne [] si le retriever n'est pas disponible ou si aucun résultat pertinent.
        """
        if not self._available or self.vectorstore is None:
            return []

        query = (query or "").strip()
        if not query:
            return []

        # 1. Recherche vectorielle
        try:
            docs = self.vectorstore.similarity_search(query, k=self.top_k_retrieve)
            logger.debug("--- CHUNKS RECUPÉRÉS AVANT RERANKING ---")
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", "Unknown")
                vid_id = doc.metadata.get("video_id", "")
                logger.debug("  [{}] Vidéo: '{}' (ID: {})", i+1, title, vid_id)
        except Exception as exc:
            logger.error("Erreur recherche YouTube: {}", exc)
            return []

        if not docs:
            return []

        # 2. Reranking cross-encoder
        if self.reranker is not None:
            pairs = [(query, doc.page_content[:400]) for doc in docs]
            try:
                scores = self.reranker.predict(pairs).tolist()
                logger.debug("--- SCORES APRÈS RERANKING ---")
                for i, (doc, score) in enumerate(zip(docs, scores)):
                     logger.debug("  [{}] Score: {:.4f} | Vidéo: '{}'", i+1, score, doc.metadata.get("title", ""))
            except Exception as exc:
                logger.warning("Reranker erreur: {} — fallback scores vectoriels", exc)
                scores = [0.5] * len(docs)
        else:
            scores = [0.5] * len(docs)

        # 3. Déduplication par video_id (garder le meilleur score)
        best_per_video: dict[str, dict] = {}
        for doc, score in zip(docs, scores):
            vid_id = doc.metadata.get("video_id", "")
            if not vid_id:
                continue
            if vid_id not in best_per_video or score > best_per_video[vid_id]["score"]:
                # Extrait : 200 premiers caractères du contenu pertinent
                content_preview = doc.page_content
                # Enlever le "Titre : xxx\n\n" du début pour l'excerpt
                if "\n\n" in content_preview:
                    content_preview = content_preview.split("\n\n", 1)[-1]
                excerpt = content_preview[:200].strip()
                if len(content_preview) > 200:
                    excerpt += "..."

                best_per_video[vid_id] = {
                    "video_id": vid_id,
                    "title": doc.metadata.get("title", "Vidéo CNRA/RCAR"),
                    "url": doc.metadata.get("url", ""),
                    "thumbnail_url": doc.metadata.get(
                        "thumbnail_url",
                        f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg",
                    ),
                    "score": float(score),
                    "excerpt": excerpt,
                    "upload_date": doc.metadata.get("upload_date", ""),
                }

        # 4. Filtrage par seuil + tri
        filtered = [
            v for v in best_per_video.values()
            if v["score"] >= self.relevance_threshold
        ]
        filtered.sort(key=lambda x: x["score"], reverse=True)

        results = filtered[: self.top_k_final]

        if results:
            logger.info(
                "Vidéos pertinentes: {} (meilleur score: {:.3f})",
                len(results), results[0]["score"]
            )
        else:
            logger.debug("Aucune vidéo au-dessus du seuil {}", self.relevance_threshold)

        return results
