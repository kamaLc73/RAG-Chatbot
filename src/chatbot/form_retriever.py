"""
src/chatbot/form_retriever.py
==============================
Récupère les formulaires CNRA/RCAR pertinents pour une requête.

Pipeline identique au VideoRetriever :
    1. Recherche vectorielle dans chroma_db_forms (top-K candidats)
    2. Reranking cross-encoder (BAAI/bge-reranker-v2-m3)
    3. Déduplication par form_id
    4. Filtrage par seuil de pertinence
    5. Retourner top-N formulaires avec métadonnées

Chaque formulaire retourné est un dict :
    {
        "form_id":    "rcar_pension_retraite",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "title":      "Formulaire pension de retraite",
        "description": "...",
        "page_url":   "https://www.rcar.ma/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    "https://www.rcar.ma/uploads/.../DEMANDE_ALLOCATION_RETRAITE.pdf",
        "score":      0.73,
    }
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

FORMS_VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma_db_forms"
COLLECTION_NAME       = "forms"
EMBEDDING_MODEL       = "BAAI/bge-m3"
RERANKER_MODEL        = "BAAI/bge-reranker-v2-m3"

TOP_K_RETRIEVE      = 10    # Candidats vectoriels avant reranking
TOP_K_FINAL         = 1    # Formulaires retournés max
RELEVANCE_THRESHOLD = 0.2 # Score minimum reranker (plus strict que vidéos car formulaires très ciblés)


class FormRetriever:
    """
    Retriever de formulaires CNRA/RCAR avec reranking cross-encoder.

    Composable avec RAGPipeline :
        retriever = FormRetriever()
        forms = retriever.retrieve("comment demander une pension de retraite RCAR")

    Silencieux si le vectorstore n'existe pas (retourne [] sans bloquer).
    """

    def __init__(
        self,
        vectorstore_dir:    Path  = FORMS_VECTORSTORE_DIR,
        collection_name:    str   = COLLECTION_NAME,
        embedding_model:    str   = EMBEDDING_MODEL,
        reranker_model:     str   = RERANKER_MODEL,
        top_k_retrieve:     int   = TOP_K_RETRIEVE,
        top_k_final:        int   = TOP_K_FINAL,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ):
        self.top_k_retrieve      = top_k_retrieve
        self.top_k_final         = top_k_final
        self.relevance_threshold = relevance_threshold
        self._available          = False
        self.vectorstore         = None
        self.reranker            = None

        if setup_logger is not None and not is_logger_initialized():
            setup_logger(log_dir=LOGS_DIR, source="forms")

        if not vectorstore_dir.exists():
            logger.info(
                "FormRetriever: vectorstore absent ({})."
                " Lance index_forms.py pour activer les suggestions de formulaires.",
                vectorstore_dir,
            )
            return

        try:
            self._load_vectorstore(vectorstore_dir, collection_name, embedding_model)
        except Exception as exc:
            logger.warning("FormRetriever: impossible de charger le vectorstore: {}", exc)
            return

        try:
            self._load_reranker(reranker_model)
        except Exception as exc:
            logger.warning("FormRetriever: reranker indisponible ({}). Scores vectoriels uniquement.", exc)

        self._available = True
        logger.info(
            "FormRetriever prêt (reranker={}, k_retrieve={}, k_final={}, seuil={})",
            self.reranker is not None, self.top_k_retrieve,
            self.top_k_final, self.relevance_threshold,
        )

    # ── Chargement ─────────────────────────────────────────────────────────────

    def _load_vectorstore(self, directory: Path, collection_name: str, embedding_model: str) -> None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        hf_token = os.getenv("HF_TOKEN", "").strip()
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
        logger.info("Forms vectorstore: {} formulaires indexés", count)

    def _load_reranker(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(model_name, max_length=512)
        logger.info("Reranker chargé: {}", model_name)

    # ── API publique ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    def retrieve(self, query: str) -> list[dict]:
        """
        Retourne les formulaires pertinents pour une requête.
        Retourne [] si non disponible ou aucun résultat pertinent.
        """
        if not self._available or self.vectorstore is None:
            return []

        query = (query or "").strip()
        if not query:
            return []

        # 1. Recherche vectorielle
        try:
            docs = self.vectorstore.similarity_search(query, k=self.top_k_retrieve)
        except Exception as exc:
            logger.error("Erreur recherche formulaires: {}", exc)
            return []

        if not docs:
            return []

        # 2. Reranking
        if self.reranker is not None:
            pairs = [(query, doc.page_content[:600]) for doc in docs]
            try:
                scores = self.reranker.predict(pairs).tolist()
            except Exception as exc:
                logger.warning("Reranker erreur: {} — fallback scores vectoriels", exc)
                scores = [0.5] * len(docs)
        else:
            scores = [0.5] * len(docs)

        # 3. Déduplication par form_id (garder le meilleur score)
        best_per_form: dict[str, dict] = {}
        for doc, score in zip(docs, scores):
            fid = doc.metadata.get("form_id", "")
            if not fid:
                continue
            if fid not in best_per_form or score > best_per_form[fid]["score"]:
                best_per_form[fid] = {
                    "form_id":    fid,
                    "org":        doc.metadata.get("org", ""),
                    "category":   doc.metadata.get("category", ""),
                    "category_slug": doc.metadata.get("category_slug", ""),
                    "title":      doc.metadata.get("title", "Formulaire"),
                    "description": doc.metadata.get("description", ""),
                    "page_url":   doc.metadata.get("page_url", ""),
                    "pdf_url":    doc.metadata.get("pdf_url", ""),
                    "filename":   doc.metadata.get("filename", ""),
                    "has_pdf_text": doc.metadata.get("has_pdf_text", False),
                    "score":      float(score),
                }

        # 4. Filtrage par seuil + tri
        filtered = [
            v for v in best_per_form.values()
            if v["score"] >= self.relevance_threshold
        ]
        filtered.sort(key=lambda x: x["score"], reverse=True)
        results = filtered[: self.top_k_final]

        if results:
            logger.info(
                "Formulaires pertinents: {} (meilleur score: {:.3f})",
                len(results), results[0]["score"]
            )
        else:
            logger.debug("Aucun formulaire au-dessus du seuil {}", self.relevance_threshold)

        return results
