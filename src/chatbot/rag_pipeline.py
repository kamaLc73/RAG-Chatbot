"""
src/chatbot/rag_pipeline.py
============================
Pipeline RAG principal + reranking sur chroma_db + suggestions vidéos + formulaires.

Flux de retrieval documentaire (NOUVEAU) :
    1. similarity_search(k=RAG_RETRIEVE_K)   → 12 candidats vectoriels
    2. shared_reranker.predict()              → score cross-encoder sur chaque candidat
    3. tri décroissant → top RAG_FINAL_K=4   → chunks envoyés au LLM

Champs retournés par query() :
    {
        "response":       str,
        "context_docs":   int,        # nombre de chunks après reranking
        "videos":         list[dict],
        "forms":          list[dict],
        "original_query": str,
        "cached":         bool,
    }
"""

import os
import sys
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from os import getenv

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message=r"Accessing `__path__` from `\.models\..*",)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
except ImportError:
    torch = None

from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from loguru import logger
from sentence_transformers import CrossEncoder

try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from config.logger import setup_logger
from config.settings import BASE_DIR, LOGS_DIR

try:
    from chatbot.video_retriever import VideoRetriever
    _VIDEO_RETRIEVER_AVAILABLE = True
except ImportError:
    _VIDEO_RETRIEVER_AVAILABLE = False
    logger.warning("VideoRetriever non disponible.")

try:
    from chatbot.form_retriever import FormRetriever
    _FORM_RETRIEVER_AVAILABLE = True
except ImportError:
    _FORM_RETRIEVER_AVAILABLE = False
    logger.warning("FormRetriever non disponible.")

load_dotenv(BASE_DIR / ".env")

VECTORSTORE_RELATIVE_PATH = Path("data") / "vectorstore" / "chroma_db"
DEFAULT_EMBEDDING_MODEL   = "BAAI/bge-m3"
RERANKER_MODEL            = "BAAI/bge-reranker-v2-m3"
HF_TOKEN = getenv("HF_TOKEN", "").strip().strip('"\'')

OLLAMA_MODEL       = getenv("OLLAMA_MODEL", "mistral:latest").strip().strip('"\'')
OLLAMA_BASE_URL    = getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY     = getenv("OLLAMA_API_KEY", "").strip().strip('"\'')
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 600
OLLAMA_NUM_CTX     = 4096
OLLAMA_KEEP_ALIVE  = "30m"

# ── Paramètres retrieval principal ────────────────────────────────────────────
# RAG_RETRIEVE_K : candidats récupérés par similarity_search (pool pour le reranker)
# RAG_FINAL_K    : chunks conservés après reranking → envoyés au LLM
# Règle : RAG_RETRIEVE_K >= RAG_FINAL_K (typiquement 3x)
RAG_RETRIEVE_K        = 12    # Était RAG_TOP_K = 4 — élargi pour donner du choix au reranker
RAG_FINAL_K           = 4     # Chunks finaux après reranking
RAG_MAX_CONTEXT_CHARS = 2800
RAG_MAX_DOC_CHARS     = 700
RAG_CACHE_SIZE        = 100


class RAGPipeline:
    def __init__(
        self,
        local_model: str = OLLAMA_MODEL,
        vectorstore_path: Path | str = VECTORSTORE_RELATIVE_PATH,
        collection_name: str = "rcar_cnra_fr",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_video_suggestions: bool = True,
        enable_form_suggestions:  bool = True,
    ):
        resolved_path = Path(vectorstore_path)
        if not resolved_path.is_absolute():
            resolved_path = (BASE_DIR / resolved_path).resolve()

        self.collection_name   = collection_name
        self.retrieval_k       = RAG_RETRIEVE_K
        self.final_k           = RAG_FINAL_K
        self.max_context_chars = RAG_MAX_CONTEXT_CHARS
        self.max_doc_chars     = RAG_MAX_DOC_CHARS
        self.response_cache: OrderedDict = OrderedDict()

        logger.info("VectorStore: {} (collection={})", resolved_path, collection_name)

        # ── Device ────────────────────────────────────────────────────────────
        try:
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        except Exception:
            device = "cpu"

        # ── Embeddings (instance unique partagée) ─────────────────────────────
        logger.info("Embedding: {} sur {}", embedding_model, device)
        model_kwargs: dict = {"device": device}
        if HF_TOKEN:
            model_kwargs["token"] = HF_TOKEN

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

        # FP16 sur GPU : −50% VRAM, ~+20% throughput
        try:
            if device == "cuda" and hasattr(self.embeddings, "client"):
                if hasattr(self.embeddings.client, "model"):
                    self.embeddings.client.model.half()
                    logger.info("Embeddings castées en FP16 (GPU)")
        except Exception as exc:
            logger.debug("Embeddings FP16 cast échoué (fallback FP32): {}", exc)

        self._warm_up_embeddings()

        # ── Reranker (instance unique partagée) ───────────────────────────────
        self.shared_reranker: CrossEncoder | None = None
        try:
            self.shared_reranker = CrossEncoder(
                RERANKER_MODEL,
                max_length=512,
                device=device,
            )
            if device == "cuda" and self.shared_reranker.model is not None:
                try:
                    self.shared_reranker.model.half()
                    logger.info("Reranker casté en FP16 (GPU)")
                except Exception as exc:
                    logger.debug("Reranker FP16 cast échoué: {}", exc)
            logger.info("Reranker chargé: {} sur {}", RERANKER_MODEL, device)
        except Exception as exc:
            logger.warning("Reranker non disponible ({}), fallback scores vectoriels", exc)

        # ── VectorStore principal ─────────────────────────────────────────────
        self.vectorstore = Chroma(
            persist_directory=str(resolved_path),
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        self.collection_count = self._get_collection_count(resolved_path, collection_name)
        logger.info("Collection '{}': {} chunks", collection_name, self.collection_count)
        if self.collection_count == 0:
            logger.warning("Collection vide — lancez index_data.py pour l'indexation.")

        # ── VideoRetriever (instance partagée) ────────────────────────────────
        self.video_retriever = None
        if enable_video_suggestions and _VIDEO_RETRIEVER_AVAILABLE:
            try:
                self.video_retriever = VideoRetriever(
                    shared_embeddings=self.embeddings,
                    shared_reranker=self.shared_reranker,
                )
                logger.info(
                    "Suggestions vidéo : {}",
                    "activées" if self.video_retriever.available else "désactivées (lance index_videos.py)",
                )
            except Exception as exc:
                logger.warning("VideoRetriever init échoué: {}", exc)

        # ── FormRetriever (instance partagée) ─────────────────────────────────
        self.form_retriever = None
        if enable_form_suggestions and _FORM_RETRIEVER_AVAILABLE:
            try:
                self.form_retriever = FormRetriever(
                    shared_embeddings=self.embeddings,
                    shared_reranker=self.shared_reranker,
                )
                logger.info(
                    "Suggestions formulaires : {}",
                    "activées" if self.form_retriever.available else "désactivées (lance index_forms.py)",
                )
            except Exception as exc:
                logger.warning("FormRetriever init échoué: {}", exc)

        # ── LLM ───────────────────────────────────────────────────────────────
        client_kwargs = (
            {"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}} if OLLAMA_API_KEY else None
        )
        llm_kwargs = dict(
            model=local_model,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
            num_predict=OLLAMA_NUM_PREDICT,
            num_ctx=OLLAMA_NUM_CTX,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        if client_kwargs:
            llm_kwargs["client_kwargs"] = client_kwargs

        logger.info("LLM: {} @ {}", local_model, OLLAMA_BASE_URL)
        self.llm    = ChatOllama(**llm_kwargs)
        self.parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt()),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | self.parser

    # ── Helpers privés ────────────────────────────────────────────────────────

    def _get_collection_count(self, path: Path, collection_name: str) -> int:
        try:
            return PersistentClient(path=str(path)).get_collection(collection_name).count()
        except Exception:
            return 0

    def _system_prompt(self) -> str:
        return """Vous etes un assistant virtuel officiel specialise dans les organismes de retraite et d'assurance marocains RCAR et CNRA.

    DEFINITIONS:
    - RCAR (Regime Collectif d'Allocation de Retraite) : regime de retraite complementaire destine aux agents non titulaires de l'Etat et des collectivites locales, ainsi qu'au personnel des etablissements publics.
    - CNRA (Caisse Nationale de Retraites et d'Assurances) : etablissement public gerant les rentes d'accidents du travail, de circulation, les rentes viageres et les produits d'assurance-vie au Maroc. Gere par la CDG.

    VOTRE ROLE:
    Vous aidez les affilies, retraites, employeurs et citoyens marocains a comprendre leurs droits, demarches, prestations et procedures liees au RCAR et a la CNRA, en vous basant exclusivement sur les informations officielles extraites des sites rcar.ma et cnra.ma.

    REGLES STRICTES:
    - Repondez uniquement en francais.
    - Basez-vous EXCLUSIVEMENT sur le contexte fourni ci-dessous. N'inventez, n'interpolez ou ne supposez aucune information absente du contexte.
    - Si le contexte ne contient pas la reponse, dites clairement : "Je ne dispose pas de cette information dans ma base documentaire. Consultez directement rcar.ma ou cnra.ma, ou contactez leurs services."
    - Ne donnez jamais de conseils juridiques, fiscaux ou financiers personnalises.
    - Soyez precis sur les organismes : distinguez toujours ce qui concerne le RCAR de ce qui concerne la CNRA.

    FORMAT DE REPONSE:
    - Texte brut uniquement, sans aucune syntaxe Markdown.
    - Reponses structurees en phrases courtes et claires.
    - Pour les listes, commencez chaque element par "- " sur une nouvelle ligne.
    - Longueur adaptee a la question.

    CONTEXTE DOCUMENTAIRE:
    {context}"""

    def _warm_up_embeddings(self) -> None:
        try:
            t = time.perf_counter()
            self.embeddings.embed_query("verification initiale")
            logger.info("Warmup embeddings: {:.3f}s", time.perf_counter() - t)
        except Exception as exc:
            logger.warning("Warmup ignoré: {}", exc)

    def _build_context(self, docs: list) -> str:
        parts, total = [], 0
        for doc in docs:
            chunk = (doc.page_content or "")[: self.max_doc_chars]
            if not chunk or total + len(chunk) > self.max_context_chars:
                break
            parts.append(chunk)
            total += len(chunk) + 1
        return "\n".join(parts)

    def _cache_get(self, key: str) -> dict | None:
        entry = self.response_cache.get(key)
        if entry:
            self.response_cache.move_to_end(key)
        return entry

    def _cache_set(self, key: str, value: dict) -> None:
        self.response_cache[key] = value
        self.response_cache.move_to_end(key)
        while len(self.response_cache) > RAG_CACHE_SIZE:
            self.response_cache.popitem(last=False)

    def _retrieve_and_rerank_docs(self, query: str) -> list:
        """
        Recherche vectorielle sur chroma_db + reranking cross-encoder.

        Étapes :
          1. similarity_search(k=RAG_RETRIEVE_K)  → pool de candidats (défaut 12)
          2. shared_reranker.predict()             → score pertinence de chaque chunk
          3. tri + top RAG_FINAL_K                 → chunks finaux (défaut 4)

        Si le reranker est indisponible, retourne les RAG_FINAL_K premiers
        résultats vectoriels (comportement identique à avant).
        """
        # Étape 1 — pool vectoriel élargi
        candidates = self.vectorstore.similarity_search(query, k=self.retrieval_k)

        if not candidates:
            return []

        # Étape 2 — reranking (si disponible)
        if self.shared_reranker is not None:
            pairs = [(query, doc.page_content[:600]) for doc in candidates]
            try:
                scores = self.shared_reranker.predict(pairs).tolist()

                # Log debug : montre l'ordre avant/après reranking
                logger.debug("── Reranking chroma_db ({} candidats) ──", len(candidates))
                ranked = sorted(
                    zip(scores, candidates),
                    key=lambda x: x[0],
                    reverse=True,
                )
                for i, (sc, doc) in enumerate(ranked):
                    src = doc.metadata.get("relative_source", doc.metadata.get("source", "?"))
                    logger.debug(
                        "  [{}/{}] score={:.4f} | {}",
                        i + 1, len(ranked), sc,
                        src.split("/")[-1][:60] if src else "?"
                    )

                # Étape 3 — top-K après reranking
                top_docs = [doc for _, doc in ranked[: self.final_k]]
                logger.info(
                    "Reranking docs: {} candidats → {} retenus (meilleur score: {:.4f})",
                    len(candidates), len(top_docs), ranked[0][0] if ranked else 0,
                )
                return top_docs

            except Exception as exc:
                logger.warning("Reranker erreur sur chroma_db ({}), fallback vectoriel", exc)

        # Fallback : pas de reranker → top RAG_FINAL_K résultats vectoriels
        return candidates[: self.final_k]

    def _retrieve_videos(self, query: str) -> list[dict]:
        if not self.video_retriever or not self.video_retriever.available:
            return []
        try:
            return self.video_retriever.retrieve(query)
        except Exception as exc:
            logger.warning("Erreur recherche vidéo: {}", exc)
            return []

    def _retrieve_forms(self, query: str) -> list[dict]:
        if not self.form_retriever or not self.form_retriever.available:
            return []
        try:
            return self.form_retriever.retrieve(query)
        except Exception as exc:
            logger.warning("Erreur recherche formulaires: {}", exc)
            return []

    # ── API publique ──────────────────────────────────────────────────────────

    def query(self, query: str) -> dict:
        """
        Traite une question utilisateur.

        Retourne :
            response, context_docs, videos, forms, original_query, cached
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return {"response": "Veuillez saisir une question.", "error": True, "videos": [], "forms": []}

        if cached := self._cache_get(cleaned.lower()):
            return {**cached, "original_query": query, "cached": True}

        if self.collection_count == 0:
            return {
                "response": "Base vectorielle vide — lancez index_data.py puis réessayez.",
                "error": True, "videos": [], "forms": [],
            }

        # ── Retrieval + reranking documentaire ───────────────────────────────
        t0 = time.perf_counter()
        try:
            docs    = self._retrieve_and_rerank_docs(cleaned)
            context = self._build_context(docs)
            logger.info(
                "Retrieval+reranking docs: {} chunks en {:.3f}s",
                len(docs), time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.error("Erreur retrieval: {}", exc)
            return {"response": "Erreur lors de la recherche.", "error": True, "videos": [], "forms": []}

        if not docs:
            return {"response": "Aucun passage pertinent trouvé.", "context_docs": 0, "videos": [], "forms": []}

        # ── Vidéos + formulaires en parallèle ─────────────────────────────────
        videos, forms = [], []
        t_par = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_v = pool.submit(self._retrieve_videos, cleaned)
                fut_f = pool.submit(self._retrieve_forms, cleaned)
                videos = fut_v.result(timeout=30)
                forms  = fut_f.result(timeout=30)
            logger.info(
                "Parallel retrieval: vidéos={} formulaires={} en {:.3f}s",
                len(videos), len(forms), time.perf_counter() - t_par,
            )
        except Exception as exc:
            logger.warning("Parallel retrieval échoué (fallback séquentiel): {}", exc)
            videos = self._retrieve_videos(cleaned)
            forms  = self._retrieve_forms(cleaned)

        # ── Génération LLM ────────────────────────────────────────────────────
        t1 = time.perf_counter()
        try:
            response = str(self.chain.invoke({"context": context, "question": cleaned})).strip()
            logger.info("Generation: {:.3f}s | {} chars", time.perf_counter() - t1, len(response))
        except Exception as exc:
            logger.error("Erreur génération: {}", exc)
            return {
                "response": f"Erreur de génération. Vérifiez qu'Ollama est lancé (ollama pull {OLLAMA_MODEL}).",
                "error": True, "videos": videos, "forms": forms,
            }

        payload = {
            "response":     response,
            "context_docs": len(docs),
            "videos":       videos,
            "forms":        forms,
        }
        self._cache_set(cleaned.lower(), payload)
        return {**payload, "original_query": query, "cached": False}


if __name__ == "__main__":
    setup_logger(log_dir=LOGS_DIR, source="pipeline")
    rag = RAGPipeline()
    print(rag.query("comment faire une demande de pension de retraite RCAR ?"))