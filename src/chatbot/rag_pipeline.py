"""
src/chatbot/rag_pipeline.py
============================
Pipeline RAG principal + reranking sur chroma_db + suggestions vidéos + formulaires.

Flux de retrieval documentaire :
    1. similarity_search(k=RAG_RETRIEVE_K)   → 12 candidats vectoriels
    2. shared_reranker.predict()              → score cross-encoder sur chaque candidat
    3. tri décroissant → top RAG_FINAL_K=4   → chunks envoyés au LLM

Champs retournés par query() — TOUJOURS présents (succès ET erreur) :
    {
        "response":          str,
        "context_docs":      int,        # nombre de chunks après reranking
        "videos":            list[dict],
        "forms":             list[dict],
        "intent":            str,        # intent classifié (défaut: "retrieval")
        "intent_confidence": float,      # score cosinus moyen (défaut: 0.0)
        "original_query":    str,
        "cached":            bool,
        "error":             bool,       # présent uniquement si True
    }
"""
# ── Compatibilité Python 3.8+ pour les annotations de type ───────────────────
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
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
    from chatbot.intent_classifier import IntentClassifier
    _INTENT_CLASSIFIER_AVAILABLE = True
except ImportError:
    _INTENT_CLASSIFIER_AVAILABLE = False
    logger.warning("IntentClassifier non disponible.")

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

VECTORSTORE_RELATIVE_PATH = Path("data") / "vectorstore" / "chroma_db_unified"
DEFAULT_EMBEDDING_MODEL   = "BAAI/bge-m3"
RERANKER_MODEL            = "BAAI/bge-reranker-v2-m3"
HF_TOKEN = getenv("HF_TOKEN", "").strip().strip('"\'')

OLLAMA_MODEL       = getenv("OLLAMA_MODEL", "mistral:latest").strip().strip('"\'')
OLLAMA_BASE_URL    = getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY     = getenv("OLLAMA_API_KEY", "").strip().strip('"\'')
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 1200
OLLAMA_NUM_CTX     = 4096
OLLAMA_KEEP_ALIVE  = "30m"

# ── Paramètres retrieval principal ────────────────────────────────────────────
# RAG_RETRIEVE_K : candidats récupérés par similarity_search (pool pour le reranker)
# RAG_FINAL_K    : chunks conservés après reranking → envoyés au LLM
# Règle : RAG_RETRIEVE_K >= RAG_FINAL_K (typiquement 3x)
RAG_RETRIEVE_K        = 12    # Pool élargi pour donner du choix au reranker
RAG_FINAL_K           = 4     # Chunks finaux après reranking
RAG_MAX_CONTEXT_CHARS = 4000
RAG_MAX_DOC_CHARS     = 1000
RAG_CACHE_SIZE        = 100
FAQ_RERANK_BOOST      = 0.20

# ── Mots-cles off-scope par organisme (texte normalise ASCII) ───────────────
CNRA_KEYWORDS = {
    "cnra",
    "recore",
    "assurance vie",
    "accident du travail",
    "accident travail",
    "rente accident travail",
    "rente circulation",
    "rente accident",
    "rentes viageres",
    "rente viagere",
    "rente at cnra",
    "rentes at cnra",
    "fram",
    "crac",
    "douayer zmane",
    "ihtiyate",
}

RCAR_KEYWORDS = {
    "rcar",
    "retraite complementaire",
    "agents non titulaires",
    "regime collectif d allocation de retraite",
    "allocation de retraite",
    "retraite rcar",
    "pension rcar",
    "ehtiyati",
    "regime general rcar",
    "regime complementaire rcar",
    "cotisation rcar",
    "affiliation rcar",
    "pension complementaire",
}


class RAGPipeline:
    def __init__(
        self,
        local_model: str = OLLAMA_MODEL,
        vectorstore_path: Path | str = VECTORSTORE_RELATIVE_PATH,
        collection_name: str = "rcar_cnra_unified",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_intent_classifier: bool = True,
        enable_video_suggestions: bool = True,
        enable_form_suggestions:  bool = True,
    ):
        resolved_path = Path(vectorstore_path)
        if not resolved_path.is_absolute():
            resolved_path = (BASE_DIR / resolved_path).resolve()

        self.collection_name   = collection_name
        self.retrieval_k       = max(RAG_RETRIEVE_K, RAG_FINAL_K * 6)
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

        # ── Intent Classifier (réutilise les embeddings partagés) ─────────────
        self.intent_classifier: IntentClassifier | None = None
        if enable_intent_classifier and _INTENT_CLASSIFIER_AVAILABLE:
            try:
                self.intent_classifier = IntentClassifier(
                    shared_embeddings=self.embeddings,
                )
                stats = self.intent_classifier.get_stats()
                logger.info(
                    "IntentClassifier prêt: {} intents, {} exemples",
                    stats.get("total_intents", 0), stats.get("total_examples", 0)
                )
            except Exception as exc:
                logger.warning("IntentClassifier init échoué: {} — gate désactivé", exc)

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

    def _error_response(
        self,
        message: str,
        original_query: str = "",
        *,
        videos: list | None = None,
        forms: list | None = None,
        intent: str = "retrieval",
        intent_confidence: float = 0.0,
        context_docs: int = 0,
    ) -> dict:
        """
        Construit un payload d'erreur uniforme, avec TOUTES les clés du
        payload nominal. Évite les KeyError côté appelant si l'accès se
        fait sans .get().
        """
        return {
            "response":          message,
            "error":             True,
            "context_docs":      context_docs,
            "videos":            videos or [],
            "forms":             forms or [],
            "intent":            intent,
            "intent_confidence": intent_confidence,
            "original_query":    original_query,
            "cached":            False,
        }

    def _build_org_identity(self, org: str) -> str:
        """
        Génère le bloc d'identité org-spécifique pour le prompt système.
        Injecte clairement au LLM son rôle et ses limites selon l'org actif.
        """
        if org == "cnra":
            return (
                "VOTRE IDENTITE ET SCOPE:\n"
                "Vous etes l'assistant officiel de la CNRA (Caisse Nationale de Retraites et d'Assurances).\n"
                "DEFINITION CNRA:\n"
                "- Etablissement public gerant les rentes d'accidents du travail et de circulation,\n"
                "  les rentes viageres et les produits d'assurance-vie au Maroc (branche CDG).\n"
                "\n"
                "Vous repondez UNIQUEMENT sur les sujets CNRA :\n"
                "  - Rentes d'accidents du travail et de circulation\n"
                "  - Rentes viageres et produits d'assurance-vie\n"
                "  - Procedures et demarches CNRA\n"
                "\n"
                "RESTRICTIONS STRICTES :\n"
                "- Vous n'etes PAS l'assistant du RCAR.\n"
                "- Si l'utilisateur pose une question EXCLUSIVEMENT sur le RCAR (retraite complementaire agents non titulaires),\n"
                "  repondez : 'Cette question concerne le RCAR, pas la CNRA. Je suis l'assistant CNRA uniquement.\n"
                "  Consultez le site rcar.ma ou l'assistant RCAR pour cette demande.'\n"
                "- N'ajoutez AUCUNE information sur le RCAR dans votre reponse."
            )
        elif org == "rcar":
            return (
                "VOTRE IDENTITE ET SCOPE:\n"
                "Vous etes l'assistant officiel du RCAR (Regime Collectif d'Allocation de Retraite).\n"
                "DEFINITION RCAR:\n"
                "- Regime de retraite complementaire destine aux agents non titulaires de l'Etat,\n"
                "  des collectivites locales et au personnel des etablissements publics.\n"
                "\n"
                "Vous repondez UNIQUEMENT sur les sujets RCAR :\n"
                "  - Retraite complementaire pour agents non titulaires\n"
                "  - Regimes general et complementaire du RCAR\n"
                "  - Demarches d'affiliation et de retraite\n"
                "  - Procedures administratives RCAR\n"
                "\n"
                "RESTRICTIONS STRICTES :\n"
                "- Vous n'etes PAS l'assistant de la CNRA.\n"
                "- Si l'utilisateur pose une question EXCLUSIVEMENT sur la CNRA (rentes AT/circulation, assurance-vie),\n"
                "  repondez : 'Cette question concerne la CNRA, pas le RCAR. Je suis l'assistant RCAR uniquement.\n"
                "  Consultez le site cnra.ma ou l'assistant CNRA pour cette demande.'\n"
                "- N'ajoutez AUCUNE information sur la CNRA dans votre reponse."
            )
        else:  # org == "all"
            return (
                "VOTRE IDENTITE ET SCOPE:\n"
                "Vous etes l'assistant conjoint officiel du RCAR et de la CNRA.\n"
                "DEFINITIONS:\n"
                "- RCAR (Regime Collectif d'Allocation de Retraite) : retraite complementaire pour\n"
                "  agents non titulaires de l'Etat et des collectivites locales.\n"
                "- CNRA (Caisse Nationale de Retraites et d'Assurances) : rentes AT/circulation,\n"
                "  rentes viageres, assurance-vie (branche CDG).\n"
                "\n"
                "DISTINCTIONS IMPORTANTES :\n"
                "- Precisez toujours quel organisme concerne la reponse.\n"
                "- Si une question concerne UN SEUL organisme, mentionnez-le clairement.\n"
                "- Si une question concerne les DEUX, clarifiez les roles de chacun."
            )

    def _system_prompt(self) -> str:
        return """Vous etes un assistant virtuel officiel.

    {org_identity}

    REGLES STRICTES:
    - Repondez uniquement en francais.
    - Basez-vous EXCLUSIVEMENT sur le contexte fourni ci-dessous. N'inventez, n'interpolez ou ne supposez aucune information absente du contexte.
    - Si le contexte ne contient pas la reponse, dites clairement : "Je ne dispose pas de cette information dans ma base documentaire. Consultez directement rcar.ma ou cnra.ma, ou contactez leurs services."
    - Ne donnez jamais de conseils juridiques, fiscaux ou financiers personnalises.
    - Soyez precis sur les organismes : distinguez toujours ce qui concerne le RCAR de ce qui concerne la CNRA.

    RESSOURCES SUPPLEMENTAIRES DISPONIBLES:
    {supplementary_hint}

    FORMAT DE REPONSE:
    - Texte brut uniquement, sans aucune syntaxe Markdown.
    - Reponses structurees en phrases courtes et claires.
    - Pour les listes, commencez chaque element par "- " sur une nouvelle ligne.
    - Longueur adaptee a la question.
    - Si des videos ou formulaires sont disponibles (indiques dans RESSOURCES SUPPLEMENTAIRES), mentionnez-les brievement a la fin de votre reponse.
    - Si AUCUNE ressource n'est disponible, ne mentionnez PAS les videos ou formulaires. Ne dites JAMAIS \"aucune video\", \"aucun formulaire\" ou toute phrase similaire indiquant leur absence.

    CONTEXTE DOCUMENTAIRE:
    {context}"""

    def _warm_up_embeddings(self) -> None:
        try:
            t = time.perf_counter()
            self.embeddings.embed_query("verification initiale")
            logger.info("Warmup embeddings: {:.3f}s", time.perf_counter() - t)
        except Exception as exc:
            logger.warning("Warmup ignoré: {}", exc)

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return f" {normalized.strip()} "

    def _contains_any_keyword(self, normalized_text: str, keywords: set[str]) -> bool:
        return any(f" {kw} " in normalized_text for kw in keywords)

    def _detect_offscope_org(self, query: str, org: str) -> str:
        """
        Detecte si la question concerne exclusivement l'autre organisme.
        Retourne "cnra", "rcar" ou "" si rien de detecte.

        Version conservative : ne bloque que si le mot-cle est present ET
        suffisamment discriminant (longueur > 4 chars pour eviter les faux positifs).
        """
        if org == "all":
            return ""
        normalized = self._normalize_text(query)

        # Seuil : ignorer les keywords trop courts (risque de faux positifs)
        min_keyword_len = 5

        if org == "rcar":
            for kw in CNRA_KEYWORDS:
                if len(kw) >= min_keyword_len and f" {kw} " in normalized:
                    return "cnra"

        if org == "cnra":
            for kw in RCAR_KEYWORDS:
                if len(kw) >= min_keyword_len and f" {kw} " in normalized:
                    return "rcar"
        return ""

    def _offscope_response(self, org: str, target: str, original_query: str) -> dict:
        if target == "cnra":
            message = (
                "Cette question concerne la CNRA, pas le RCAR. Je suis l'assistant RCAR uniquement. "
                "Consultez le site cnra.ma ou l'assistant CNRA pour cette demande."
            )
        else:
            message = (
                "Cette question concerne le RCAR, pas la CNRA. Je suis l'assistant CNRA uniquement. "
                "Consultez le site rcar.ma ou l'assistant RCAR pour cette demande."
            )

        return {
            "response":          message,
            "context_docs":      0,
            "videos":            [],
            "forms":             [],
            "intent":            "out_of_scope",
            "intent_confidence": 1.0,
            "original_query":    original_query,
            "cached":            False,
            "org":               org,
        }

    def _build_org_filter(self, org: str, doc_type: str) -> dict:
        """
        Construit le filtre ChromaDB WHERE selon l'organisme actif.
        
        - org="all"  → filtre sur type uniquement
        - org="cnra" | "rcar" → filtre $and type + org (inclut 'both')
        """
        if org == "all":
            return {"type": {"$eq": doc_type}}
        return {
            "$and": [
                {"type": {"$eq": doc_type}},
                {"org":  {"$in": [org, "both"]}},
            ]
        }

    def _build_supplementary_hint(self, videos: list, forms: list) -> str:
        """
        Informe le LLM des ressources disponibles (vidéos et formulaires).
        Évite que le LLM réponde "je n'ai pas de vidéo/formulaire" alors
        qu'ils sont affichés dans l'interface.
        """
        parts = []
        if videos:
            titles = ", ".join(f'"{v.get("title", "vidéo")}"' for v in videos[:2])
            parts.append(
                f"Des videos YouTube pertinentes sont disponibles pour cette question "
                f"({titles}). Elles seront affichees a l'utilisateur."
            )
        if forms:
            titles = ", ".join(f'"{f.get("title", "formulaire")}"' for f in forms[:2])
            parts.append(
                f"Des formulaires PDF sont disponibles ({titles}). Ils seront affiches "
                f"a l'utilisateur avec un lien de telechargement."
            )
        if not parts:
            return "Aucune ressource supplementaire disponible. Ne mentionnez PAS les videos ou formulaires dans votre reponse."
        return " ".join(parts)

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

    @staticmethod
    def _doc_priority(doc) -> int:
        source_type = str(doc.metadata.get("source_type", "")).lower()
        if source_type == "faq":
            return 3
        if source_type == "web":
            return 2
        if source_type == "bibliotheque":
            return 1
        try:
            return int(doc.metadata.get("source_priority", 0))
        except (TypeError, ValueError):
            return 0

    def _retrieve_and_rerank_docs(self, query: str, org: str = "all") -> list:
        """
        Recherche vectorielle sur chroma_db + reranking cross-encoder.

        Étapes :
          1. similarity_search(k=RAG_RETRIEVE_K)  → pool de candidats (défaut 12)
          2. shared_reranker.predict()             → score pertinence de chaque chunk
          3. tri + top RAG_FINAL_K                 → chunks finaux (défaut 4)

        Si le reranker est indisponible, retourne les RAG_FINAL_K premiers
        résultats vectoriels (comportement identique à avant).
        """
        # Étape 1 — pool vectoriel élargi, filtré sur type=doc + org
        candidates = self.vectorstore.similarity_search(
            query,
            k=self.retrieval_k,
            filter=self._build_org_filter(org, "doc"),
        )

        if not candidates:
            return []

        # Étape 2 — reranking (si disponible)
        if self.shared_reranker is not None:
            pairs = [(query, doc.page_content[:600]) for doc in candidates]
            try:
                raw_scores = self.shared_reranker.predict(pairs).tolist()
                scores = [
                    score + (FAQ_RERANK_BOOST if self._doc_priority(doc) >= 3 else 0.0)
                    for score, doc in zip(raw_scores, candidates)
                ]

                logger.debug("── Reranking chroma_db ({} candidats) ──", len(candidates))
                ranked = sorted(
                    zip(scores, candidates),
                    key=lambda x: x[0],
                    reverse=True,
                )
                for i, (sc, doc) in enumerate(ranked):
                    src = doc.metadata.get("relative_source", doc.metadata.get("source", "?"))
                    logger.debug(
                        "  [{}/{}] score={:.4f} | priority={} | {}",
                        i + 1, len(ranked), sc, self._doc_priority(doc),
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
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (self._doc_priority(item[1]), -item[0]),
            reverse=True,
        )
        return [doc for _, doc in ranked[: self.final_k]]

    def retrieve_documents(self, query: str, org: str = "all") -> list:
        """Evaluation-friendly public wrapper around document retrieval."""
        return self._retrieve_and_rerank_docs(query, org=org)

    def _retrieve_videos(self, query: str, org: str = "all") -> list[dict]:
        if not self.video_retriever or not self.video_retriever.available:
            return []
        try:
            return self.video_retriever.retrieve(query, org=org)
        except Exception as exc:
            logger.warning("Erreur recherche vidéo: {}", exc)
            return []

    def _retrieve_forms(self, query: str, org: str = "all") -> list[dict]:
        if not self.form_retriever or not self.form_retriever.available:
            return []
        try:
            return self.form_retriever.retrieve(query, org=org)
        except Exception as exc:
            logger.warning("Erreur recherche formulaires: {}", exc)
            return []

    def retrieve_videos(self, query: str, org: str = "all", limit: int | None = None) -> list[dict]:
        """Return video retrieval results without running chat generation."""
        videos = self._retrieve_videos(query, org=org)
        return videos[:limit] if limit else videos

    def retrieve_forms(self, query: str, org: str = "all", limit: int | None = None) -> list[dict]:
        """Return form retrieval results without running chat generation."""
        forms = self._retrieve_forms(query, org=org)
        return forms[:limit] if limit else forms

    # ── API publique ──────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        org: str = "all",
        include_contexts: bool = False,
        use_intent_classifier: bool = True,
        include_resources: bool = True,
        **_: object,
    ) -> dict:
        """
        Traite une question utilisateur avec filtrage par organisme.

        Retourne TOUJOURS un dict avec les clés :
            response, context_docs, videos, forms, intent, intent_confidence,
            original_query, org, cached.
        La valeur intent peut etre "out_of_scope" en cas de blocage.
        La clé "error": True est ajoutée en cas d'échec.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return self._error_response(
                "Veuillez saisir une question.",
                query,
            )

        # ── Off-scope guard (evite reponses CNRA/RCAR dans le mauvais mode) ──
        offscope = self._detect_offscope_org(cleaned, org)
        if offscope:
            logger.info("Off-scope detecte (org={}, cible={})", org, offscope)
            return self._offscope_response(org, offscope, query)

        # ── Cache ─────────────────────────────────────────────────────────────
        # Clé de cache incluant l'org (évite les collisions inter-orgs)
        cache_key = f"{org}:{cleaned.lower()}:{include_contexts}:{use_intent_classifier}:{include_resources}"
        # Renommé 'hit' pour éviter la collision de nom avec la clé "cached"
        # du payload retourné dans le dict déballé juste après.
        if hit := self._cache_get(cache_key):
            return {**hit, "original_query": query, "cached": True}

        if self.collection_count == 0:
            return self._error_response(
                "Base vectorielle vide — lancez index_data.py puis réessayez.",
                query,
            )

        # ── Retrieval + reranking documentaire ───────────────────────────────
        t0 = time.perf_counter()
        try:
            docs    = self._retrieve_and_rerank_docs(cleaned, org=org)
            context = self._build_context(docs)
            logger.info(
                "Retrieval+reranking docs: {} chunks (org={}) en {:.3f}s",
                len(docs), org, time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.error("Erreur retrieval: {}", exc)
            return self._error_response("Erreur lors de la recherche.", query)

        if not docs:
            return self._error_response(
                "Aucun passage pertinent trouvé.",
                query,
                context_docs=0,
            )

        # ── Intent classification + gate retrievers ───────────────────────────
        videos, forms = [], []
        classification = {"intent": "retrieval", "confidence": 0.0, "loaded": False}

        if use_intent_classifier and self.intent_classifier and self.intent_classifier.is_loaded:
            try:
                classification = self.intent_classifier.classify(cleaned)
                logger.info(
                    "Intent: {} [tier {}] conf={:.3f}",
                    classification["intent"], classification["tier"], classification["confidence"]
                )
            except Exception as exc:
                logger.warning("Intent classification échouée: {} — gate pass-through", exc)

        run_videos = include_resources and (
            self.intent_classifier is None or
            not use_intent_classifier or
            not self.intent_classifier.is_loaded or
            self.intent_classifier.should_retrieve_videos(classification)
        )
        run_forms = include_resources and (
            self.intent_classifier is None or
            not use_intent_classifier or
            not self.intent_classifier.is_loaded or
            self.intent_classifier.should_retrieve_forms(classification)
        )

        t_par = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_v = pool.submit(self._retrieve_videos, cleaned, org) if run_videos else None
                fut_f = pool.submit(self._retrieve_forms, cleaned, org) if run_forms else None
                videos = fut_v.result(timeout=30) if fut_v else []
                forms  = fut_f.result(timeout=30) if fut_f else []
            logger.info(
                "Retrieval: vidéos={} (gate={}) formulaires={} (gate={}) en {:.3f}s",
                len(videos), "ON" if run_videos else "OFF",
                len(forms),  "ON" if run_forms  else "OFF",
                time.perf_counter() - t_par,
            )
        except Exception as exc:
            logger.warning("Parallel retrieval échoué (fallback séquentiel): {}", exc)
            if run_videos:
                videos = self._retrieve_videos(cleaned, org)
            if run_forms:
                forms = self._retrieve_forms(cleaned, org)

        # ── Génération LLM ────────────────────────────────────────────────────
        t1 = time.perf_counter()
        try:
            supplementary_hint = self._build_supplementary_hint(videos, forms)
            org_identity = self._build_org_identity(org)
            response = str(self.chain.invoke({
                "context":            context,
                "question":           cleaned,
                "supplementary_hint": supplementary_hint,
                "org_identity":       org_identity,
            })).strip()
            logger.info("Generation: {:.3f}s | {} chars", time.perf_counter() - t1, len(response))
        except Exception as exc:
            logger.error("Erreur génération: {}", exc)
            return self._error_response(
                f"Erreur de génération. Vérifiez qu'Ollama est lancé (ollama pull {OLLAMA_MODEL}).",
                query,
                videos=videos,
                forms=forms,
                intent=classification.get("intent", "retrieval"),
                intent_confidence=classification.get("confidence", 0.0),
                context_docs=len(docs),
            )

        # ── Payload nominal ───────────────────────────────────────────────────
        payload = {
            "response":          response,
            "context_docs":      len(docs),
            "videos":            videos,
            "forms":             forms,
            "intent":            classification.get("intent", "retrieval"),
            "intent_confidence": classification.get("confidence", 0.0),
            "org":               org,
        }
        if include_contexts:
            payload.update({
                "contexts": [doc.page_content for doc in docs],
                "context_metadata": [dict(doc.metadata or {}) for doc in docs],
                "context_items": [
                    {"text": doc.page_content, "metadata": dict(doc.metadata or {})}
                    for doc in docs
                ],
            })
        self._cache_set(cache_key, payload)
        return {**payload, "original_query": query, "cached": False}


if __name__ == "__main__":
    setup_logger(log_dir=LOGS_DIR, source="pipeline")
    rag = RAGPipeline()
    print(rag.query("comment faire une demande de pension de retraite RCAR ?"))
