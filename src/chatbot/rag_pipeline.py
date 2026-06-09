from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
import warnings
from collections import OrderedDict
from pathlib import Path
from os import getenv

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=r"Accessing `__path__` from `\.models\..*")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
except ImportError:
    torch = None

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from loguru import logger

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from config.logger import setup_logger
from config.settings import BASE_DIR, LOGS_DIR, VESPA_PORT, VESPA_URL
from store.vespa_store import count_schema, make_vespa_app, query_schema

try:
    from chatbot.intent_classifier import IntentClassifier
    _INTENT_CLASSIFIER_AVAILABLE = True
except ImportError:
    _INTENT_CLASSIFIER_AVAILABLE = False
    logger.warning("IntentClassifier non disponible.")

try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass

load_dotenv(BASE_DIR / ".env")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
HF_TOKEN = getenv("HF_TOKEN", "").strip().strip('"\'')

OLLAMA_MODEL = getenv("OLLAMA_MODEL", "ministral-3:14b-cloud").strip().strip('"\'')
OLLAMA_BASE_URL = getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY = getenv("OLLAMA_API_KEY", "").strip().strip('"\'')
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 1200
OLLAMA_NUM_CTX = 4096
OLLAMA_KEEP_ALIVE = "30m"

RAG_RETRIEVE_K = 12
RAG_FINAL_K = 4
RAG_MAX_CONTEXT_CHARS = 4000
RAG_MAX_DOC_CHARS = 1000
RAG_CACHE_SIZE = 100
DOC_RERANK_CANDIDATES_PER_FINAL = 8
OFFICIAL_DOC_RETRIEVE_K = 96
FAQ_RERANK_BOOST = 0.10
OFFICIAL_DOC_NONFAQ_RERANK_BOOST = 0.28
OFFICIAL_DOC_BIBLIOTHEQUE_EXTRA_BOOST = 0.05
OFFICIAL_DOC_FAQ_RERANK_PENALTY = -0.05
OFFICIAL_DOC_QUERY_TERMS = (
    " document officiel ",
    " documents officiels ",
    " informations officielles ",
    " texte officiel ",
    " textes officiels ",
    " decret ",
    " reglement ",
    " loi ",
    " arrete ",
    " dahir ",
    " circulaire ",
    " galerie documentaire ",
)

VIDEO_RETRIEVE_K = 10
VIDEO_FINAL_K = 1
FORM_RETRIEVE_K = 10
FORM_FINAL_K = 1
VIDEO_RELEVANCE_THRESHOLD = 0.185
VIDEO_RERANK_RELEVANCE_THRESHOLD = 0.0
VIDEO_RERANK_CONFIDENCE_THRESHOLD = 0.100
VIDEO_RERANK_MARGIN_THRESHOLD = 0.200
FORM_RELEVANCE_THRESHOLD = 0.220
FORM_RERANK_RELEVANCE_THRESHOLD = 0.800

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
        vespa_url: str = VESPA_URL,
        vespa_port: int = VESPA_PORT,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_video_suggestions: bool = True,
        enable_form_suggestions: bool = True,
        enable_intent_classifier: bool = True,
    ):
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        self.vespa = make_vespa_app(vespa_url, vespa_port)
        self.enable_video_suggestions = enable_video_suggestions
        self.enable_form_suggestions = enable_form_suggestions
        self.enable_intent_classifier = enable_intent_classifier
        self.retrieval_k = max(RAG_RETRIEVE_K, RAG_FINAL_K * DOC_RERANK_CANDIDATES_PER_FINAL)
        self.final_k = RAG_FINAL_K
        self.max_context_chars = RAG_MAX_CONTEXT_CHARS
        self.max_doc_chars = RAG_MAX_DOC_CHARS
        self.response_cache: OrderedDict = OrderedDict()

        logger.info("Vespa: {}:{}", vespa_url, vespa_port)

        try:
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        except Exception:
            device = "cpu"

        logger.info("Embedding: {} sur {}", embedding_model, device)
        model_kwargs: dict = {"device": device}
        if HF_TOKEN:
            model_kwargs["token"] = HF_TOKEN

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

        try:
            if device == "cuda" and hasattr(self.embeddings, "client"):
                if hasattr(self.embeddings.client, "model"):
                    self.embeddings.client.model.half()
                    logger.info("Embeddings castees en FP16 (GPU)")
        except Exception as exc:
            logger.debug("Embeddings FP16 cast echoue (fallback FP32): {}", exc)

        self._warm_up_embeddings()

        self.shared_reranker: CrossEncoder | None = None
        try:
            if CrossEncoder is None:
                raise ImportError("sentence_transformers indisponible")
            self.shared_reranker = CrossEncoder(
                RERANKER_MODEL,
                max_length=512,
                device=device,
            )
            if device == "cuda" and self.shared_reranker.model is not None:
                try:
                    self.shared_reranker.model.half()
                    logger.info("Reranker caste en FP16 (GPU)")
                except Exception as exc:
                    logger.debug("Reranker FP16 cast echoue: {}", exc)
            logger.info("Reranker charge: {} sur {}", RERANKER_MODEL, device)
        except Exception as exc:
            logger.warning("Reranker non disponible ({}), fallback ranking Vespa", exc)

        self.intent_classifier: IntentClassifier | None = None
        if self.enable_intent_classifier and _INTENT_CLASSIFIER_AVAILABLE:
            try:
                self.intent_classifier = IntentClassifier(shared_embeddings=self.embeddings)
                stats = self.intent_classifier.get_stats()
                logger.info(
                    "IntentClassifier pret: {} intents, {} exemples",
                    stats.get("total_intents", 0),
                    stats.get("total_examples", 0),
                )
            except Exception as exc:
                logger.warning("IntentClassifier init echoue: {} - gate desactive", exc)
        elif not self.enable_intent_classifier:
            logger.info("IntentClassifier desactive par configuration")

        self.collection_count = self._get_doc_count()
        logger.info("Vespa schema doc: {} chunks", self.collection_count)
        if self.collection_count == 0:
            logger.warning("Schema doc vide ou Vespa indisponible - lancez index_data.py apres deploy Vespa.")

        client_kwargs = {"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}} if OLLAMA_API_KEY else None
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
        self.llm = ChatOllama(**llm_kwargs)
        self.parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt()),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | self.parser

    def _get_doc_count(self) -> int:
        try:
            return count_schema(self.vespa, "doc")
        except Exception as exc:
            logger.warning("Impossible de compter les documents Vespa: {}", exc)
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
        return {
            "response": message,
            "error": True,
            "context_docs": context_docs,
            "videos": videos or [],
            "forms": forms or [],
            "intent": intent,
            "intent_confidence": intent_confidence,
            "original_query": original_query,
            "cached": False,
        }

    def _build_org_identity(self, org: str) -> str:
        if org == "cnra":
            return (
                "VOTRE IDENTITE ET SCOPE:\n"
                "Vous etes l'assistant officiel de la CNRA.\n"
                "Vous repondez uniquement sur les sujets CNRA: rentes d'accidents du travail et de circulation, "
                "rentes viageres, assurance-vie, procedures et demarches CNRA.\n"
                "Si la question concerne exclusivement le RCAR, dites que cette question concerne le RCAR, pas la CNRA."
            )
        if org == "rcar":
            return (
                "VOTRE IDENTITE ET SCOPE:\n"
                "Vous etes l'assistant officiel du RCAR.\n"
                "Vous repondez uniquement sur les sujets RCAR: retraite complementaire, regimes general et complementaire, "
                "affiliation, cotisation, pension et procedures RCAR.\n"
                "Si la question concerne exclusivement la CNRA, dites que cette question concerne la CNRA, pas le RCAR."
            )
        return (
            "VOTRE IDENTITE ET SCOPE:\n"
            "Vous etes l'assistant conjoint officiel du RCAR et de la CNRA.\n"
            "Distinguez toujours les informations qui concernent le RCAR de celles qui concernent la CNRA."
        )

    def _system_prompt(self) -> str:
        return """Vous etes un assistant virtuel officiel.

{org_identity}

REGLES STRICTES:
- Repondez uniquement en francais.
- Basez-vous EXCLUSIVEMENT sur le CONTEXTE DOCUMENTAIRE ci-dessous pour repondre au fond de la question. N'inventez, n'interpolez ou ne supposez aucune information absente du contexte.
- Si le contexte ne contient pas la reponse, dites clairement : "Je ne dispose pas de cette information dans ma base documentaire. Consultez directement rcar.ma ou cnra.ma, ou contactez leurs services."
- Ne donnez jamais de conseils juridiques, fiscaux ou financiers personnalises.
- Soyez precis sur les organismes : distinguez toujours ce qui concerne le RCAR de ce qui concerne la CNRA.
- Les ressources supplementaires ne font PAS partie du contexte documentaire. Ne les utilisez jamais pour justifier ou construire la reponse de fond.

RESSOURCES SUPPLEMENTAIRES DISPONIBLES:
{supplementary_hint}

FORMAT DE REPONSE:
- Texte brut uniquement, sans aucune syntaxe Markdown.
- Reponses structurees en phrases courtes et claires.
- Pour les listes, commencez chaque element par "- " sur une nouvelle ligne.
- Longueur adaptee a la question.
- Si des ressources supplementaires sont disponibles, ajoutez uniquement a la fin : "Des ressources complementaires sont affichees ci-dessous."
- Ne citez jamais les titres des videos ou formulaires dans votre reponse. Ils sont deja affiches separement dans l'interface.
- Si aucune ressource supplementaire n'est disponible, ne mentionnez pas les videos ou formulaires.

CONTEXTE DOCUMENTAIRE:
{context}"""

    def _warm_up_embeddings(self) -> None:
        try:
            t = time.perf_counter()
            self.embeddings.embed_query("verification initiale")
            logger.info("Warmup embeddings: {:.3f}s", time.perf_counter() - t)
        except Exception as exc:
            logger.warning("Warmup ignore: {}", exc)

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return f" {normalized.strip()} "

    def _detect_offscope_org(self, query: str, org: str) -> str:
        if org == "all":
            return ""
        normalized = self._normalize_text(query)
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
            "response": message,
            "context_docs": 0,
            "videos": [],
            "forms": [],
            "intent": "out_of_scope",
            "intent_confidence": 1.0,
            "original_query": original_query,
            "cached": False,
            "org": org,
        }

    def _build_supplementary_hint(self, videos: list, forms: list) -> str:
        kinds = []
        if videos:
            kinds.append("videos")
        if forms:
            kinds.append("formulaires")
        if not kinds:
            return "Aucune ressource supplementaire disponible. Ne mentionnez PAS les videos ou formulaires dans votre reponse."
        return (
            "Des ressources supplementaires sont disponibles dans l'interface: "
            + ", ".join(kinds)
            + ". Ne citez pas leurs titres et ne les utilisez pas comme contexte documentaire. "
            "Ajoutez seulement la phrase finale autorisee si cela reste naturel."
        )

    def _build_context(self, docs: list[Document]) -> str:
        parts, total = [], 0
        for doc in docs:
            chunk = (doc.page_content or "")[: self.max_doc_chars]
            if not chunk or total + len(chunk) > self.max_context_chars:
                break
            parts.append(chunk)
            total += len(chunk) + 1
        return "\n".join(parts)

    def _context_payload(self, docs: list[Document]) -> dict:
        context_items = [
            {
                "text": doc.page_content or "",
                "metadata": dict(doc.metadata or {}),
            }
            for doc in docs
        ]
        return {
            "contexts": [item["text"] for item in context_items],
            "context_metadata": [item["metadata"] for item in context_items],
            "context_items": context_items,
        }

    def _maybe_attach_contexts(self, payload: dict, docs: list[Document], include_contexts: bool) -> dict:
        if include_contexts:
            return {**payload, **self._context_payload(docs)}
        return payload

    @staticmethod
    def _has_context_payload(payload: dict) -> bool:
        return isinstance(payload.get("context_items"), list) or isinstance(payload.get("contexts"), list)

    def _cache_payload(self, payload: dict, docs: list[Document], include_contexts: bool) -> dict:
        if include_contexts:
            return {**payload, **self._context_payload(docs)}
        return payload

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
    def _doc_source_type(doc: Document) -> str:
        return str(doc.metadata.get("source_type", "")).lower()

    def _is_official_doc_query(self, normalized_query: str) -> bool:
        return any(term in normalized_query for term in OFFICIAL_DOC_QUERY_TERMS)

    def _doc_source_boost(self, normalized_query: str, doc: Document) -> tuple[float, str]:
        source_type = self._doc_source_type(doc)
        if self._is_official_doc_query(normalized_query):
            if source_type in {"web", "bibliotheque"}:
                boost = OFFICIAL_DOC_NONFAQ_RERANK_BOOST
                mode = "official_nonfaq"
                if source_type == "bibliotheque":
                    boost += OFFICIAL_DOC_BIBLIOTHEQUE_EXTRA_BOOST
                    mode = "official_bibliotheque"
                return boost, mode
            if source_type == "faq":
                return OFFICIAL_DOC_FAQ_RERANK_PENALTY, "official_faq_penalty"
            return 0.0, "official_neutral"

        if source_type == "faq":
            return FAQ_RERANK_BOOST, "faq_default"
        return 0.0, "neutral"

    def _doc_source_tiebreaker(self, normalized_query: str, doc: Document) -> int:
        source_type = self._doc_source_type(doc)
        if self._is_official_doc_query(normalized_query):
            if source_type == "bibliotheque":
                return 3
            if source_type == "web":
                return 2
            if source_type == "faq":
                return 1
            return 0
        if source_type == "faq":
            return 3
        if source_type == "web":
            return 2
        if source_type == "bibliotheque":
            return 1
        return 0

    def _rerank_docs(self, query: str, candidates: list[Document]) -> list[Document]:
        if not candidates:
            return []
        if self.shared_reranker is None:
            return candidates[: self.final_k]

        pairs = [(query, doc.page_content[:600]) for doc in candidates]
        try:
            raw_scores = self.shared_reranker.predict(pairs).tolist()
            normalized_query = self._normalize_text(query)
            ranked = sorted(
                (
                    (
                        score + boost,
                        score,
                        boost,
                        mode,
                        self._doc_source_tiebreaker(normalized_query, doc),
                        doc,
                    )
                    for score, doc in zip(raw_scores, candidates)
                    for boost, mode in [self._doc_source_boost(normalized_query, doc)]
                ),
                key=lambda item: (item[0], item[4], item[1]),
                reverse=True,
            )
            top_docs = []
            for rank, (score, raw_score, boost, mode, _, doc) in enumerate(ranked[: self.final_k], start=1):
                doc.metadata["vespa_rerank_score"] = float(score)
                doc.metadata["vespa_rerank_raw_score"] = float(raw_score)
                doc.metadata["vespa_rerank_source_boost"] = float(boost)
                doc.metadata["vespa_rerank_mode"] = mode
                doc.metadata["vespa_rerank_rank"] = rank
                top_docs.append(doc)
            logger.info(
                "Reranking Vespa docs: {} candidats -> {} retenus (meilleur score: {:.4f})",
                len(candidates),
                len(top_docs),
                ranked[0][0] if ranked else 0.0,
            )
            return top_docs
        except Exception as exc:
            logger.warning("Reranker erreur sur Vespa ({}), fallback ranking Vespa", exc)
            return candidates[: self.final_k]

    def _retrieve_docs_vespa(self, query: str, query_embedding: list[float], org: str = "all") -> list[Document]:
        candidate_k = self.retrieval_k
        if self._is_official_doc_query(self._normalize_text(query)):
            candidate_k = max(candidate_k, OFFICIAL_DOC_RETRIEVE_K)
        hits = query_schema(
            self.vespa,
            schema="doc",
            query_text=query,
            query_embedding=query_embedding,
            org=org,
            target_hits=candidate_k,
            hits=candidate_k,
        )
        candidates = []
        for index, hit in enumerate(hits, start=1):
            fields = hit.fields
            candidates.append(
                Document(
                    page_content=fields.get("text", ""),
                    metadata={
                        "doc_id": fields.get("doc_id", ""),
                        "org": fields.get("org", ""),
                        "source_type": fields.get("source_type", ""),
                        "faq_scope": fields.get("faq_scope", ""),
                        "relative_source": fields.get("relative_source", ""),
                        "chunk_index": fields.get("chunk_index", 0),
                        "vespa_relevance": hit.relevance,
                        "vespa_rank": index,
                    },
                )
            )
        return self._rerank_docs(query, candidates)

    @staticmethod
    def _video_rerank_text(fields: dict) -> str:
        title = fields.get("title", "")
        transcript = fields.get("transcript", "")
        return f"Titre: {title}\n\n{transcript[:600]}".strip()

    def _format_video_hit(
        self,
        hit,
        *,
        score: float,
        rerank_score: float | None = None,
        rank: int = 0,
        ranking_mode: str = "vespa",
    ) -> dict:
        fields = hit.fields
        return {
            "video_id": fields.get("video_id", ""),
            "title": fields.get("title", ""),
            "url": fields.get("url", ""),
            "thumbnail_url": fields.get("thumbnail_url", ""),
            "upload_date": fields.get("upload_date", ""),
            "org": fields.get("org", ""),
            "score": float(score),
            "vespa_score": float(hit.relevance),
            "rerank_score": float(rerank_score) if rerank_score is not None else None,
            "rank": rank,
            "ranking_mode": ranking_mode,
        }

    def _rank_video_hits(
        self,
        query: str,
        hits: list,
        *,
        limit: int,
        threshold: float,
        threshold_is_explicit: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        if not hits:
            return [], []

        vespa_threshold = threshold if threshold_is_explicit else VIDEO_RELEVANCE_THRESHOLD
        if self.shared_reranker is None:
            candidates = [
                self._format_video_hit(hit, score=hit.relevance, rank=index)
                for index, hit in enumerate(hits, start=1)
            ]
            filtered = [item for item in candidates if item["score"] >= vespa_threshold]
            return filtered[:limit], candidates

        pairs = [(query, self._video_rerank_text(hit.fields)) for hit in hits]
        try:
            raw_scores = self.shared_reranker.predict(pairs)
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            scores = [float(score) for score in raw_scores]
        except Exception as exc:
            logger.warning("Reranker erreur sur videos Vespa ({}), fallback score Vespa", exc)
            candidates = [
                self._format_video_hit(hit, score=hit.relevance, rank=index)
                for index, hit in enumerate(hits, start=1)
            ]
            filtered = [item for item in candidates if item["score"] >= vespa_threshold]
            return filtered[:limit], candidates

        best_per_video: dict[str, dict] = {}
        for hit, rerank_score in zip(hits, scores):
            video_id = hit.fields.get("video_id", "")
            if not video_id:
                continue
            item = self._format_video_hit(
                hit,
                score=rerank_score,
                rerank_score=rerank_score,
                ranking_mode="rerank",
            )
            previous = best_per_video.get(video_id)
            if previous is None or item["rerank_score"] > previous["rerank_score"]:
                best_per_video[video_id] = item

        reranked = sorted(
            best_per_video.values(),
            key=lambda item: (item["rerank_score"], item["vespa_score"]),
            reverse=True,
        )
        if not reranked:
            return [], []

        top_score = reranked[0]["rerank_score"]
        runner_up_score = reranked[1]["rerank_score"] if len(reranked) > 1 else float("-inf")
        use_rerank_order = (
            top_score >= VIDEO_RERANK_CONFIDENCE_THRESHOLD
            and top_score - runner_up_score >= VIDEO_RERANK_MARGIN_THRESHOLD
        )

        if use_rerank_order:
            candidates = reranked
            active_threshold = threshold
            ranking_mode = "rerank"
            for item in candidates:
                item["score"] = item["rerank_score"]
        else:
            candidates = sorted(
                best_per_video.values(),
                key=lambda item: (item["vespa_score"], item["rerank_score"]),
                reverse=True,
            )
            active_threshold = vespa_threshold
            ranking_mode = "vespa"
            for item in candidates:
                item["score"] = item["vespa_score"]

        for rank, item in enumerate(candidates, start=1):
            item["rank"] = rank
            item["ranking_mode"] = ranking_mode
        filtered = [item for item in candidates if item["score"] >= active_threshold]
        return filtered[:limit], candidates

    def _retrieve_videos_with_debug(
        self,
        query: str,
        query_embedding: list[float],
        org: str = "all",
        *,
        limit: int | None = None,
        target_hits: int | None = None,
        threshold: float | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if not self.enable_video_suggestions:
            return [], []
        final_k = limit or VIDEO_FINAL_K
        candidate_k = max(target_hits or VIDEO_RETRIEVE_K, final_k)
        threshold_is_explicit = threshold is not None
        if threshold is None:
            min_score = (
                VIDEO_RERANK_RELEVANCE_THRESHOLD
                if self.shared_reranker is not None
                else VIDEO_RELEVANCE_THRESHOLD
            )
        else:
            min_score = threshold
        try:
            hits = query_schema(
                self.vespa,
                schema="video",
                query_text=query,
                query_embedding=query_embedding,
                org=org,
                target_hits=candidate_k,
                hits=candidate_k,
            )
            return self._rank_video_hits(
                query,
                hits,
                limit=final_k,
                threshold=min_score,
                threshold_is_explicit=threshold_is_explicit,
            )
        except Exception as exc:
            logger.warning("Erreur recherche video Vespa: {}", exc)
            return [], []

    def _retrieve_videos(self, query: str, query_embedding: list[float], org: str = "all") -> list[dict]:
        results, _ = self._retrieve_videos_with_debug(
            query,
            query_embedding,
            org=org,
        )
        return results

    @staticmethod
    def _form_rerank_text(fields: dict) -> str:
        parts = [
            f"Formulaire : {fields.get('title', '')}",
            f"Categorie : {fields.get('category', '')}",
        ]
        content = str(fields.get("content", "") or "").strip()
        if content:
            parts.append(content[:600])
        return "\n".join(part for part in parts if part.strip())

    def _format_form_hit(self, hit, *, score: float, rerank_score: float | None = None, rank: int = 0) -> dict:
        fields = hit.fields
        return {
            "form_id": fields.get("form_id", ""),
            "title": fields.get("title", ""),
            "category": fields.get("category", ""),
            "pdf_url": fields.get("pdf_url", ""),
            "page_url": fields.get("page_url", ""),
            "org": fields.get("org", ""),
            "score": float(score),
            "vespa_score": float(hit.relevance),
            "rerank_score": float(rerank_score) if rerank_score is not None else None,
            "rank": rank,
        }

    def _rank_form_hits(
        self,
        query: str,
        hits: list,
        *,
        limit: int,
        threshold: float,
    ) -> tuple[list[dict], list[dict]]:
        if not hits:
            return [], []

        if self.shared_reranker is None:
            candidates = [
                self._format_form_hit(hit, score=hit.relevance, rank=index)
                for index, hit in enumerate(hits, start=1)
            ]
            filtered = [item for item in candidates if item["score"] >= threshold]
            return filtered[:limit], candidates

        pairs = [(query, self._form_rerank_text(hit.fields)) for hit in hits]
        try:
            scores = [float(score) for score in self.shared_reranker.predict(pairs).tolist()]
        except Exception as exc:
            logger.warning("Reranker erreur sur formulaires Vespa ({}), fallback score Vespa", exc)
            candidates = [
                self._format_form_hit(hit, score=hit.relevance, rank=index)
                for index, hit in enumerate(hits, start=1)
            ]
            filtered = [item for item in candidates if item["score"] >= threshold]
            return filtered[:limit], candidates

        best_per_form: dict[str, dict] = {}
        for hit, rerank_score in zip(hits, scores):
            item = self._format_form_hit(hit, score=rerank_score, rerank_score=rerank_score)
            form_id = item["form_id"]
            if not form_id:
                continue
            if form_id not in best_per_form or item["score"] > best_per_form[form_id]["score"]:
                best_per_form[form_id] = item

        candidates = sorted(best_per_form.values(), key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(candidates, start=1):
            item["rank"] = rank
        filtered = [item for item in candidates if item["score"] >= threshold]
        return filtered[:limit], candidates

    def _retrieve_forms_with_debug(
        self,
        query: str,
        query_embedding: list[float],
        org: str = "all",
        *,
        limit: int | None = None,
        target_hits: int | None = None,
        threshold: float | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if not self.enable_form_suggestions:
            return [], []
        final_k = limit or FORM_FINAL_K
        candidate_k = max(target_hits or FORM_RETRIEVE_K, final_k)
        if threshold is None:
            min_score = (
                FORM_RERANK_RELEVANCE_THRESHOLD
                if self.shared_reranker is not None
                else FORM_RELEVANCE_THRESHOLD
            )
        else:
            min_score = threshold
        try:
            hits = query_schema(
                self.vespa,
                schema="form",
                query_text=query,
                query_embedding=query_embedding,
                org=org,
                target_hits=candidate_k,
                hits=candidate_k,
            )
            return self._rank_form_hits(query, hits, limit=final_k, threshold=min_score)
        except Exception as exc:
            logger.warning("Erreur recherche formulaires Vespa: {}", exc)
            return [], []

    def _retrieve_forms(
        self,
        query: str,
        query_embedding: list[float],
        org: str = "all",
        *,
        limit: int | None = None,
        target_hits: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        results, _ = self._retrieve_forms_with_debug(
            query,
            query_embedding,
            org=org,
            limit=limit,
            target_hits=target_hits,
            threshold=threshold,
        )
        return results

    def retrieve_videos(
        self,
        query: str,
        org: str = "all",
        limit: int | None = None,
        *,
        target_hits: int | None = None,
        threshold: float | None = None,
        include_debug: bool = False,
    ) -> list[dict] | dict:
        cleaned = (query or "").strip()
        if not cleaned:
            return {"results": [], "candidates": []} if include_debug else []
        query_embedding = self.embeddings.embed_query(cleaned)
        results, candidates = self._retrieve_videos_with_debug(
            cleaned,
            query_embedding,
            org=org,
            limit=limit,
            target_hits=target_hits,
            threshold=threshold,
        )
        if include_debug:
            return {"results": results, "candidates": candidates}
        return results

    def retrieve_forms(
        self,
        query: str,
        org: str = "all",
        limit: int | None = None,
        *,
        target_hits: int | None = None,
        threshold: float | None = None,
        include_debug: bool = False,
    ) -> list[dict] | dict:
        cleaned = (query or "").strip()
        if not cleaned:
            return {"results": [], "candidates": []} if include_debug else []
        query_embedding = self.embeddings.embed_query(cleaned)
        results, candidates = self._retrieve_forms_with_debug(
            cleaned,
            query_embedding,
            org=org,
            limit=limit,
            target_hits=target_hits,
            threshold=threshold,
        )
        if include_debug:
            return {"results": results, "candidates": candidates}
        return results

    def _is_resource_only_query(self, query: str, resource: str) -> bool:
        normalized = self._normalize_text(query)
        words = normalized.strip().split()
        if len(words) > 8:
            return False
        return self._has_resource_signal(query, resource)

    def _has_resource_signal(self, query: str, resource: str) -> bool:
        normalized = self._normalize_text(query)
        if resource == "video":
            return any(
                token in normalized
                for token in (
                    " video ",
                    " videos ",
                    " youtube ",
                    " tutoriel video ",
                    " en video ",
                    " voir une video ",
                    " regarder une video ",
                )
            )
        if resource == "form":
            return any(
                token in normalized
                for token in (
                    "demande",
                    " formulaire ",
                    " formulaires ",
                    " imprimes ",
                    " imprime ",
                    " pdf ",
                    " telecharger ",
                    " document a remplir ",
                    " documents a remplir ",
                    " dossier a remplir ",
                    " dossier de demande ",
                    " papier a remplir ",
                    " papiers a fournir ",
                    " pieces a fournir ",
                    " documents a fournir ",
                )
            )
        return False

    def _classify_intent(self, query: str) -> dict:
        classification = {
            "intent": "retrieval",
            "confidence": 0.0,
            "reasoning": "IntentClassifier disabled"
            if not self.enable_intent_classifier
            else "IntentClassifier unavailable",
            "loaded": False,
        }

        if self.intent_classifier and self.intent_classifier.is_loaded:
            try:
                classification = self.intent_classifier.classify(query)
                logger.info(
                    "Intent: {} conf={:.3f}",
                    classification.get("intent", "retrieval"),
                    classification.get("confidence", 0.0),
                )
            except Exception as exc:
                logger.warning("Intent classification echouee: {} - fallback retrieval", exc)

        return classification

    def _simple_payload(
        self,
        response: str,
        query: str,
        org: str,
        classification: dict,
        intent: str | None = None,
    ) -> dict:
        return {
            "response": response,
            "context_docs": 0,
            "videos": [],
            "forms": [],
            "intent": intent or classification.get("intent", "retrieval"),
            "intent_confidence": classification.get("confidence", 0.0),
            "original_query": query,
            "cached": False,
            "org": org,
        }

    def _greeting_response(self, org: str) -> str:
        if org == "rcar":
            return "Bonjour. Je suis l'assistant RCAR. Posez-moi une question sur le RCAR ou ses procedures."
        if org == "cnra":
            return "Bonjour. Je suis l'assistant CNRA. Posez-moi une question sur la CNRA ou ses procedures."
        return "Bonjour. Je peux vous aider sur les informations RCAR et CNRA disponibles dans la base documentaire."

    def _should_short_circuit_out_of_scope(self, query: str, classification: dict) -> bool:
        intent = classification.get("intent", "retrieval")
        if intent == "out_of_scope":
            return True

        confidence = float(classification.get("confidence", 0.0) or 0.0)
        reasoning = str(classification.get("reasoning", "")).lower()
        if intent == "retrieval" and confidence < 0.60 and "out_of_scope" in reasoning:
            return True

        normalized = self._normalize_text(query)
        obvious_outside_terms = (
            " restaurant ",
            " hotel ",
            " meteo ",
            " football ",
            " match ",
            " recette ",
            " cuisine ",
            " voyage ",
            " billet avion ",
            " film ",
            " musique ",
        )
        return any(term in normalized for term in obvious_outside_terms)

    def _should_run_auxiliary_retriever(self, classification: dict, kind: str, query: str) -> bool:
        has_signal = self._has_resource_signal(query, kind)
        if self.intent_classifier is None or not self.intent_classifier.is_loaded:
            return has_signal

        confidence = float(classification.get("confidence", 0.0) or 0.0)
        threshold = getattr(self.intent_classifier, "gate_confidence_threshold", 0.60)
        if confidence < threshold:
            return False

        if kind == "video":
            return has_signal and self.intent_classifier.should_retrieve_videos(classification)
        if kind == "form":
            return has_signal and self.intent_classifier.should_retrieve_forms(classification)
        return False

    def query(
        self,
        query: str,
        org: str = "all",
        include_contexts: bool = False,
        use_intent_classifier: bool = True,
    ) -> dict:
        query_started_at = time.perf_counter()
        cleaned = (query or "").strip()
        if not cleaned:
            return self._error_response("Veuillez saisir une question.", query)

        offscope = self._detect_offscope_org(cleaned, org)
        if offscope:
            logger.info("Off-scope detecte (org={}, cible={})", org, offscope)
            return self._offscope_response(org, offscope, query)

        mode = "intent" if use_intent_classifier else "docs"
        cache_key = f"{org}:{mode}:{cleaned.lower()}"
        if hit := self._cache_get(cache_key):
            payload = {**hit, "original_query": query, "cached": True}
            timings = dict(payload.get("timings") or {})
            timings["cache_hit_seconds"] = round(time.perf_counter() - query_started_at, 6)
            payload["timings"] = timings
            if include_contexts:
                if self._has_context_payload(payload):
                    context_items = payload.get("context_items")
                    if isinstance(context_items, list):
                        payload["context_docs"] = len(context_items)
                else:
                    payload = {**payload, "contexts": [], "context_metadata": [], "context_items": []}
            return payload

        if use_intent_classifier:
            classification = self._classify_intent(cleaned)
            intent = classification.get("intent", "retrieval")

            if intent == "greeting":
                payload = self._simple_payload(self._greeting_response(org), query, org, classification)
                self._cache_set(cache_key, {k: v for k, v in payload.items() if k not in {"original_query", "cached"}})
                return payload

            if intent == "prompt_injection":
                payload = self._simple_payload(
                    "Je ne peux pas suivre cette demande. Je peux uniquement aider avec des informations officielles RCAR/CNRA a partir de la base documentaire.",
                    query,
                    org,
                    classification,
                )
                self._cache_set(cache_key, {k: v for k, v in payload.items() if k not in {"original_query", "cached"}})
                return payload

            if self._should_short_circuit_out_of_scope(cleaned, classification):
                payload = self._simple_payload(
                    "Cette question ne concerne pas le RCAR ou la CNRA. Je peux uniquement repondre aux questions liees a ces organismes et a leurs procedures.",
                    query,
                    org,
                    classification,
                    intent="out_of_scope",
                )
                self._cache_set(cache_key, {k: v for k, v in payload.items() if k not in {"original_query", "cached"}})
                return payload
        else:
            classification = {
                "intent": "retrieval",
                "confidence": 1.0,
                "reasoning": "intent classifier disabled",
            }

        run_videos = use_intent_classifier and self._should_run_auxiliary_retriever(classification, "video", cleaned)
        run_forms = use_intent_classifier and self._should_run_auxiliary_retriever(classification, "form", cleaned)
        resource_only_video = use_intent_classifier and self._is_resource_only_query(cleaned, "video")
        resource_only_form = use_intent_classifier and self._is_resource_only_query(cleaned, "form")

        if resource_only_video or resource_only_form:
            t_aux = time.perf_counter()
            videos, forms = [], []
            timings = {}
            try:
                t_embed = time.perf_counter()
                query_embedding = self.embeddings.embed_query(cleaned)
                timings["embedding_seconds"] = round(time.perf_counter() - t_embed, 6)
                t_resources = time.perf_counter()
                if resource_only_video and run_videos:
                    videos = self._retrieve_videos(cleaned, query_embedding, org)
                if resource_only_form and run_forms:
                    forms = self._retrieve_forms(cleaned, query_embedding, org)
                timings["resources_retrieval_seconds"] = round(time.perf_counter() - t_resources, 6)
                logger.info(
                    "Retrieval Vespa resources-only: videos={} (gate={}) formulaires={} (gate={}) en {:.3f}s",
                    len(videos),
                    "ON" if run_videos else "OFF",
                    len(forms),
                    "ON" if run_forms else "OFF",
                    time.perf_counter() - t_aux,
                )
            except Exception as exc:
                logger.error("Erreur retrieval ressources Vespa: {}", exc)
                return self._error_response("Erreur lors de la recherche de ressources.", query)

            if resource_only_video:
                response = (
                    "Des videos pertinentes sont affichees ci-dessous."
                    if videos
                    else "Aucune video pertinente n'a ete trouvee dans la base documentaire."
                )
                intent = classification.get("intent", "needs_video")
            else:
                response = (
                    "Les formulaires pertinents sont affiches ci-dessous."
                    if forms
                    else "Aucun formulaire pertinent n'a ete trouve dans la base documentaire."
                )
                intent = classification.get("intent", "needs_form")

            payload = {
                "response": response,
                "context_docs": 0,
                "videos": videos,
                "forms": forms,
                "intent": intent,
                "intent_confidence": classification.get("confidence", 0.0),
                "org": org,
            }
            timings["total_seconds"] = round(time.perf_counter() - query_started_at, 6)
            payload["timings"] = timings
            cache_payload = self._cache_payload(payload, [], include_contexts)
            self._cache_set(cache_key, cache_payload)
            return {**cache_payload, "original_query": query, "cached": False}

        if self.collection_count == 0:
            return self._error_response(
                "Base Vespa vide ou indisponible - lancez Vespa, deployez l'application, puis index_data.py.",
                query,
            )

        t0 = time.perf_counter()
        timings = {}
        try:
            t_embed = time.perf_counter()
            query_embedding = self.embeddings.embed_query(cleaned)
            timings["embedding_seconds"] = round(time.perf_counter() - t_embed, 6)
            t_docs = time.perf_counter()
            docs = self._retrieve_docs_vespa(cleaned, query_embedding, org=org)
            timings["docs_retrieval_seconds"] = round(time.perf_counter() - t_docs, 6)
            t_context = time.perf_counter()
            context = self._build_context(docs)
            timings["context_build_seconds"] = round(time.perf_counter() - t_context, 6)
            logger.info(
                "Retrieval Vespa docs: {} chunks (org={}) en {:.3f}s",
                len(docs),
                org,
                time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.error("Erreur retrieval Vespa: {}", exc)
            return self._error_response("Erreur lors de la recherche.", query)

        if not docs:
            return self._error_response("Aucun passage pertinent trouve.", query, context_docs=0)

        videos, forms = [], []

        t_aux = time.perf_counter()
        if run_videos:
            videos = self._retrieve_videos(cleaned, query_embedding, org)
        if run_forms:
            forms = self._retrieve_forms(cleaned, query_embedding, org)
        timings["resources_retrieval_seconds"] = round(time.perf_counter() - t_aux, 6)
        logger.info(
            "Retrieval Vespa: videos={} (gate={}) formulaires={} (gate={}) en {:.3f}s",
            len(videos),
            "ON" if run_videos else "OFF",
            len(forms),
            "ON" if run_forms else "OFF",
            time.perf_counter() - t_aux,
        )

        if videos and self._is_resource_only_query(cleaned, "video"):
            payload = {
                "response": "Des videos pertinentes sont affichees ci-dessous.",
                "context_docs": len(docs),
                "videos": videos,
                "forms": forms,
                "intent": classification.get("intent", "needs_video"),
                "intent_confidence": classification.get("confidence", 0.0),
                "org": org,
            }
            timings["total_seconds"] = round(time.perf_counter() - query_started_at, 6)
            payload["timings"] = timings
            cache_payload = self._cache_payload(payload, docs, include_contexts)
            self._cache_set(cache_key, cache_payload)
            return {**cache_payload, "original_query": query, "cached": False}

        if forms and self._is_resource_only_query(cleaned, "form"):
            payload = {
                "response": "Les formulaires pertinents sont affiches ci-dessous.",
                "context_docs": len(docs),
                "videos": videos,
                "forms": forms,
                "intent": classification.get("intent", "needs_form"),
                "intent_confidence": classification.get("confidence", 0.0),
                "org": org,
            }
            timings["total_seconds"] = round(time.perf_counter() - query_started_at, 6)
            payload["timings"] = timings
            cache_payload = self._cache_payload(payload, docs, include_contexts)
            self._cache_set(cache_key, cache_payload)
            return {**cache_payload, "original_query": query, "cached": False}

        t1 = time.perf_counter()
        try:
            response = str(
                self.chain.invoke(
                    {
                        "context": context,
                        "question": cleaned,
                        "supplementary_hint": self._build_supplementary_hint(videos, forms),
                        "org_identity": self._build_org_identity(org),
                    }
                )
            ).strip()
            timings["generation_seconds"] = round(time.perf_counter() - t1, 6)
            logger.info("Generation: {:.3f}s | {} chars", timings["generation_seconds"], len(response))
        except Exception as exc:
            logger.error("Erreur generation: {}", exc)
            return self._error_response(
                f"Erreur de generation. Verifiez qu'Ollama est lance (ollama pull {OLLAMA_MODEL}).",
                query,
                videos=videos,
                forms=forms,
                intent=classification.get("intent", "retrieval"),
                intent_confidence=classification.get("confidence", 0.0),
                context_docs=len(docs),
            )

        payload = {
            "response": response,
            "context_docs": len(docs),
            "videos": videos,
            "forms": forms,
            "intent": classification.get("intent", "retrieval"),
            "intent_confidence": classification.get("confidence", 0.0),
            "org": org,
        }
        timings["total_seconds"] = round(time.perf_counter() - query_started_at, 6)
        payload["timings"] = timings
        cache_payload = self._cache_payload(payload, docs, include_contexts)
        self._cache_set(cache_key, cache_payload)
        return {**cache_payload, "original_query": query, "cached": False}


if __name__ == "__main__":
    setup_logger(log_dir=LOGS_DIR, source="pipeline")
    rag = RAGPipeline()
    print(rag.query("comment faire une demande de pension de retraite RCAR ?"))
