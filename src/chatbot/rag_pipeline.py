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
HF_TOKEN = getenv("HF_TOKEN", "").strip().strip('"\'')

OLLAMA_MODEL = getenv("OLLAMA_MODEL", "mistral:latest").strip().strip('"\'')
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

VIDEO_RETRIEVE_K = 10
VIDEO_FINAL_K = 2
FORM_RETRIEVE_K = 10
FORM_FINAL_K = 1
VIDEO_RELEVANCE_THRESHOLD = 0.185
FORM_RELEVANCE_THRESHOLD = 0.220

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
    ):
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        self.vespa = make_vespa_app(vespa_url, vespa_port)
        self.enable_video_suggestions = enable_video_suggestions
        self.enable_form_suggestions = enable_form_suggestions
        self.retrieval_k = RAG_RETRIEVE_K
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

        self.intent_classifier: IntentClassifier | None = None
        if _INTENT_CLASSIFIER_AVAILABLE:
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

    def _retrieve_docs_vespa(self, query: str, query_embedding: list[float], org: str = "all") -> list[Document]:
        hits = query_schema(
            self.vespa,
            schema="doc",
            query_text=query,
            query_embedding=query_embedding,
            org=org,
            target_hits=self.retrieval_k,
            hits=self.final_k,
        )
        docs = []
        for hit in hits:
            fields = hit.fields
            docs.append(
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
                    },
                )
            )
        return docs

    def _retrieve_videos(self, query: str, query_embedding: list[float], org: str = "all") -> list[dict]:
        if not self.enable_video_suggestions:
            return []
        try:
            hits = query_schema(
                self.vespa,
                schema="video",
                query_text=query,
                query_embedding=query_embedding,
                org=org,
                target_hits=VIDEO_RETRIEVE_K,
                hits=VIDEO_RETRIEVE_K,
            )
            hits = [h for h in hits if h.relevance >= VIDEO_RELEVANCE_THRESHOLD][:VIDEO_FINAL_K]
            return [
                {
                    "video_id": h.fields.get("video_id", ""),
                    "title": h.fields.get("title", ""),
                    "url": h.fields.get("url", ""),
                    "thumbnail_url": h.fields.get("thumbnail_url", ""),
                    "upload_date": h.fields.get("upload_date", ""),
                    "org": h.fields.get("org", ""),
                    "score": h.relevance,
                }
                for h in hits
            ]
        except Exception as exc:
            logger.warning("Erreur recherche video Vespa: {}", exc)
            return []

    def _retrieve_forms(self, query: str, query_embedding: list[float], org: str = "all") -> list[dict]:
        if not self.enable_form_suggestions:
            return []
        try:
            hits = query_schema(
                self.vespa,
                schema="form",
                query_text=query,
                query_embedding=query_embedding,
                org=org,
                target_hits=FORM_RETRIEVE_K,
                hits=FORM_RETRIEVE_K,
            )
            hits = [h for h in hits if h.relevance >= FORM_RELEVANCE_THRESHOLD][:FORM_FINAL_K]
            return [
                {
                    "form_id": h.fields.get("form_id", ""),
                    "title": h.fields.get("title", ""),
                    "category": h.fields.get("category", ""),
                    "pdf_url": h.fields.get("pdf_url", ""),
                    "page_url": h.fields.get("page_url", ""),
                    "org": h.fields.get("org", ""),
                    "score": h.relevance,
                }
                for h in hits
            ]
        except Exception as exc:
            logger.warning("Erreur recherche formulaires Vespa: {}", exc)
            return []

    def _is_resource_only_query(self, query: str, resource: str) -> bool:
        normalized = self._normalize_text(query)
        words = normalized.strip().split()
        if len(words) > 8:
            return False
        if resource == "video":
            return any(token in normalized for token in (" video ", " videos ", " youtube "))
        if resource == "form":
            return any(token in normalized for token in (" formulaire ", " formulaires ", " imprimes ", " imprime "))
        return False

    def _classify_intent(self, query: str) -> dict:
        classification = {
            "intent": "retrieval",
            "confidence": 0.0,
            "tier": 1,
            "reasoning": "IntentClassifier unavailable",
            "loaded": False,
        }

        if self.intent_classifier and self.intent_classifier.is_loaded:
            try:
                classification = self.intent_classifier.classify(query)
                logger.info(
                    "Intent: {} [tier {}] conf={:.3f}",
                    classification.get("intent", "retrieval"),
                    classification.get("tier", 1),
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
        if self.intent_classifier is None or not self.intent_classifier.is_loaded:
            return True

        confidence = float(classification.get("confidence", 0.0) or 0.0)
        threshold = getattr(self.intent_classifier, "gate_confidence_threshold", 0.60)
        if classification.get("intent") == "retrieval" and confidence < threshold:
            return self._is_resource_only_query(query, kind)

        if kind == "video":
            return self.intent_classifier.should_retrieve_videos(classification)
        if kind == "form":
            return self.intent_classifier.should_retrieve_forms(classification)
        return False

    def query(self, query: str, org: str = "all") -> dict:
        cleaned = (query or "").strip()
        if not cleaned:
            return self._error_response("Veuillez saisir une question.", query)

        offscope = self._detect_offscope_org(cleaned, org)
        if offscope:
            logger.info("Off-scope detecte (org={}, cible={})", org, offscope)
            return self._offscope_response(org, offscope, query)

        cache_key = f"{org}:{cleaned.lower()}"
        if hit := self._cache_get(cache_key):
            return {**hit, "original_query": query, "cached": True}

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

        if self.collection_count == 0:
            return self._error_response(
                "Base Vespa vide ou indisponible - lancez Vespa, deployez l'application, puis index_data.py.",
                query,
            )

        t0 = time.perf_counter()
        try:
            query_embedding = self.embeddings.embed_query(cleaned)
            docs = self._retrieve_docs_vespa(cleaned, query_embedding, org=org)
            context = self._build_context(docs)
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
        run_videos = self._should_run_auxiliary_retriever(classification, "video", cleaned)
        run_forms = self._should_run_auxiliary_retriever(classification, "form", cleaned)

        t_aux = time.perf_counter()
        if run_videos:
            videos = self._retrieve_videos(cleaned, query_embedding, org)
        if run_forms:
            forms = self._retrieve_forms(cleaned, query_embedding, org)
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
            self._cache_set(cache_key, payload)
            return {**payload, "original_query": query, "cached": False}

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
            self._cache_set(cache_key, payload)
            return {**payload, "original_query": query, "cached": False}

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
            logger.info("Generation: {:.3f}s | {} chars", time.perf_counter() - t1, len(response))
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
        self._cache_set(cache_key, payload)
        return {**payload, "original_query": query, "cached": False}


if __name__ == "__main__":
    setup_logger(log_dir=LOGS_DIR, source="pipeline")
    rag = RAGPipeline()
    print(rag.query("comment faire une demande de pension de retraite RCAR ?"))
