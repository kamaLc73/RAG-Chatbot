import os
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from os import getenv

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message=r"Accessing `__path__` from `\.models\..*",)
warnings.filterwarnings("ignore", category=FutureWarning)

from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from loguru import logger

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


load_dotenv(BASE_DIR / ".env")

VECTORSTORE_RELATIVE_PATH = Path("data") / "vectorstore" / "chroma_db"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
HF_TOKEN = getenv("HF_TOKEN", "").strip().strip('"\'')

OLLAMA_MODEL       = getenv("OLLAMA_MODEL", "mistral:latest").strip().strip('"\'')
OLLAMA_BASE_URL    = getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY     = getenv("OLLAMA_API_KEY", "").strip().strip('"\'')
OLLAMA_TEMPERATURE = 0.1       # Moins de créativité = moins de "remplissage" inventé
OLLAMA_NUM_PREDICT = 600       # Suffisant pour une réponse structurée et précise
OLLAMA_NUM_CTX     = 4096      # Confortable : prompt + contexte + réponse sans coupure
OLLAMA_KEEP_ALIVE  = "30m"

RAG_TOP_K             = 4      # 4 chunks bien choisis > 5 chunks dont un hors sujet
RAG_MAX_CONTEXT_CHARS = 2800   # Cohérent avec NUM_CTX : laisse ~270 tokens pour la réponse
RAG_MAX_DOC_CHARS     = 700    # Chunks plus courts = plus de diversité dans le contexte
RAG_CACHE_SIZE        = 100    # OK, à garder



class RAGPipeline:
    def __init__(
        self,
        local_model: str = OLLAMA_MODEL,
        vectorstore_path: Path | str = VECTORSTORE_RELATIVE_PATH,
        collection_name: str = "rcar_cnra_fr",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        resolved_path = Path(vectorstore_path)
        if not resolved_path.is_absolute():
            resolved_path = (BASE_DIR / resolved_path).resolve()

        self.collection_name  = collection_name
        self.retrieval_k      = RAG_TOP_K
        self.max_context_chars = RAG_MAX_CONTEXT_CHARS
        self.max_doc_chars    = RAG_MAX_DOC_CHARS
        self.response_cache: OrderedDict = OrderedDict()

        logger.info("VectorStore: {} (collection={})", resolved_path, collection_name)

        # ── Embeddings ────────────────────────────────────────────────────────
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        logger.info("Embedding: {} sur {}", embedding_model, device)
        model_kwargs = {"device": device}
        if HF_TOKEN:
            model_kwargs["token"] = HF_TOKEN

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._warm_up_embeddings()

        # ── VectorStore ───────────────────────────────────────────────────────
        self.vectorstore = Chroma(
            persist_directory=str(resolved_path),
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        self.collection_count = self._get_collection_count(resolved_path, collection_name)
        logger.info("Collection '{}': {} chunks", collection_name, self.collection_count)
        if self.collection_count == 0:
            logger.warning("Collection vide — lancez index_data.py pour l'indexation.")

        # ── LLM ───────────────────────────────────────────────────────────────
        is_cloud = "ollama.com" in OLLAMA_BASE_URL
        client_kwargs = (
            {"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}} if OLLAMA_API_KEY else None
        )
        if is_cloud and not OLLAMA_API_KEY:
            logger.warning("ollama.com détecté sans OLLAMA_API_KEY — authentification requise.")

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
        except Exception as exc:
            logger.warning("Compteur collection indisponible: {}", exc)
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
    - Texte brut uniquement, sans aucune syntaxe Markdown (pas de **, ##, __, backticks, ni tirets triples).
    - Respectez les regles normales de capitalisation du francais :
        * Chaque phrase commence par une majuscule.
        * Les noms propres et sigles (CNRA, RCAR, CDG, Maroc) gardent leurs majuscules.
        * Le reste du texte est en minuscules normales.
    - Reponses structurees en phrases courtes et claires.
    - Pour les listes, commencez chaque element par "- " sur une nouvelle ligne.
    - Pour les titres de section, ecrivez la premiere lettre en majuscule, suivie de deux-points. Exemple : "Missions principales :"
    - Longueur adaptee a la question : courte si la question est simple, detaillee si elle est complexe.

    EXEMPLE DE BONNE REPONSE:
    La CNRA est un etablissement public marocain.

    Missions principales :
    - Gestion des rentes d accidents du travail.
    - Emission de produits d assurance-vie.
    FIN DE L EXEMPLE.
    CONTEXTE DOCUMENTAIRE:
    {context}"""

    def _warm_up_embeddings(self) -> None:
        try:
            t = time.perf_counter()
            self.embeddings.embed_query("verification initiale")
            logger.info("Warmup embeddings: {:.3f}s", time.perf_counter() - t)
        except Exception as exc:
            logger.warning("Warmup ignoré: {}", exc)

    def _build_context(self, docs) -> str:
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

    # ── API publique ──────────────────────────────────────────────────────────

    def query(self, query: str) -> dict:
        """Traite une question et retourne la réponse + métadonnées."""
        cleaned = (query or "").strip()
        if not cleaned:
            return {"response": "Veuillez saisir une question.", "error": True}

        if cached := self._cache_get(cleaned.lower()):
            logger.info("Cache hit")
            return {**cached, "original_query": query, "cached": True}

        if self.collection_count == 0:
            return {
                "response": "Base vectorielle vide — lancez index_data.py puis réessayez.",
                "error": True,
            }

        # Retrieval
        t0 = time.perf_counter()
        try:
            docs = self.vectorstore.similarity_search(cleaned, k=self.retrieval_k)
            context = self._build_context(docs)
            logger.info("Retrieval: {} docs en {:.3f}s", len(docs), time.perf_counter() - t0)

            # ── Debug chunks ──────────────────────────────────────────────────────
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "source inconnue")
                preview = (doc.page_content or "").replace("\n", " ").strip()[:150]
                logger.debug("Chunk {}/{} | source={} | apercu: {}...", i, len(docs), source, preview)
            logger.debug("Contexte total envoye au LLM: {} chars", len(context))
            # ─────────────────────────────────────────────────────────────────────

        except Exception as exc:
            logger.error("Erreur retrieval: {}", exc)
            return {"response": "Erreur lors de la recherche vectorielle.", "error": True}

        if not docs:
            return {"response": "Aucun passage pertinent trouvé.", "context_docs": 0}

        # Generation
        t1 = time.perf_counter()
        try:
            response = str(self.chain.invoke({"context": context, "question": cleaned})).strip()
            logger.info("Generation: {:.3f}s | {} chars", time.perf_counter() - t1, len(response))
        except Exception as exc:
            logger.error("Erreur génération: {}", exc)
            return {
                "response": (
                    "Erreur de génération. Vérifiez qu'Ollama est lancé "
                    f"(ollama pull {OLLAMA_MODEL})."
                ),
                "error": True,
            }

        payload = {"response": response, "context_docs": len(docs)}
        self._cache_set(cleaned.lower(), payload)
        return {**payload, "original_query": query, "cached": False}


if __name__ == "__main__":
    setup_logger(log_dir=LOGS_DIR, source="pipeline")
    rag = RAGPipeline()
    logger.info(rag.query("c'est quoi CNRA ?"))