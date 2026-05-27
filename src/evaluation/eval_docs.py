"""
Evaluation documentaire RAGAS du chatbot RCAR/CNRA.

Objectif:
  - Evaluer uniquement la tache `doc`.
  - Utiliser le golden dataset documentaire.
  - Collecter les reponses du RAG et les chunks documentaires recuperes.
  - Calculer les 6 metriques RAGAS classiques.
  - Sauvegarder les artefacts JSON/CSV/Markdown pour comparer ChromaDB et Vespa.

Modes de collecte:
  - local: instancie directement chatbot.rag_pipeline.RAGPipeline.

Exemples:
  .\\.venv\\Scripts\\python.exe src\\evaluation\\eval_docs.py --run-name docs_smoke --limit 3 --skip-ragas
  .\\.venv\\Scripts\\python.exe src\\evaluation\\eval_docs.py --run-name docs_chroma

Le juge RAGAS utilise Cerebras par defaut avec RAGAS_CEREBRAS_MODEL, ou gpt-oss-120b.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
EVALUATION_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from eval_common import (
    DEFAULT_EMBEDDING_MODEL,
    configure_logging,
    get_git_metadata,
    latency_stats,
    make_run_dir,
    org_for,
    safe_mean,
    write_csv,
    write_jsonl,
)

EVALUATION_DATA_DIR = PROJECT_ROOT / "data" / "evaluation"
RUNS_DIR = EVALUATION_DATA_DIR / "runs"
DATA_FILE_DEFAULT = EVALUATION_DATA_DIR / "docs_test_set_48.jsonl"
LEGACY_DATA_FILE = EVALUATION_DATA_DIR / "docs_test_set_29.jsonl"
SUPPORT_DATA_DIR = PROJECT_ROOT / "data" / "supportstagerag"
DOC_SOURCE_TYPES = ("faq", "web", "bibliotheque")
ORG_LABELS = {"RCAR": "le RCAR", "CNRA": "la CNRA"}
COMMON_DOMAIN_KEYWORDS = (
    "affiliation",
    "assurance",
    "beneficiaire",
    "capital",
    "cotisation",
    "demande",
    "dossier",
    "formulaire",
    "pension",
    "prestation",
    "procedure",
    "rachat",
    "rente",
    "retraite",
)
ORG_DOMAIN_KEYWORDS = {
    "RCAR": (
        "rcar",
        "affilie",
        "recouvrement",
        "regime collectif",
        "regime complementaire",
        "regime general",
    ),
    "CNRA": (
        "accident",
        "addamane",
        "cnra",
        "dwayer",
        "fram",
        "rac",
        "rat",
        "recore",
        "rente",
        "solidarite",
        "travail",
        "zemane",
    ),
}
LOW_VALUE_SOURCE_KEYWORDS = (
    "actualite",
    "aid adha",
    "blanchiment",
    "charte commune des portails",
    "charte des services publics",
    "code ethique",
    "deontologie",
    "droit d acces a l information",
    "donnees personnelles",
    "echange des donnees juridiques",
    "espace media",
    "flux rss",
    "formulaire piece",
    "galerie documentaire",
    "liens utiles",
    "marches publics",
    "mention legale",
    "nous rejoindre",
    "point de contact",
    "politique de management",
    "politique management",
)
GENERIC_TITLE_PATTERNS = (
    r"^a decide",
    r"^article\s+",
    r"^chapitre\s+",
    r"^dispositions?\s+",
    r"^galerie documentaire$",
    r"^introduction$",
    r"^contact$",
    r"^le directeur",
    r"^le ministre",
    r"^le premier ministre",
    r"^livre\s+",
    r"^louange",
    r"^objet$",
    r"^page\s+\d+",
    r"^section\s+",
    r"^sommaire$",
    r"^table des matieres$",
    r"^texte",
    r"^titre\s+",
)

DEFAULT_CEREBRAS_JUDGE_MODEL = os.getenv("RAGAS_CEREBRAS_MODEL", "gpt-oss-120b")
DEFAULT_CEREBRAS_BASE_URL = os.getenv("RAGAS_CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
DEFAULT_CEREBRAS_REQUESTS_PER_MINUTE = float(os.getenv("RAGAS_CEREBRAS_REQUESTS_PER_MINUTE", "2"))
DEFAULT_CEREBRAS_MAX_COMPLETION_TOKENS = int(os.getenv("RAGAS_CEREBRAS_MAX_COMPLETION_TOKENS", "1536"))
DEFAULT_CEREBRAS_API_KEY_ENV = os.getenv("RAGAS_CEREBRAS_API_KEY_ENV", "CEREBRAS_API_KEY_1")

logger = logging.getLogger(__name__)


def get_api_key_from_env(env_name: str) -> str:
    api_key = os.getenv(env_name, "").strip().strip('"\'')
    if not api_key:
        raise RuntimeError(f"{env_name} is required for RAGAS evaluation via Cerebras.")
    return api_key


def load_golden(data_file: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset introuvable: {data_file}")
    with data_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().strip("*:：")


def normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    no_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", no_accents.lower()).strip()


def infer_source_type(path: str | Path) -> str:
    parts = [part.lower() for part in Path(path).parts]
    for source_type in DOC_SOURCE_TYPES:
        if source_type in parts:
            return source_type
    return "unknown"


def humanize_title(value: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value or "")
    text = re.sub(r"(?i)^question\s+\d+\s*[:.-]\s*", "", text.strip())
    text = re.sub(r"[*#`]+", "", text)
    text = re.sub(r"(?i)^question\s+\d+\s*[:.-]\s*", "", text.strip())
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]


def is_generic_title(title: str) -> bool:
    text = normalize_for_matching(title)
    if len(text) < 4:
        return True
    if re.fullmatch(r"[\d\s./-]+", text):
        return True
    return any(re.search(pattern, text) for pattern in GENERIC_TITLE_PATTERNS)


def clean_answer(text: str, max_chars: int = 900) -> str:
    cleaned = re.sub(r"=+\s*PAGE\s+\d+\s*=+", " ", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = re.sub(r"^\*{0,2}R\s*:\*{0,2}\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:max_chars].strip()


def fallback_title(path: Path, headings: list[tuple[int, str]], lines: list[str]) -> str:
    for _, title in headings:
        if not is_generic_title(title):
            return title
    for line in lines[:30]:
        title = humanize_title(line.strip())
        if not is_generic_title(title):
            return title
    return humanize_title(path.stem)


def extract_faq_pairs(path: Path, organization: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"(?im)^(?:#{1,4}\s*)?(?:\*\*)?\s*Q\s*[:：]\s*(?:\*\*)?(.*?)(?:\*\*)?\s*$"
    )
    matches = list(pattern.finditer(text))
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        question = normalize_question(match.group(1))
        if not question or len(question) < 12:
            continue
        answer_start = match.end()
        answer_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        answer = clean_answer(text[answer_start:answer_end])
        if len(answer) < 40:
            continue
        rows.append(
            {
                "question": question,
                "expected_answer": answer,
                "category": infer_doc_category(question, answer),
                "organization": organization,
                "source_type": "faq",
                "source_doc": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )
    return rows


def infer_doc_category(question: str, answer: str) -> str:
    text = normalize_for_matching(f"{question} {answer}")
    if any(token in text for token in ("formulaire", "demande", "dossier", "piece", "documents", "procedure", "comment proceder")):
        return "Procédure Officielle"
    if any(token in text for token in ("pension", "retraite", "cotisation", "affiliation", "rachat", "rente", "capital")):
        return "Pension/Cotisation/Éligibilité"
    if any(token in text for token in ("adresse", "coordonnees", "mot de passe", "espace", "mise a jour", "attestation")):
        return "Gestion Administrative"
    return "Concepts & Définitions"


def extract_section_rows(path: Path, organization: str, source_type: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    heading_pattern = re.compile(r"^\s{0,3}#{1,4}\s+(.+?)\s*$")
    org_label = ORG_LABELS.get(organization, organization)
    headings = [
        (index, humanize_title(match.group(1).strip("* ")))
        for index, line in enumerate(lines)
        if (match := heading_pattern.match(line))
    ]

    rows: list[dict[str, Any]] = []
    if headings:
        for index, (line_index, title) in enumerate(headings):
            if title.lower() in {"faq", "foire aux questions"} or is_generic_title(title):
                continue
            next_line = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
            answer = clean_answer("\n".join(lines[line_index + 1:next_line]))
            if len(answer) < 140:
                continue
            rows.append(
                {
                    "question": f"Pour {org_label}, que précise le document officiel concernant {title} ?",
                    "expected_answer": answer,
                    "category": infer_doc_category(title, answer),
                    "organization": organization,
                    "source_type": source_type,
                    "source_doc": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                }
            )

    if rows:
        return rows

    cleaned = clean_answer(text, max_chars=900)
    if len(cleaned) < 180:
        return []
    title = fallback_title(path, headings, lines)
    return [
        {
            "question": f"Pour {org_label}, quelles informations officielles sont fournies dans {title} ?",
            "expected_answer": cleaned,
            "category": infer_doc_category(title, cleaned),
            "organization": organization,
            "source_type": source_type,
            "source_doc": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
    ]


def collect_source_rows(organization: str, source_type: str) -> list[dict[str, Any]]:
    org_dir = SUPPORT_DATA_DIR / organization.lower() / source_type
    rows: list[dict[str, Any]] = []
    for path in sorted(org_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            if source_type == "faq":
                rows.extend(extract_faq_pairs(path, organization))
            else:
                rows.extend(extract_section_rows(path, organization, source_type))
    return rows


def candidate_score(row: dict[str, Any], organization: str, source_type: str) -> int:
    text = normalize_for_matching(
        " ".join(str(row.get(key, "")) for key in ("question", "expected_answer", "category", "source_doc"))
    )
    source_doc = normalize_for_matching(str(row.get("source_doc", "")))
    score = 0
    if source_type == "faq":
        score += 20
    if organization.lower() in text:
        score += 8
    for keyword in ORG_DOMAIN_KEYWORDS.get(organization, ()):
        if keyword in text:
            score += 4
    for keyword in COMMON_DOMAIN_KEYWORDS:
        if keyword in text:
            score += 2
    for keyword in LOW_VALUE_SOURCE_KEYWORDS:
        if keyword in text:
            score -= 8
    for keyword in ORG_DOMAIN_KEYWORDS.get("CNRA" if organization == "RCAR" else "RCAR", ()):
        if keyword in text:
            score -= 4
    if source_type == "web" and " faq " in f" {source_doc} ":
        score -= 10
    answer_len = len(str(row.get("expected_answer", "")))
    if answer_len >= 250:
        score += 2
    if answer_len < 160:
        score -= 3
    return score


def make_starter_dataset(path: Path, per_org: int = 25, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset existe deja: {path}. Utilise --overwrite pour le remplacer.")

    selected: list[dict[str, Any]] = []
    legacy_seen_questions: set[str] = set()

    if LEGACY_DATA_FILE.exists():
        for row in load_golden(LEGACY_DATA_FILE):
            org = str(row.get("organization", "")).upper()
            if org not in {"RCAR", "CNRA"}:
                continue
            key = normalize_question(row.get("question", "")).lower()
            if key in legacy_seen_questions:
                continue
            row = {**row, "source_type": infer_source_type(row.get("source_doc", ""))}
            selected.append(row)
            legacy_seen_questions.add(key)

    final_rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for organization in ("RCAR", "CNRA"):
        base_quota = per_org // len(DOC_SOURCE_TYPES)
        quotas = {source_type: base_quota for source_type in DOC_SOURCE_TYPES}
        for source_type in DOC_SOURCE_TYPES[: per_org % len(DOC_SOURCE_TYPES)]:
            quotas[source_type] += 1

        org_rows: list[dict[str, Any]] = []
        local_seen: set[str] = set()
        for source_type in DOC_SOURCE_TYPES:
            source_rows = [
                row
                for row in selected
                if str(row.get("organization", "")).upper() == organization and row.get("source_type") == source_type
            ]
            source_rows.extend(collect_source_rows(organization, source_type))
            source_rows.sort(
                key=lambda row: (
                    candidate_score(row, organization, source_type),
                    normalize_question(row.get("question", "")),
                ),
                reverse=True,
            )

            source_selected: list[dict[str, Any]] = []
            for row in source_rows:
                key = normalize_question(row.get("question", "")).lower()
                if key in local_seen or key in seen_questions:
                    continue
                source_selected.append(row)
                local_seen.add(key)
                seen_questions.add(key)
                if len(source_selected) == quotas[source_type]:
                    break
            if len(source_selected) < quotas[source_type]:
                raise RuntimeError(
                    f"Impossible de trouver {quotas[source_type]} questions {source_type} pour {organization}; "
                    f"trouve {len(source_selected)}."
                )
            org_rows.extend(source_selected)

        final_rows.extend(org_rows[:per_org])

    write_jsonl(path, final_rows)
    counts = Counter(row.get("organization") for row in final_rows)
    source_counts = Counter((row.get("organization"), row.get("source_type")) for row in final_rows)
    logger.info("Dataset docs ecrit: %d lignes -> %s | org=%s sources=%s", len(final_rows), path, dict(counts), dict(source_counts))


def sample_per_org(rows: list[dict[str, Any]], per_org: int | None) -> list[dict[str, Any]]:
    if not per_org:
        return rows

    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        org = org_for(row)
        if org not in {"rcar", "cnra"}:
            continue
        if counts[org] >= per_org:
            continue
        selected.append(row)
        counts[org] += 1
        if counts["rcar"] >= per_org and counts["cnra"] >= per_org:
            break

    if counts["rcar"] < per_org or counts["cnra"] < per_org:
        raise RuntimeError(
            f"Impossible de construire un sample equilibre {per_org}/{per_org}; "
            f"trouve rcar={counts['rcar']} cnra={counts['cnra']}."
        )
    return selected


def serialize_doc(doc: Any) -> dict[str, Any]:
    return {
        "text": getattr(doc, "page_content", "") or "",
        "metadata": dict(getattr(doc, "metadata", {}) or {}),
    }


def retrieve_context_documents(rag: Any, question: str, org: str) -> list[Any]:
    """Best-effort context retrieval for the Vespa branch and older Chroma-style branches."""
    if hasattr(rag, "_retrieve_docs_vespa") and hasattr(rag, "embeddings"):
        query_embedding = rag.embeddings.embed_query(question)
        return rag._retrieve_docs_vespa(question, query_embedding, org=org)

    if hasattr(rag, "retrieve_documents"):
        try:
            return rag.retrieve_documents(question, org=org)
        except TypeError:
            return rag.retrieve_documents(question)

    store = None
    for attr in ("vectorstore", "db", "collection", "store"):
        candidate = getattr(rag, attr, None)
        if candidate is not None and hasattr(candidate, "similarity_search"):
            store = candidate
            break

    if store is None:
        return []

    filter_arg = None
    if hasattr(rag, "_build_org_filter"):
        try:
            filter_arg = rag._build_org_filter(org, "doc")
        except TypeError:
            try:
                filter_arg = rag._build_org_filter(org)
            except TypeError:
                filter_arg = None

    k = int(getattr(rag, "final_k", getattr(rag, "retrieval_k", 4)))
    kwargs: dict[str, Any] = {"k": k}
    if filter_arg is not None:
        kwargs["filter"] = filter_arg
    try:
        return store.similarity_search(question, **kwargs)
    except TypeError:
        return store.similarity_search(question, k=k)


def query_local_rag(rag: Any, question: str, org: str) -> dict[str, Any]:
    contexts_docs: list[Any] = []
    started = time.perf_counter()
    try:
        try:
            payload = rag.query(
                question,
                org=org,
                include_contexts=True,
                use_intent_classifier=False,
                include_resources=False,
            )
        except TypeError:
            contexts_docs = retrieve_context_documents(rag, question, org)
            try:
                payload = rag.query(question, org=org)
            except TypeError:
                payload = rag.query(question)
        latency = time.perf_counter() - started
    except Exception as exc:
        return {
            "answer": "",
            "contexts": [],
            "context_metadata": [],
            "latency_seconds": time.perf_counter() - started,
            "error": repr(exc),
            "raw": {},
        }

    raw_contexts = payload.get("contexts") or payload.get("retrieved_contexts") or []
    contexts = raw_contexts if raw_contexts else [serialize_doc(doc)["text"] for doc in contexts_docs]
    metadata = payload.get("context_metadata") or [serialize_doc(doc)["metadata"] for doc in contexts_docs]

    return {
        "answer": payload.get("response", payload.get("answer", "")),
        "contexts": contexts,
        "context_metadata": metadata,
        "latency_seconds": latency,
        "error": payload.get("error", ""),
        "raw": payload,
    }


def collect_local_responses(questions: list[dict[str, Any]], delay: float = 0.0) -> list[dict[str, Any]]:
    from chatbot.rag_pipeline import RAGPipeline

    rag = RAGPipeline(
        enable_intent_classifier=False,
        enable_video_suggestions=False,
        enable_form_suggestions=False,
    )
    results: list[dict[str, Any]] = []
    for index, row in enumerate(questions, start=1):
        question = row["question"]
        org = org_for(row)
        logger.info("[%d/%d] local org=%s | %s", index, len(questions), org, question[:90])
        resp = query_local_rag(rag, question, org)
        results.append(build_result_row(row, org, resp))
        logger.info("  -> %.2fs | contexts=%d | error=%s", resp["latency_seconds"], len(resp["contexts"]), bool(resp["error"]))
        if delay > 0 and index < len(questions):
            time.sleep(delay)
    return results


def build_result_row(row: dict[str, Any], org: str, resp: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row["question"],
        "answer": resp["answer"],
        "contexts": resp["contexts"],
        "context_metadata": resp["context_metadata"],
        "ground_truth": row["expected_answer"],
        "latency_seconds": resp["latency_seconds"],
        "category": row.get("category", ""),
        "organization": row.get("organization", ""),
        "org": org,
        "source_type": row.get("source_type", ""),
        "source_doc": row.get("source_doc", ""),
        "error": resp["error"],
        "raw": resp["raw"],
    }


def build_ragas_dataset(results: list[dict[str, Any]]):
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    source_rows = []
    skipped = 0
    for result in results:
        if result.get("error") or not result.get("answer") or not result.get("contexts"):
            skipped += 1
            continue
        source_rows.append(result)
        samples.append(
            SingleTurnSample(
                user_input=result["question"],
                retrieved_contexts=result["contexts"],
                response=result["answer"],
                reference=result["ground_truth"],
            )
        )
    if not samples:
        raise RuntimeError("Aucune reponse RAG exploitable pour RAGAS.")
    logger.info("RAGAS dataset: %d samples valides, %d skipped", len(samples), skipped)
    return EvaluationDataset(samples=samples), skipped, source_rows


def setup_judge(
    judge_model: str,
    judge_base_url: str,
    judge_requests_per_minute: float,
    judge_max_completion_tokens: int,
    judge_api_key_env: str,
):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    judge_api_key = get_api_key_from_env(judge_api_key_env)

    requests_per_second = max(judge_requests_per_minute, 0.1) / 60.0
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.5,
        max_bucket_size=1,
    )
    llm = ChatOpenAI(
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        temperature=0.0,
        max_completion_tokens=judge_max_completion_tokens,
        timeout=120,
        max_retries=2,
        rate_limiter=rate_limiter,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def run_ragas(
    dataset: Any,
    judge_model: str,
    judge_base_url: str,
    judge_requests_per_minute: float,
    judge_max_completion_tokens: int,
    judge_api_key_env: str,
    max_workers: int,
    timeout: int,
    max_wait: int,
):
    from ragas import evaluate
    from ragas.run_config import RunConfig

    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
        context_recall,
        faithfulness,
    )

    if hasattr(answer_relevancy, "strictness"):
        answer_relevancy.strictness = 1

    llm_wrapper, emb_wrapper = setup_judge(
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_requests_per_minute=judge_requests_per_minute,
        judge_max_completion_tokens=judge_max_completion_tokens,
        judge_api_key_env=judge_api_key_env,
    )
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity,
    ]
    run_config = RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_retries=3,
        max_wait=max_wait,
    )
    return evaluate(
        dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=emb_wrapper,
        run_config=run_config,
        raise_exceptions=False,
    )


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    excluded = {
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
        "question",
        "org",
        "organization",
        "category",
        "source_type",
        "source_doc",
    }
    for column in {key for row in rows for key in row}:
        if column in excluded:
            continue
        values = []
        for row in rows:
            value = row.get(column)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            mean = safe_mean(values)
            if mean is not None:
                summary[column] = mean
    return summary


def summarize_metric_rows_by(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key) or "unknown"), []).append(row)
    return {
        group: {
            "count": len(group_rows),
            "metrics": summarize_metric_rows(group_rows),
        }
        for group, group_rows in sorted(groups.items())
    }


def metric_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    excluded = {
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
        "question",
        "org",
        "organization",
        "category",
        "source_type",
        "source_doc",
    }
    counts: dict[str, int] = {}
    for column in {key for row in rows for key in row}:
        if column in excluded:
            continue
        count = 0
        for row in rows:
            value = row.get(column)
            if isinstance(value, (int, float)) and float(value) == float(value):
                count += 1
        if count:
            counts[column] = count
    return counts


def summarize_scores(scores: Any, source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    frame = scores.to_pandas()
    rows = frame.to_dict(orient="records")
    for ragas_row, source_row in zip(rows, source_rows):
        ragas_row["question"] = source_row.get("question", "")
        ragas_row["org"] = source_row.get("org", "")
        ragas_row["organization"] = source_row.get("organization", "")
        ragas_row["category"] = source_row.get("category", "")
        ragas_row["source_type"] = source_row.get("source_type", "")
        ragas_row["source_doc"] = source_row.get("source_doc", "")
    return rows, summarize_metric_rows(rows)


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Evaluation RAGAS - {summary['run_name']}",
        "",
        f"- Branch: `{summary.get('branch', '')}`",
        f"- Commit: `{summary.get('commit', '')}`",
        f"- Collector: `{summary['collector']}`",
        f"- Dataset: `{summary['dataset']}`",
        f"- Samples: `{summary['dataset_size']}`",
        f"- Judge provider: `{summary['judge_provider']}`",
        f"- Judge base URL: `{summary.get('judge_base_url')}`",
        f"- Judge key env: `{summary.get('judge_api_key_env')}`",
        f"- Judge max completion tokens: `{summary.get('judge_max_completion_tokens')}`",
    ]
    if summary.get("dataset_source_type_counts"):
        lines.extend(["", "## Dataset Split", ""])
        for label, count in summary["dataset_source_type_counts"].items():
            lines.append(f"- {label}: `{count}`")
    lines.extend(
        [
            "",
            "## RAGAS",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for metric, value in (summary.get("ragas_summary") or {}).items():
        lines.append(f"| {metric} | {value:.4f} |")
    if summary.get("ragas_metric_counts"):
        lines.extend(["", "## RAGAS Coverage", ""])
        for metric, count in summary["ragas_metric_counts"].items():
            lines.append(f"- {metric}: `{count}/{summary['dataset_size']}`")
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- RAG collection seconds: `{summary.get('rag_collection_seconds')}`",
            f"- RAGAS seconds: `{summary.get('ragas_seconds')}`",
            f"- RAG errors: `{summary.get('rag_errors')}`",
            f"- RAG success: `{summary.get('rag_success')}`",
            f"- RAGAS skipped: `{summary.get('ragas_skipped')}`",
            f"- Latency mean: `{summary.get('latency_mean')}`",
            f"- Latency median: `{summary.get('latency_median')}`",
            f"- Latency p95: `{summary.get('latency_p95')}`",
            f"- Latency max: `{summary.get('latency_max')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation RAGAS du chatbot RCAR/CNRA")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de questions pour un smoke test")
    parser.add_argument("--data-file", type=Path, default=DATA_FILE_DEFAULT)
    parser.add_argument("--run-name", default=None, help="Nom du run, ex: vespa_migration ou chroma_main")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--skip-ragas", action="store_true", help="Collecter les reponses sans lancer RAGAS")
    parser.add_argument("--from-responses", type=Path, default=None, help="Reutiliser un rag_responses.json existant")
    parser.add_argument("--sample-per-org", type=int, default=None, help="Garder N lignes RCAR et N lignes CNRA")
    parser.add_argument("--delay", type=float, default=0.0, help="Delai entre questions")
    parser.add_argument("--judge-model", default=DEFAULT_CEREBRAS_JUDGE_MODEL, help="Modele juge RAGAS via Cerebras")
    parser.add_argument("--judge-base-url", default=DEFAULT_CEREBRAS_BASE_URL, help="Endpoint OpenAI-compatible du juge")
    parser.add_argument(
        "--judge-rpm",
        type=float,
        default=DEFAULT_CEREBRAS_REQUESTS_PER_MINUTE,
        help="Requetes par minute maximum pour le juge Cerebras",
    )
    parser.add_argument(
        "--judge-max-completion-tokens",
        type=int,
        default=DEFAULT_CEREBRAS_MAX_COMPLETION_TOKENS,
        help="Budget de sortie max par appel juge Cerebras",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default=DEFAULT_CEREBRAS_API_KEY_ENV,
        help="Variable d'environnement contenant la cle API Cerebras du juge RAGAS",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--ragas-timeout", type=int, default=300)
    parser.add_argument("--max-wait", type=int, default=300)
    parser.add_argument("--make-starter-dataset", action="store_true")
    parser.add_argument("--per-org", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging("evaluation")

    if args.make_starter_dataset:
        make_starter_dataset(args.data_file, per_org=args.per_org, overwrite=args.overwrite)
        return 0

    if not args.skip_ragas:
        get_api_key_from_env(args.judge_api_key_env)

    timestamp, run_name, run_dir = make_run_dir(args.output_dir, args.run_name, "docs")

    git_meta = get_git_metadata()
    rag_started = time.perf_counter()

    if args.from_responses:
        results = json.loads(args.from_responses.read_text(encoding="utf-8"))
        results = sample_per_org(results, args.sample_per_org)
        if args.limit:
            results = results[: args.limit]
        rag_seconds = 0.0
    else:
        questions = sample_per_org(load_golden(args.data_file), args.sample_per_org)
        if args.limit:
            questions = questions[: args.limit]
        results = collect_local_responses(questions, delay=args.delay)
        rag_seconds = time.perf_counter() - rag_started

    responses_file = run_dir / "rag_responses.json"
    responses_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    errors = sum(1 for row in results if row.get("error"))
    lat_stats = latency_stats(results)
    dataset_source_type_counts = dict(
        sorted(
            Counter(
                f"{str(row.get('organization') or row.get('org') or 'unknown').upper()}:{row.get('source_type') or 'unknown'}"
                for row in results
            ).items()
        )
    )

    ragas_rows: list[dict[str, Any]] = []
    ragas_summary: dict[str, float] = {}
    ragas_summary_by_org: dict[str, Any] = {}
    ragas_summary_by_category: dict[str, Any] = {}
    ragas_summary_by_source_type: dict[str, Any] = {}
    ragas_metric_counts: dict[str, int] = {}
    ragas_skipped = 0
    ragas_seconds = 0.0

    if not args.skip_ragas:
        dataset, ragas_skipped, ragas_source_rows = build_ragas_dataset(results)
        ragas_started = time.perf_counter()
        scores = run_ragas(
            dataset=dataset,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_requests_per_minute=args.judge_rpm,
            judge_max_completion_tokens=args.judge_max_completion_tokens,
            judge_api_key_env=args.judge_api_key_env,
            max_workers=args.max_workers,
            timeout=args.ragas_timeout,
            max_wait=args.max_wait,
        )
        ragas_seconds = time.perf_counter() - ragas_started
        ragas_rows, ragas_summary = summarize_scores(scores, ragas_source_rows)
        ragas_summary_by_org = summarize_metric_rows_by(ragas_rows, "org")
        ragas_summary_by_category = summarize_metric_rows_by(ragas_rows, "category")
        ragas_summary_by_source_type = summarize_metric_rows_by(ragas_rows, "source_type")
        ragas_metric_counts = metric_counts(ragas_rows)

    write_csv(run_dir / "ragas_scores.csv", ragas_rows)

    summary: dict[str, Any] = {
        "run_name": run_name,
        "created_at": timestamp,
        "branch": git_meta.get("branch", ""),
        "commit": git_meta.get("commit", ""),
        "collector": "local",
        "base_url": "",
        "dataset": str(args.data_file),
        "dataset_size": len(results),
        "dataset_source_type_counts": dataset_source_type_counts,
        "judge_provider": "cerebras",
        "judge_llm": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_rpm": args.judge_rpm,
        "judge_api_key_env": args.judge_api_key_env,
        "judge_max_completion_tokens": args.judge_max_completion_tokens,
        "judge_embeddings": DEFAULT_EMBEDDING_MODEL,
        "skip_ragas": args.skip_ragas,
        "rag_collection_seconds": rag_seconds,
        "ragas_seconds": ragas_seconds,
        "rag_errors": errors,
        "rag_success": len(results) - errors,
        "ragas_skipped": ragas_skipped,
        "ragas_summary": ragas_summary,
        "ragas_metric_counts": ragas_metric_counts,
        "ragas_summary_by_org": ragas_summary_by_org,
        "ragas_summary_by_category": ragas_summary_by_category,
        "ragas_summary_by_source_type": ragas_summary_by_source_type,
        **lat_stats,
    }

    (run_dir / "ragas_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_markdown(run_dir / "summary.md", summary)
    logger.info("Artifacts sauvegardes dans %s", run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Evaluation docs RAGAS echouee")
        raise
