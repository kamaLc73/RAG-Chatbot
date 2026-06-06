from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.documents import Document
from loguru import logger
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ...deps import get_current_superuser, get_rag_pipeline
from ..conversations import normalize_org


router = APIRouter(
    prefix="/admin/retrieval-test",
    tags=["admin-retrieval-test"],
    dependencies=[Depends(get_current_superuser)],
)


class RetrievalTestRequest(BaseModel):
    query: str
    org: str | None = "all"
    mode: Literal["retrieval", "full"] = "full"
    include_resources: bool = True
    use_intent_classifier: bool = True


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resource_id(item: dict[str, Any], resource_type: str) -> str:
    if resource_type == "video":
        return str(item.get("video_id") or item.get("id") or item.get("url") or item.get("title") or "")
    return str(item.get("form_id") or item.get("id") or item.get("pdf_url") or item.get("title") or "")


def _serialize_resource(item: Any, resource_type: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"type": resource_type, "title": str(item)}

    payload = {str(key): _jsonable(value) for key, value in item.items()}
    payload["type"] = resource_type
    payload["id"] = _resource_id(payload, resource_type)
    payload["title"] = str(payload.get("title") or payload.get("id") or "Ressource")
    if resource_type == "form":
        payload["url"] = payload.get("pdf_url") or payload.get("page_url") or payload.get("url") or ""
    else:
        payload["url"] = payload.get("url") or ""
    return payload


def _serialize_chunk(doc: Document | dict[str, Any], index: int) -> dict[str, Any]:
    if isinstance(doc, Document):
        text = doc.page_content or ""
        metadata = dict(doc.metadata or {})
    else:
        text = str(doc.get("text") or "")
        metadata = dict(doc.get("metadata") or {})

    title = (
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("relative_source")
        or metadata.get("doc_id")
        or f"Chunk {index}"
    )
    return {
        "rank": index,
        "title": str(title),
        "text": text,
        "org": str(metadata.get("org") or ""),
        "source_type": str(metadata.get("source_type") or ""),
        "relative_source": str(metadata.get("relative_source") or ""),
        "chunk_index": metadata.get("chunk_index"),
        "vespa_rank": metadata.get("vespa_rank"),
        "vespa_relevance": _safe_float(metadata.get("vespa_relevance")),
        "rerank_score": _safe_float(metadata.get("vespa_rerank_score")),
        "rerank_raw_score": _safe_float(metadata.get("vespa_rerank_raw_score")),
        "source_boost": _safe_float(metadata.get("vespa_rerank_source_boost")),
        "rerank_mode": metadata.get("vespa_rerank_mode"),
        "metadata": _jsonable(metadata),
    }


def _context_items_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    context_items = result.get("context_items")
    if isinstance(context_items, list):
        return [item for item in context_items if isinstance(item, dict)]

    contexts = result.get("contexts")
    metadata = result.get("context_metadata")
    if not isinstance(contexts, list):
        return []
    return [
        {
            "text": str(context or ""),
            "metadata": metadata[index] if isinstance(metadata, list) and index < len(metadata) else {},
        }
        for index, context in enumerate(contexts)
    ]


def _render_prompt_debug(rag: Any, org: str, context: str, videos: list[dict[str, Any]], forms: list[dict[str, Any]]) -> dict[str, str]:
    try:
        supplementary_hint = rag._build_supplementary_hint(videos, forms)
        org_identity = rag._build_org_identity(org)
        system_prompt = rag._system_prompt().format(
            org_identity=org_identity,
            supplementary_hint=supplementary_hint,
            context=context,
        )
    except Exception as exc:
        supplementary_hint = ""
        org_identity = ""
        system_prompt = f"Prompt indisponible: {exc}"

    return {
        "system": system_prompt,
        "context": context,
        "org_identity": org_identity,
        "supplementary_hint": supplementary_hint,
    }


def _disabled_classification(reason: str) -> dict[str, Any]:
    return {
        "intent": "retrieval",
        "confidence": 1.0,
        "reasoning": reason,
        "loaded": False,
    }


def _drop_response_cache_key(rag: Any, query: str, org: str, use_intent_classifier: bool) -> None:
    cache = getattr(rag, "response_cache", None)
    if cache is None:
        return

    mode = "intent" if use_intent_classifier else "docs"
    key = f"{org}:{mode}:{query.strip().lower()}"
    try:
        cache.pop(key, None)
    except AttributeError:
        return


def _run_retrieval_only(rag: Any, query: str, org: str, include_resources: bool, use_intent_classifier: bool) -> dict[str, Any]:
    started = time.perf_counter()
    timings: dict[str, float] = {}

    t_classify = time.perf_counter()
    if use_intent_classifier:
        classification = rag._classify_intent(query)
    else:
        classification = _disabled_classification("Intent classifier disabled for this diagnostic run")
    timings["classification_seconds"] = round(time.perf_counter() - t_classify, 6)

    offscope = rag._detect_offscope_org(query, org)
    if offscope:
        response = rag._offscope_response(org, offscope, query)
        timings["total_seconds"] = round(time.perf_counter() - started, 6)
        return {
            "status": "ok",
            "mode": "retrieval",
            "query": query,
            "org": org,
            "response": response.get("response"),
            "intent": "out_of_scope",
            "intent_confidence": 1.0,
            "intent_reasoning": "Org filter detected an off-scope question",
            "loaded": True,
            "cached": False,
            "timings": timings,
            "chunks": [],
            "videos": [],
            "forms": [],
            "video_candidates": [],
            "form_candidates": [],
            "gates": {"videos": {"enabled": False}, "forms": {"enabled": False}},
            "prompt": _render_prompt_debug(rag, org, "", [], []),
        }

    t_embed = time.perf_counter()
    query_embedding = rag.embeddings.embed_query(query)
    timings["embedding_seconds"] = round(time.perf_counter() - t_embed, 6)

    run_videos = False
    run_forms = False
    resource_only_video = bool(use_intent_classifier and rag._is_resource_only_query(query, "video"))
    resource_only_form = bool(use_intent_classifier and rag._is_resource_only_query(query, "form"))
    resource_only = resource_only_video or resource_only_form

    t_docs = time.perf_counter()
    if resource_only:
        docs = []
        timings["docs_retrieval_seconds"] = 0.0
    else:
        docs = rag._retrieve_docs_vespa(query, query_embedding, org=org)
        timings["docs_retrieval_seconds"] = round(time.perf_counter() - t_docs, 6)

    t_context = time.perf_counter()
    context = "" if resource_only else rag._build_context(docs)
    timings["context_build_seconds"] = round(time.perf_counter() - t_context, 6)

    videos: list[dict[str, Any]] = []
    forms: list[dict[str, Any]] = []
    video_candidates: list[dict[str, Any]] = []
    form_candidates: list[dict[str, Any]] = []
    t_resources = time.perf_counter()
    if include_resources:
        run_videos = bool(use_intent_classifier and rag._should_run_auxiliary_retriever(classification, "video", query))
        run_forms = bool(use_intent_classifier and rag._should_run_auxiliary_retriever(classification, "form", query))
        if run_videos:
            videos_raw, video_candidates_raw = rag._retrieve_videos_with_debug(query, query_embedding, org=org)
            videos = [_serialize_resource(item, "video") for item in videos_raw]
            video_candidates = [_serialize_resource(item, "video") for item in video_candidates_raw]
        if run_forms:
            forms_raw, form_candidates_raw = rag._retrieve_forms_with_debug(query, query_embedding, org=org)
            forms = [_serialize_resource(item, "form") for item in forms_raw]
            form_candidates = [_serialize_resource(item, "form") for item in form_candidates_raw]
    timings["resources_retrieval_seconds"] = round(time.perf_counter() - t_resources, 6)
    timings["total_seconds"] = round(time.perf_counter() - started, 6)

    chunks = [_serialize_chunk(doc, index) for index, doc in enumerate(docs, start=1)]
    prompt = _render_prompt_debug(rag, org, context, videos, forms)
    classifier = getattr(rag, "intent_classifier", None)

    return {
        "status": "ok",
        "mode": "retrieval",
        "query": query,
        "org": org,
        "response": "",
        "intent": classification.get("intent", "retrieval"),
        "intent_confidence": classification.get("confidence", 0.0),
        "intent_reasoning": classification.get("reasoning", ""),
        "loaded": classification.get("loaded", False),
        "cached": False,
        "timings": timings,
        "chunks": chunks,
        "videos": videos,
        "forms": forms,
        "video_candidates": video_candidates,
        "form_candidates": form_candidates,
        "gates": {
            "videos": {
                "enabled": run_videos,
                "signal": bool(getattr(classifier, "has_video_signal", lambda _: False)(query)) if classifier else False,
            },
            "forms": {
                "enabled": run_forms,
                "signal": bool(getattr(classifier, "has_form_signal", lambda _: False)(query)) if classifier else False,
            },
        },
        "prompt": prompt,
    }


def _run_full_query(rag: Any, query: str, org: str, include_resources: bool, use_intent_classifier: bool) -> dict[str, Any]:
    started = time.perf_counter()
    resource_only = bool(
        use_intent_classifier
        and (
            rag._is_resource_only_query(query, "video")
            or rag._is_resource_only_query(query, "form")
        )
    )
    _drop_response_cache_key(rag, query, org, use_intent_classifier)
    result = rag.query(
        query,
        org=org,
        include_contexts=not resource_only,
        use_intent_classifier=use_intent_classifier,
    )
    _drop_response_cache_key(rag, query, org, use_intent_classifier)
    timings = dict(result.get("timings") or {})
    timings.setdefault("total_seconds", round(time.perf_counter() - started, 6))

    context_items = _context_items_from_result(result)
    chunks = [_serialize_chunk(item, index) for index, item in enumerate(context_items, start=1)]
    raw_videos = result.get("videos") if isinstance(result.get("videos"), list) else []
    raw_forms = result.get("forms") if isinstance(result.get("forms"), list) else []
    videos = [_serialize_resource(item, "video") for item in raw_videos if isinstance(item, dict)] if include_resources else []
    forms = [_serialize_resource(item, "form") for item in raw_forms if isinstance(item, dict)] if include_resources else []
    context = "\n".join(str(item.get("text") or "") for item in context_items)
    video_candidates: list[dict[str, Any]] = []
    form_candidates: list[dict[str, Any]] = []

    t_candidate_debug = time.perf_counter()
    if include_resources:
        classification = (
            rag._classify_intent(query)
            if use_intent_classifier
            else _disabled_classification("Intent classifier disabled for this diagnostic run")
        )
        run_videos = bool(use_intent_classifier and rag._should_run_auxiliary_retriever(classification, "video", query))
        run_forms = bool(use_intent_classifier and rag._should_run_auxiliary_retriever(classification, "form", query))
        if run_videos or run_forms:
            query_embedding = rag.embeddings.embed_query(query)
            if run_videos:
                _, video_candidates_raw = rag._retrieve_videos_with_debug(query, query_embedding, org=org)
                video_candidates = [_serialize_resource(item, "video") for item in video_candidates_raw]
            if run_forms:
                _, form_candidates_raw = rag._retrieve_forms_with_debug(query, query_embedding, org=org)
                form_candidates = [_serialize_resource(item, "form") for item in form_candidates_raw]
    timings["resource_candidates_seconds"] = round(time.perf_counter() - t_candidate_debug, 6)
    timings["endpoint_total_seconds"] = round(time.perf_counter() - started, 6)

    return {
        "status": "ok",
        "mode": "full",
        "query": query,
        "org": result.get("org", org),
        "response": result.get("response", ""),
        "intent": result.get("intent", "retrieval"),
        "intent_confidence": result.get("intent_confidence", 0.0),
        "intent_reasoning": "",
        "loaded": True,
        "cached": False,
        "timings": timings,
        "chunks": chunks,
        "videos": videos,
        "forms": forms,
        "video_candidates": video_candidates,
        "form_candidates": form_candidates,
        "gates": {
            "videos": {"enabled": bool(videos)},
            "forms": {"enabled": bool(forms)},
        },
        "prompt": _render_prompt_debug(rag, str(result.get("org", org)), context, videos, forms),
    }


def _run_test(rag: Any, payload: RetrievalTestRequest) -> dict[str, Any]:
    query = payload.query.strip()
    org = normalize_org(payload.org)
    if payload.mode == "retrieval":
        return _run_retrieval_only(rag, query, org, payload.include_resources, payload.use_intent_classifier)
    return _run_full_query(rag, query, org, payload.include_resources, payload.use_intent_classifier)


@router.post("/query")
async def run_retrieval_test(
    payload: RetrievalTestRequest,
    rag=Depends(get_rag_pipeline),
) -> dict[str, Any]:
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="La requête est obligatoire")

    try:
        return await run_in_threadpool(_run_test, rag, payload)
    except Exception as exc:
        logger.exception("Erreur pendant le test de recuperation admin")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
