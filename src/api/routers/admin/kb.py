from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
import requests
from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

from ...cache import api_cache
from ...config import settings
from ...database import get_db
from ...deps import get_current_superuser
from ...models import KbSource, KbSourceItem


router = APIRouter(prefix="/admin/kb", tags=["admin-kb"], dependencies=[Depends(get_current_superuser)])

MAX_TREE_HITS = 10000
VISIT_PAGE_SIZE = 400
KB_CACHE_TTL_SECONDS = settings.kb_cache_ttl_seconds
UPLOAD_ROOT = Path("data/uploads/kb")
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}


def _extract_hits(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        return response.get("root", {}).get("children", []) or []
    if hasattr(response, "json"):
        payload = response.json() if callable(response.json) else response.json
        return payload.get("root", {}).get("children", []) or []
    hits = getattr(response, "hits", None)
    return list(hits or [])


def _fields(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        return hit.get("fields", {}) or {}
    return getattr(hit, "fields", {}) or {}


def _query_schema_fields(app: Any, schema: str, fields: str, hits: int) -> list[dict[str, Any]]:
    if hits <= 0:
        return []
    response = app.query(
        body={
            "yql": f"select {fields} from {schema} where true",
            "hits": min(hits, MAX_TREE_HITS),
            "timeout": "10s",
        }
    )
    return [_fields(hit) for hit in _extract_hits(response)]


def _vespa_http_base(url: str, port: int) -> str:
    parsed = urlsplit(url)
    if parsed.scheme and parsed.hostname and parsed.port:
        return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    return f"{url.rstrip('/')}:{port}"


def _visit_schema_fields(schema: str, total_count: int, field_set: str) -> list[dict[str, Any]]:
    if total_count <= 0:
        return []

    from src.store.vespa_store import DEFAULT_CONTENT_CLUSTER, DEFAULT_VESPA_PORT, DEFAULT_VESPA_URL

    base_url = _vespa_http_base(DEFAULT_VESPA_URL, DEFAULT_VESPA_PORT)
    url = f"{base_url}/document/v1/{schema}/{schema}/docid/"
    params: dict[str, Any] = {
        "cluster": DEFAULT_CONTENT_CLUSTER,
        "selection": "true",
        "wantedDocumentCount": min(VISIT_PAGE_SIZE, MAX_TREE_HITS),
        "fieldSet": field_set,
    }
    fields: list[dict[str, Any]] = []

    while len(fields) < min(total_count, MAX_TREE_HITS):
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        for document in payload.get("documents", []) or []:
            document_fields = document.get("fields", {}) if isinstance(document, dict) else {}
            if document_fields:
                fields.append(document_fields)
                if len(fields) >= min(total_count, MAX_TREE_HITS):
                    break
        continuation = payload.get("continuation")
        if not continuation:
            break
        params["continuation"] = continuation

    return fields


def _org_key(value: Any) -> str:
    text = str(value or "both").strip().lower()
    if text in {"cnra", "rcar"}:
        return text
    return "both"


def _org_label(key: str) -> str:
    return {"cnra": "CNRA", "rcar": "RCAR"}.get(key, key.upper())


def _target_orgs(value: Any) -> tuple[list[str], bool]:
    org = _org_key(value)
    if org == "both":
        return ["cnra", "rcar"], True
    return [org], False


def _source_label(source_type: str) -> str:
    mapping = {
        "faq": "FAQ",
        "web": "Web",
        "bibliotheque": "Biblioth.",
        "form": "Form.",
        "video": "Vidéo",
        "admin_upload": "Import",
        "unknown": "Autre",
    }
    return mapping.get(source_type or "unknown", source_type.title())


def _normalize_import_org(value: str) -> str:
    text = (value or "both").strip().lower()
    if text in {"cnra", "rcar"}:
        return text
    if text in {"both", "all", "shared", "partage", "partagé", "les deux", "cnra & rcar"}:
        return "both"
    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Organisme invalide")


def _safe_upload_name(filename: str) -> str:
    import re

    name = Path(filename or "document.txt").name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "document.txt"


def _extract_pdf_text(path: Path) -> str:
    import fitz

    doc = fitz.open(path)
    try:
        return "\n\n".join(page.get_text("text") for page in doc).strip()
    finally:
      doc.close()


def _extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text("\n", strip=True)
    return text.strip()


def _get_embeddings(request: Request):
    rag = getattr(request.app.state, "rag", None)
    embeddings = getattr(rag, "embeddings", None)
    if embeddings is not None:
        return embeddings

    from langchain_huggingface import HuggingFaceEmbeddings
    from src.chatbot.rag_pipeline import DEFAULT_EMBEDDING_MODEL

    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def _document_to_vespa_items(
    *,
    source_id: int,
    title: str,
    text: str,
    org: str,
    source_type: str,
    faq_scope: str,
    relative_source: str,
    embeddings,
) -> list[dict[str, Any]]:
    from src.chatbot.index_data import build_chunker
    from src.store.vespa_store import stable_data_id

    splitter = build_chunker()
    chunks = splitter.split_documents(
        [
            Document(
                page_content=text,
                metadata={
                    "title": title,
                    "org": org,
                    "source_type": source_type,
                    "faq_scope": faq_scope,
                    "relative_source": relative_source,
                },
            )
        ]
    )
    if not chunks:
        raise ValueError("Aucun chunk généré à partir du document.")

    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    documents: list[dict[str, Any]] = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        vespa_id = stable_data_id("admin_doc", source_id, index)
        documents.append(
            {
                "id": vespa_id,
                "fields": {
                    "doc_id": vespa_id,
                    "text": chunk.page_content,
                    "org": org,
                    "source_type": source_type,
                    "faq_scope": faq_scope,
                    "relative_source": relative_source,
                    "chunk_index": index,
                    "embedding": vector,
                },
            }
        )
    return documents


def _feed_document_import(
    *,
    source_id: int,
    title: str,
    text: str,
    org: str,
    source_type: str,
    faq_scope: str,
    relative_source: str,
    embeddings,
) -> list[str]:
    from src.store.vespa_store import feed_documents, make_vespa_app

    documents = _document_to_vespa_items(
        source_id=source_id,
        title=title,
        text=text,
        org=org,
        source_type=source_type,
        faq_scope=faq_scope,
        relative_source=relative_source,
        embeddings=embeddings,
    )
    feed_documents(make_vespa_app(), schema="doc", documents=documents)
    return [str(document["id"]) for document in documents]


def _feed_form_import(
    *,
    source_id: int,
    title: str,
    category: str,
    content: str,
    org: str,
    pdf_url: str,
    page_url: str,
    embeddings,
) -> str:
    from src.store.vespa_store import feed_documents, make_vespa_app, stable_data_id

    searchable_content = "\n".join(
        part
        for part in [
            f"Organisation : {org}",
            f"Categorie : {category}",
            f"Formulaire : {title}",
            f"URL PDF : {pdf_url}" if pdf_url else "",
            f"URL page : {page_url}" if page_url else "",
            "",
            content.strip(),
        ]
        if part
    )
    vector = embeddings.embed_documents([searchable_content])[0]
    form_id = stable_data_id("admin_form", source_id, title)
    feed_documents(
        make_vespa_app(),
        schema="form",
        documents=[
            {
                "id": form_id,
                "fields": {
                    "form_id": form_id,
                    "title": title,
                    "category": category,
                    "content": searchable_content,
                    "org": org,
                    "pdf_url": pdf_url,
                    "page_url": page_url,
                    "embedding": vector,
                },
            }
        ],
    )
    return form_id


def _extract_youtube_id(url: str) -> str:
    parsed = urlsplit(url or "")
    if parsed.hostname and "youtu.be" in parsed.hostname:
        return parsed.path.strip("/").split("/", 1)[0]
    query_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_id:
        return query_id
    return ""


def _feed_video_import(
    *,
    source_id: int,
    title: str,
    transcript: str,
    org: str,
    url: str,
    upload_date: str,
    video_id: str,
    embeddings,
) -> str:
    from src.store.vespa_store import feed_documents, make_vespa_app, stable_data_id

    clean_video_id = video_id or _extract_youtube_id(url) or f"admin_video_{source_id}"
    vespa_id = stable_data_id("video", clean_video_id)
    searchable_content = f"Titre : {title}\n\n{transcript.strip()}"
    vector = embeddings.embed_documents([searchable_content])[0]
    thumbnail_url = f"https://img.youtube.com/vi/{clean_video_id}/mqdefault.jpg" if clean_video_id else ""
    feed_documents(
        make_vespa_app(),
        schema="video",
        documents=[
            {
                "id": vespa_id,
                "fields": {
                    "video_id": clean_video_id,
                    "title": title,
                    "transcript": searchable_content,
                    "org": org,
                    "url": url,
                    "thumbnail_url": thumbnail_url,
                    "upload_date": upload_date,
                    "embedding": vector,
                },
            }
        ],
    )
    return vespa_id


def _delete_vespa_item(schema: str, vespa_id: str) -> None:
    from src.store.vespa_store import make_vespa_app

    make_vespa_app().delete_data(schema=schema, data_id=vespa_id)


def _source_dict(source: KbSource, items_count: int | None = None) -> dict[str, Any]:
    return {
        "id": str(source.id),
        "type": source.type,
        "title": source.title,
        "org": source.org,
        "source_type": source.source_type,
        "source_path": source.source_path,
        "source_url": source.source_url,
        "status": source.status,
        "error": source.error,
        "items_count": len(source.items) if items_count is None else items_count,
        "created_at": source.created_at.isoformat(),
        "updated_at": source.updated_at.isoformat(),
    }


def _bucket_name(fields: dict[str, Any]) -> str:
    source_type = str(fields.get("source_type") or "unknown").strip() or "unknown"
    faq_scope = str(fields.get("faq_scope") or "").strip()
    relative_source = str(fields.get("relative_source") or "").strip()
    parts = [part for part in relative_source.replace("\\", "/").split("/") if part]

    if faq_scope and faq_scope != "unknown":
        return faq_scope
    if len(parts) >= 3:
        return parts[2]
    return source_type


def _new_org_node(org: str) -> dict[str, Any]:
    return {
        "id": org,
        "label": _org_label(org),
        "kind": "org",
        "chunks": 0,
        "documents": 0,
        "items": 0,
        "shared_items": 0,
        "children": {},
    }


def _new_bucket_node(bucket_id: str, label: str, source_type: str) -> dict[str, Any]:
    return {
        "id": bucket_id,
        "label": label,
        "kind": source_type,
        "badge": _source_label(source_type),
        "chunks": 0,
        "documents": 0,
        "items": 0,
        "shared_items": 0,
        "_sources": set(),
    }


def _public_tree(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for org in ("cnra", "rcar"):
        node = nodes.get(org, _new_org_node(org))
        children = []
        for child in sorted(node["children"].values(), key=lambda item: (item["kind"], item["label"].lower())):
            child["documents"] = len(child.pop("_sources", set())) or child["documents"]
            children.append(child)
        node = {**node, "children": children}
        result.append(node)
    return result


def _build_kb_payload() -> dict[str, Any]:
    from src.store.vespa_store import count_schema, make_vespa_app

    app = make_vespa_app()
    doc_chunks = count_schema(app, "doc")
    video_count = count_schema(app, "video")
    form_count = count_schema(app, "form")

    doc_hits = _visit_schema_fields(
        "doc",
        doc_chunks,
        "doc:doc_id,org,source_type,faq_scope,relative_source,chunk_index",
    )
    form_hits = _visit_schema_fields("form", form_count, "form:form_id,title,category,org,pdf_url,page_url")
    video_hits = _visit_schema_fields("video", video_count, "video:video_id,title,org,url,upload_date")

    shared_items = 0
    source_types: Counter[str] = Counter()
    source_documents: set[str] = set()
    tree_nodes: dict[str, dict[str, Any]] = defaultdict(dict)

    for fields in doc_hits:
        target_orgs, is_shared = _target_orgs(fields.get("org"))
        if is_shared:
            shared_items += 1
        source_type = str(fields.get("source_type") or "unknown").strip() or "unknown"
        source_types[source_type] += 1
        relative_source = str(fields.get("relative_source") or "").strip()
        if relative_source:
            source_documents.add(relative_source)

        bucket = _bucket_name(fields)
        bucket_id = f"doc:{source_type}:{bucket}"
        for org in target_orgs:
            tree_nodes.setdefault(org, _new_org_node(org))
            child = tree_nodes[org]["children"].setdefault(bucket_id, _new_bucket_node(bucket_id, bucket, source_type))
            child["chunks"] += 1
            child["items"] += 1
            if is_shared:
                child["shared_items"] += 1
                tree_nodes[org]["shared_items"] += 1
            if relative_source:
                child["_sources"].add(relative_source)
            tree_nodes[org]["chunks"] += 1
            tree_nodes[org]["items"] += 1

    for fields in form_hits:
        target_orgs, is_shared = _target_orgs(fields.get("org"))
        if is_shared:
            shared_items += 1
        for org in target_orgs:
            tree_nodes.setdefault(org, _new_org_node(org))
            child = tree_nodes[org]["children"].setdefault("form:forms", _new_bucket_node("form:forms", "Formulaires", "form"))
            child["items"] += 1
            child["documents"] += 1
            if is_shared:
                child["shared_items"] += 1
                tree_nodes[org]["shared_items"] += 1
            tree_nodes[org]["items"] += 1

    for fields in video_hits:
        target_orgs, is_shared = _target_orgs(fields.get("org"))
        if is_shared:
            shared_items += 1
        for org in target_orgs:
            tree_nodes.setdefault(org, _new_org_node(org))
            child = tree_nodes[org]["children"].setdefault("video:videos", _new_bucket_node("video:videos", "Vidéos", "video"))
            child["items"] += 1
            child["documents"] += 1
            if is_shared:
                child["shared_items"] += 1
                tree_nodes[org]["shared_items"] += 1
            tree_nodes[org]["items"] += 1

    for org, node in tree_nodes.items():
        node["documents"] = sum(child["documents"] or len(child.get("_sources", set())) for child in node["children"].values())

    total_indexed_units = doc_chunks + form_count + video_count
    top_source_type = source_types.most_common(1)[0][0] if source_types else "-"

    return {
        "status": "ok",
        "schema_counts": {
            "doc_chunks": doc_chunks,
            "forms": form_count,
            "videos": video_count,
        },
        "stats": {
            "organisms": 2,
            "source_documents": len(source_documents),
            "doc_chunks": doc_chunks,
            "forms": form_count,
            "videos": video_count,
            "indexed_units": total_indexed_units,
            "shared_items": shared_items,
            "buckets": len({child_id for node in tree_nodes.values() for child_id in node["children"]}),
            "top_source_type": _source_label(top_source_type),
        },
        "source_types": dict(source_types),
        "tree": _public_tree(tree_nodes),
        "sampled": total_indexed_units > MAX_TREE_HITS,
    }


def _cached_kb_payload() -> dict[str, Any]:
    return api_cache.get_or_set("admin:kb:stats", _build_kb_payload, ttl_seconds=KB_CACHE_TTL_SECONDS)


async def _cached_kb_sources(db: AsyncSession) -> list[dict[str, Any]]:
    cached = api_cache.get("admin:kb:sources")
    if cached is not None:
        return cached

    result = await db.execute(
        select(KbSource)
        .options(selectinload(KbSource.items))
        .order_by(KbSource.created_at.desc())
    )
    payload = [_source_dict(source) for source in result.scalars().all()]
    return api_cache.set("admin:kb:sources", payload, ttl_seconds=KB_CACHE_TTL_SECONDS)


@router.get("/status")
def knowledge_base_status() -> dict[str, object]:
    try:
        payload = _cached_kb_payload()
        return {
            "status": payload["status"],
            "schemas": {
                "docs": payload["schema_counts"]["doc_chunks"],
                "videos": payload["schema_counts"]["videos"],
                "forms": payload["schema_counts"]["forms"],
            },
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "schemas": {}}


@router.get("/stats")
def knowledge_base_stats() -> dict[str, object]:
    try:
        return _cached_kb_payload()
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "schema_counts": {}, "stats": {}, "tree": []}


@router.get("")
def list_kb_items() -> list[dict[str, object]]:
    status = knowledge_base_status()
    schemas = status.get("schemas", {})
    if not isinstance(schemas, dict):
        schemas = {}
    return [
        {"id": "doc", "title": "Chunks documentaires", "source": "Vespa doc", "status": str(schemas.get("docs", 0))},
        {"id": "form", "title": "Formulaires indexes", "source": "Vespa form", "status": str(schemas.get("forms", 0))},
        {"id": "video", "title": "Vidéos indexées", "source": "Vespa video", "status": str(schemas.get("videos", 0))},
    ]


@router.get("/sources")
async def list_kb_sources(db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    return await _cached_kb_sources(db)


@router.post("/import/document")
async def import_document(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
    org: str = Form("both"),
    source_type: str = Form("admin_upload"),
    faq_scope: str = Form("admin"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    filename = _safe_upload_name(file.filename or "document.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Type de fichier non supporté. Utilisez PDF, TXT, MD ou HTML.",
        )

    normalized_org = _normalize_import_org(org)
    clean_title = (title or Path(filename).stem).strip() or Path(filename).stem
    clean_source_type = (source_type or "admin_upload").strip().lower()[:80] or "admin_upload"
    clean_faq_scope = (faq_scope or "admin").strip()[:80] or "admin"

    source = KbSource(
        type="document",
        title=clean_title,
        org=normalized_org,
        source_type=clean_source_type,
        status="processing",
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)

    upload_dir = UPLOAD_ROOT / "documents" / str(source.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / filename
    upload_path.write_bytes(await file.read())
    source.source_path = str(upload_path.as_posix())
    await db.commit()
    await db.refresh(source)

    try:
        text = await run_in_threadpool(_extract_document_text, upload_path)
        if not text.strip():
            raise ValueError("Le document ne contient pas de texte exploitable.")

        relative_source = f"admin/{normalized_org}/{source.id}/{filename}"
        embeddings = _get_embeddings(request)
        vespa_ids = await run_in_threadpool(
            _feed_document_import,
            source_id=source.id,
            title=clean_title,
            text=text,
            org=normalized_org,
            source_type=clean_source_type,
            faq_scope=clean_faq_scope,
            relative_source=relative_source,
            embeddings=embeddings,
        )

        source.status = "indexed"
        source.error = None
        db.add_all([KbSourceItem(source_id=source.id, schema="doc", vespa_id=vespa_id) for vespa_id in vespa_ids])
        await db.commit()
        await db.refresh(source)
        api_cache.clear_prefix("admin:kb:")
        return _source_dict(source, items_count=len(vespa_ids))
    except Exception as exc:
        source.status = "error"
        source.error = str(exc)
        await db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/import/form")
async def import_form(
    request: Request,
    title: str = Form(...),
    org: str = Form("both"),
    category: str = Form("Import admin"),
    pdf_url: str = Form(""),
    page_url: str = Form(""),
    content: str = Form(""),
    file: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    normalized_org = _normalize_import_org(org)
    clean_title = title.strip()
    if not clean_title:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Titre obligatoire")

    source = KbSource(
        type="form",
        title=clean_title,
        org=normalized_org,
        source_type="form",
        source_url=pdf_url.strip() or page_url.strip() or None,
        status="processing",
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)

    try:
        body = content.strip()
        if file is not None and file.filename:
            filename = _safe_upload_name(file.filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
                raise ValueError("Le fichier formulaire doit etre PDF, TXT, MD ou HTML.")
            upload_dir = UPLOAD_ROOT / "forms" / str(source.id)
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_path = upload_dir / filename
            upload_path.write_bytes(await file.read())
            source.source_path = str(upload_path.as_posix())
            body = await run_in_threadpool(_extract_document_text, upload_path)

        if len(body.strip()) < 30:
            raise ValueError("Le contenu du formulaire est trop court pour etre indexe.")

        embeddings = _get_embeddings(request)
        vespa_id = await run_in_threadpool(
            _feed_form_import,
            source_id=source.id,
            title=clean_title,
            category=category.strip() or "Import admin",
            content=body,
            org=normalized_org,
            pdf_url=pdf_url.strip(),
            page_url=page_url.strip(),
            embeddings=embeddings,
        )

        source.status = "indexed"
        source.error = None
        db.add(KbSourceItem(source_id=source.id, schema="form", vespa_id=vespa_id))
        await db.commit()
        await db.refresh(source)
        api_cache.clear_prefix("admin:kb:")
        return _source_dict(source, items_count=1)
    except Exception as exc:
        source.status = "error"
        source.error = str(exc)
        await db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/import/video")
async def import_video(
    request: Request,
    title: str = Form(...),
    org: str = Form("both"),
    url: str = Form(""),
    transcript: str = Form(...),
    upload_date: str = Form(""),
    video_id: str = Form(""),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    normalized_org = _normalize_import_org(org)
    clean_title = title.strip()
    clean_transcript = transcript.strip()
    if not clean_title:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Titre obligatoire")
    if len(clean_transcript) < 80:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Transcript trop court")

    source = KbSource(
        type="video",
        title=clean_title,
        org=normalized_org,
        source_type="video",
        source_url=url.strip() or None,
        status="processing",
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)

    try:
        embeddings = _get_embeddings(request)
        vespa_id = await run_in_threadpool(
            _feed_video_import,
            source_id=source.id,
            title=clean_title,
            transcript=clean_transcript,
            org=normalized_org,
            url=url.strip(),
            upload_date=upload_date.strip(),
            video_id=video_id.strip(),
            embeddings=embeddings,
        )

        source.status = "indexed"
        source.error = None
        db.add(KbSourceItem(source_id=source.id, schema="video", vespa_id=vespa_id))
        await db.commit()
        await db.refresh(source)
        api_cache.clear_prefix("admin:kb:")
        return _source_dict(source, items_count=1)
    except Exception as exc:
        source.status = "error"
        source.error = str(exc)
        await db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.delete("/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb_source(source_id: int, db: AsyncSession = Depends(get_db)) -> None:
    result = await db.execute(
        select(KbSource)
        .options(selectinload(KbSource.items))
        .where(KbSource.id == source_id)
    )
    source = result.scalar_one_or_none()
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source introuvable")

    for item in source.items:
        await run_in_threadpool(_delete_vespa_item, item.schema, item.vespa_id)

    await db.delete(source)
    await db.commit()
    api_cache.clear_prefix("admin:kb:")


@router.post("/reindex")
def reindex_stub() -> dict[str, object]:
    api_cache.clear_prefix("admin:kb:")
    return {
        "status": "not_started",
        "message": "Reindexing is intentionally not launched by the API stub. Run the existing indexers from the backend host.",
    }
