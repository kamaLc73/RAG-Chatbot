from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlsplit

from vespa.application import Vespa


DEFAULT_VESPA_URL = os.getenv("VESPA_URL", "http://localhost")
DEFAULT_VESPA_PORT = int(os.getenv("VESPA_PORT", "8080"))
DEFAULT_CONTENT_CLUSTER = os.getenv("VESPA_CONTENT_CLUSTER", "rcar_cnra")


@dataclass
class VespaHit:
    id: str
    relevance: float
    fields: dict[str, Any]


def make_vespa_app(
    url: str = DEFAULT_VESPA_URL,
    port: int = DEFAULT_VESPA_PORT,
) -> Vespa:
    parsed = urlsplit(url)
    if parsed.scheme and parsed.hostname and parsed.port:
        clean_url = f"{parsed.scheme}://{parsed.hostname}"
        return Vespa(url=clean_url, port=parsed.port)
    return Vespa(url=url, port=port)


def stable_data_id(*parts: Any) -> str:
    raw = "::".join(str(part or "") for part in parts)
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_")
    return normalized[:512] or "empty"


def clean_vespa_string(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(
        ch if ch in "\t\n\r" or ord(ch) >= 0x20 else " "
        for ch in text
    )


def clean_vespa_fields(fields: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in fields.items():
        if isinstance(value, str) or value is None:
            cleaned[key] = clean_vespa_string(value)
        else:
            cleaned[key] = value
    return cleaned


def build_org_yql_filter(org: str) -> str:
    if org == "all":
        return ""
    escaped = re.sub(r"[^a-zA-Z0-9_-]", "", org.lower())
    if not escaped:
        return ""
    return f' and (org contains "{escaped}" or org contains "both")'


def query_schema(
    app: Vespa,
    *,
    schema: str,
    query_text: str,
    query_embedding: list[float],
    org: str = "all",
    target_hits: int = 12,
    hits: int = 4,
    timeout: str = "3s",
) -> list[VespaHit]:
    yql = (
        f"select * from {schema} where "
        f"({{targetHits:{target_hits}}}nearestNeighbor(embedding, q_embedding) "
        f"or userInput(@query_text))"
        f"{build_org_yql_filter(org)}"
    )
    response = app.query(
        body={
            "yql": yql,
            "query_text": query_text,
            "input.query(q_embedding)": query_embedding,
            "ranking": "hybrid",
            "hits": hits,
            "timeout": timeout,
        }
    )
    raw_hits = getattr(response, "hits", None)
    if raw_hits is None and isinstance(response, dict):
        raw_hits = response.get("root", {}).get("children", [])
    if raw_hits is None and hasattr(response, "json"):
        payload = response.json() if callable(response.json) else response.json
        raw_hits = payload.get("root", {}).get("children", [])

    parsed: list[VespaHit] = []
    for hit in raw_hits or []:
        fields = hit.get("fields", {}) if isinstance(hit, dict) else getattr(hit, "fields", {})
        parsed.append(
            VespaHit(
                id=hit.get("id", "") if isinstance(hit, dict) else getattr(hit, "id", ""),
                relevance=float(hit.get("relevance", 0.0) if isinstance(hit, dict) else getattr(hit, "relevance", 0.0)),
                fields=fields or {},
            )
        )
    return parsed


def count_schema(app: Vespa, schema: str, timeout: str = "3s") -> int:
    response = app.query(body={"yql": f"select * from {schema} where true", "hits": 0, "timeout": timeout})
    if isinstance(response, dict):
        root = response.get("root", {})
    elif hasattr(response, "json"):
        payload = response.json() if callable(response.json) else response.json
        root = payload.get("root", {})
    else:
        root = {}
    return int(root.get("fields", {}).get("totalCount", 0)) if isinstance(root, dict) else 0


def delete_all_docs(app: Vespa, schema: str, content_cluster_name: str = DEFAULT_CONTENT_CLUSTER) -> Any:
    return app.delete_all_docs(
        content_cluster_name=content_cluster_name,
        schema=schema,
    )


def feed_documents(
    app: Vespa,
    *,
    schema: str,
    documents: Iterable[dict[str, Any]],
) -> int:
    count = 0
    for document in documents:
        app.feed_data_point(
            schema=schema,
            data_id=document["id"],
            fields=clean_vespa_fields(document["fields"]),
        )
        count += 1
    return count
