from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Callable, TypeVar

from .config import settings


T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLMemoryCache:
    def __init__(self, default_ttl_seconds: float = 30.0) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self._entries: dict[str, CacheEntry] = {}
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        now = monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> Any:
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        with self._lock:
            self._entries[key] = CacheEntry(value=value, expires_at=monotonic() + ttl)
        return value

    def clear(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in list(self._entries):
                if key.startswith(prefix):
                    self._entries.pop(key, None)


class HybridCache:
    def __init__(
        self,
        *,
        default_ttl_seconds: float = 30.0,
        backend: str = "memory",
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "rag-chatbot:",
        socket_timeout: float = 0.25,
        connect_timeout: float = 0.25,
        redis_retry_after_seconds: float = 5.0,
    ) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self.backend = backend
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.socket_timeout = socket_timeout
        self.connect_timeout = connect_timeout
        self.redis_retry_after_seconds = redis_retry_after_seconds
        self.memory = TTLMemoryCache(default_ttl_seconds=default_ttl_seconds)
        self._redis: Any | None = None
        self._redis_retry_at = 0.0
        self._lock = RLock()

    def _redis_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def _redis_client(self) -> Any | None:
        if self.backend != "redis":
            return None
        now = monotonic()
        if now < self._redis_retry_at:
            return None
        with self._lock:
            if self._redis is not None:
                return self._redis
            try:
                import redis

                client = redis.Redis.from_url(
                    self.redis_url,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.connect_timeout,
                    decode_responses=False,
                )
                client.ping()
                self._redis = client
                return client
            except Exception as exc:
                self._redis = None
                self._redis_retry_at = monotonic() + self.redis_retry_after_seconds
                logger.debug("Redis cache indisponible, fallback memoire: %s", exc)
                return None

    def _mark_redis_failed(self, exc: Exception) -> None:
        with self._lock:
            self._redis = None
            self._redis_retry_at = monotonic() + self.redis_retry_after_seconds
        logger.debug("Redis cache erreur, fallback memoire: %s", exc)

    def get(self, key: str) -> Any | None:
        client = self._redis_client()
        if client is not None:
            try:
                raw = client.get(self._redis_key(key))
                if raw is not None:
                    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    return json.loads(text)
            except Exception as exc:
                self._mark_redis_failed(exc)
        return self.memory.get(key)

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> Any:
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        self.memory.set(key, value, ttl_seconds=ttl)
        client = self._redis_client()
        if client is not None:
            try:
                encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
                client.setex(self._redis_key(key), max(1, int(ttl)), encoded)
            except Exception as exc:
                self._mark_redis_failed(exc)
        return value

    def get_or_set(self, key: str, factory: Callable[[], T], ttl_seconds: float | None = None) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value, ttl_seconds=ttl_seconds)
        return value

    def clear(self, key: str) -> None:
        self.memory.clear(key)
        client = self._redis_client()
        if client is not None:
            try:
                client.delete(self._redis_key(key))
            except Exception as exc:
                self._mark_redis_failed(exc)

    def clear_prefix(self, prefix: str) -> None:
        self.memory.clear_prefix(prefix)
        client = self._redis_client()
        if client is not None:
            try:
                pattern = self._redis_key(f"{prefix}*")
                keys = list(client.scan_iter(match=pattern, count=200))
                if keys:
                    client.delete(*keys)
            except Exception as exc:
                self._mark_redis_failed(exc)

    def backend_status(self) -> dict[str, object]:
        client = self._redis_client()
        if client is None:
            return {"backend": "memory" if self.backend != "redis" else "memory_fallback", "redis": False}
        return {"backend": "redis", "redis": True}


api_cache = HybridCache(
    default_ttl_seconds=30.0,
    backend=settings.cache_backend,
    redis_url=settings.redis_url,
    key_prefix=settings.cache_key_prefix,
    socket_timeout=settings.redis_socket_timeout_seconds,
    connect_timeout=settings.redis_connect_timeout_seconds,
    redis_retry_after_seconds=settings.redis_retry_after_seconds,
)
