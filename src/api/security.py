from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import settings

try:
    from passlib.context import CryptContext
except ImportError:  # pragma: no cover - fallback exists for minimal installs
    CryptContext = None  # type: ignore[assignment]

try:
    from jose import JWTError, jwt
except ImportError:  # pragma: no cover - fallback exists for minimal installs
    JWTError = Exception  # type: ignore[assignment]
    jwt = None  # type: ignore[assignment]


pwd_context = (
    CryptContext(schemes=["bcrypt"], deprecated="auto")
    if CryptContext and os.getenv("API_PASSWORD_HASH_SCHEME", "pbkdf2").lower() == "bcrypt"
    else None
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def get_password_hash(password: str) -> str:
    if pwd_context is not None:
        return pwd_context.hash(password)

    salt = secrets.token_bytes(16)
    iterations = 260000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(iterations, _b64url(salt), _b64url(digest))


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if pwd_context is not None and not hashed_password.startswith("pbkdf2_sha256$"):
        try:
            return bool(pwd_context.verify(plain_password, hashed_password))
        except Exception:
            return False

    try:
        scheme, iterations, salt, digest = hashed_password.split("$", 3)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    expected = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        _b64url_decode(salt),
        int(iterations),
    )
    return hmac.compare_digest(_b64url(expected), digest)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.access_token_minutes))
    payload: dict[str, Any] = {"sub": str(subject), "exp": int(expire.timestamp())}
    if jwt is not None:
        return str(jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm))

    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = "{}.{}".format(
        _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
    )
    signature = hmac.new(settings.jwt_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return "{}.{}".format(signing_input, _b64url(signature))


def decode_access_token(token: str) -> dict[str, Any]:
    if jwt is not None:
        return dict(jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]))

    try:
        header_part, payload_part, signature_part = token.split(".", 2)
        signing_input = "{}.{}".format(header_part, payload_part)
        expected = hmac.new(settings.jwt_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url(expected), signature_part):
            raise ValueError("bad signature")
        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(datetime.now(timezone.utc).timestamp()):
            raise ValueError("expired token")
        return payload
    except Exception as exc:
        raise JWTError(str(exc)) from exc


def make_api_key() -> str:
    return _b64url(os.urandom(32))
