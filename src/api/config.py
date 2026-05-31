from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: str = "") -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("API_APP_NAME", "RAG Chatbot API")
    debug: bool = _bool_env("API_DEBUG", False)
    database_url: str = os.getenv(
        "DATABASE_URL",
        os.getenv("API_DATABASE_URL", "postgresql+asyncpg://chatbot:chatbot_pass@localhost:5432/chatbot_db"),
    )
    cors_origins: list[str] = None  # type: ignore[assignment]
    jwt_secret: str = os.getenv("JWT_SECRET_KEY", os.getenv("API_JWT_SECRET", "change-me-local-dev"))
    jwt_algorithm: str = os.getenv("API_JWT_ALGORITHM", "HS256")
    access_token_minutes: int = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", os.getenv("API_ACCESS_TOKEN_MINUTES", "120"))
    )
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    load_rag_on_startup: bool = _bool_env("API_LOAD_RAG_ON_STARTUP", True)
    load_audio_on_startup: bool = _bool_env("API_LOAD_AUDIO_ON_STARTUP", True)
    audio_model_size: str = os.getenv("API_AUDIO_MODEL_SIZE", "small")
    audio_language: str = os.getenv("API_AUDIO_LANGUAGE", "fr")
    bootstrap_superuser_email: str = os.getenv("API_BOOTSTRAP_SUPERUSER_EMAIL", "")
    bootstrap_superuser_username: str = os.getenv("API_BOOTSTRAP_SUPERUSER_USERNAME", "")
    bootstrap_superuser_full_name: str = os.getenv("API_BOOTSTRAP_SUPERUSER_FULL_NAME", "Admin local")
    bootstrap_superuser_password: str = os.getenv("API_BOOTSTRAP_SUPERUSER_PASSWORD", "")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cors_origins",
            _csv_env(
                "API_CORS_ORIGINS",
                f"{self.frontend_url},http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
            ),
        )

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def async_database_url(self) -> str:
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if self.database_url.startswith("sqlite:///") and not self.database_url.startswith("sqlite+aiosqlite:///"):
            return self.database_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        return self.database_url


settings = Settings()
