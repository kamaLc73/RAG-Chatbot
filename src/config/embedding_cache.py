'''src/config/embedding_cache.py'''

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config.settings import BASE_DIR


load_dotenv(BASE_DIR / ".env")

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
_model_dir_env = Path(os.getenv("EMBEDDING_MODEL_DIR", str(BASE_DIR / "data" / "models" / "embeddings")))
DEFAULT_EMBEDDING_MODEL_DIR = _model_dir_env if _model_dir_env.is_absolute() else BASE_DIR / _model_dir_env
DEFAULT_HF_CACHE_DIR = BASE_DIR / "data" / "models" / "huggingface"
DEFAULT_ST_CACHE_DIR = BASE_DIR / "data" / "models" / "sentence_transformers"

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_CACHE_DIR / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(DEFAULT_ST_CACHE_DIR))


def _safe_model_dir_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name.strip()).strip("_") or "embedding_model"


def _looks_like_saved_sentence_transformer(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "modules.json").exists()


def resolve_embedding_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def local_embedding_model_path(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_dir: str | Path = DEFAULT_EMBEDDING_MODEL_DIR,
) -> Path:
    candidate = Path(model_name)
    if candidate.exists():
        return candidate.resolve()
    return Path(model_dir).resolve() / _safe_model_dir_name(model_name)


def ensure_local_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_dir: str | Path = DEFAULT_EMBEDDING_MODEL_DIR,
    *,
    force: bool = False,
) -> Path:
    target = local_embedding_model_path(model_name=model_name, model_dir=model_dir)
    if _looks_like_saved_sentence_transformer(target) and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    model_kwargs: dict[str, Any] = {}
    if hf_token:
        model_kwargs["token"] = hf_token

    model = SentenceTransformer(model_name, **model_kwargs)
    model.save(str(target))
    return target


def make_huggingface_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    model_dir: str | Path = DEFAULT_EMBEDDING_MODEL_DIR,
    device: str | None = None,
    force_download: bool = False,
    encode_kwargs: dict[str, Any] | None = None,
):
    local_model = ensure_local_embedding_model(
        model_name=model_name,
        model_dir=model_dir,
        force=force_download,
    )
    if device is None:
        device = resolve_embedding_device()

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=str(local_model),
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs or {"normalize_embeddings": True},
    )
