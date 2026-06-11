'''src/config/reranker_cache.py'''

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config.settings import BASE_DIR


load_dotenv(BASE_DIR / ".env")

DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_model_dir_env = Path(os.getenv("RERANKER_MODEL_DIR", str(BASE_DIR / "data" / "models" / "rerankers")))
DEFAULT_RERANKER_MODEL_DIR = _model_dir_env if _model_dir_env.is_absolute() else BASE_DIR / _model_dir_env
DEFAULT_HF_CACHE_DIR = BASE_DIR / "data" / "models" / "huggingface"
DEFAULT_ST_CACHE_DIR = BASE_DIR / "data" / "models" / "sentence_transformers"

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_CACHE_DIR / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(DEFAULT_ST_CACHE_DIR))


def _safe_model_dir_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name.strip()).strip("_") or "reranker_model"


def _looks_like_saved_cross_encoder(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_model_config = (path / "config.json").exists()
    has_tokenizer = any(
        (path / name).exists()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "sentencepiece.bpe.model")
    )
    return has_model_config and has_tokenizer


def local_reranker_model_path(
    model_name: str = DEFAULT_RERANKER_MODEL,
    model_dir: str | Path = DEFAULT_RERANKER_MODEL_DIR,
) -> Path:
    candidate = Path(model_name)
    if candidate.exists():
        return candidate.resolve()
    return Path(model_dir).resolve() / _safe_model_dir_name(model_name)


def ensure_local_reranker_model(
    model_name: str = DEFAULT_RERANKER_MODEL,
    model_dir: str | Path = DEFAULT_RERANKER_MODEL_DIR,
    *,
    force: bool = False,
) -> Path:
    target = local_reranker_model_path(model_name=model_name, model_dir=model_dir)
    if _looks_like_saved_cross_encoder(target) and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import CrossEncoder

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    model_kwargs: dict[str, Any] = {}
    if hf_token:
        model_kwargs["token"] = hf_token

    model = CrossEncoder(model_name, **model_kwargs)
    model.save(str(target))
    return target


def make_cross_encoder_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    model_dir: str | Path = DEFAULT_RERANKER_MODEL_DIR,
    max_length: int = 512,
    device: str | None = None,
    force_download: bool = False,
):
    local_model = ensure_local_reranker_model(
        model_name=model_name,
        model_dir=model_dir,
        force=force_download,
    )

    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        str(local_model),
        max_length=max_length,
        device=device,
    )
