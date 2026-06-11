'''src/preparation/download_embedding_model.py'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.embedding_cache import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_DIR,
    ensure_local_embedding_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and save the embedding model under data/models for local and Docker reuse.",
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--model-dir", default=str(DEFAULT_EMBEDDING_MODEL_DIR))
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite the saved model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = ensure_local_embedding_model(
        model_name=args.model,
        model_dir=args.model_dir,
        force=args.force,
    )
    print(f"Embedding model ready: {path}")


if __name__ == "__main__":
    main()
