"""Automate the local reproducibility setup for the RCAR/CNRA RAG chatbot.

The script intentionally orchestrates existing Docker Compose services instead of
duplicating indexing logic. It prepares the long-running prerequisites:

- build Docker images;
- start infrastructure services;
- prepare the Ollama model;
- deploy Vespa schemas;
- download embedding/reranker models into shared Docker volumes;
- index documents, forms and videos.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
VESPA_APP = ROOT / "src" / "store" / "vespa"

CORE_SERVICES = ["vespa", "ollama", "postgres", "redis"]
INDEX_SERVICES = ["index-docs", "index-forms", "index-videos"]
MODEL_SERVICES = ["download-embeddings", "download-reranker"]


def log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def fail(message: str, code: int = 1) -> None:
    print(f"[setup][error] {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def run(command: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    log(" ".join(command))
    return subprocess.run(command, cwd=ROOT, check=check, text=True, env=env)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def merged_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(parse_env_file(ENV_FILE))
    return env


def ensure_env_file() -> None:
    if ENV_FILE.exists():
        log(".env already exists; it will not be overwritten.")
        return
    if not ENV_EXAMPLE.exists():
        fail(".env.example is missing; cannot create .env template.")
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    log(".env created from .env.example. Fill API keys before running a full setup.")


def warn_or_fail_missing_keys(env: dict[str, str], *, strict: bool) -> None:
    required_for_full_quality = {
        "OLLAMA_API_KEY": "Ollama Cloud authentication or already signed-in Ollama volume",
        "MISTRAL_API_KEY": "Mistral OCR for scanned documents",
        "HF_TOKEN": "Hugging Face model downloads when authentication/rate limits apply",
    }
    optional_eval = {
        "CEREBRAS_API_KEY": "RAGAS judge evaluation",
        "CEREBRAS_API_KEY_1": "legacy RAGAS judge evaluation variable",
    }

    missing = [key for key in required_for_full_quality if not env.get(key)]
    if missing:
        message = "Missing API keys for full setup: " + ", ".join(missing)
        if strict:
            fail(message)
        log(message + ". Continuing because --strict-keys was not set.")

    if not any(env.get(key) for key in optional_eval):
        log("Cerebras key not found; RAGAS judging will require adding one later.")


def ensure_sources() -> None:
    sources = [
        ROOT / "data" / "supportstagerag",
        ROOT / "data" / "forms",
        ROOT / "data" / "youtube",
    ]
    missing = [str(path.relative_to(ROOT)) for path in sources if not path.exists()]
    if missing:
        fail("Missing source data directories: " + ", ".join(missing))


def wait_http(url: str, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    log(f"Ready: {url}")
                    return
        except URLError:
            pass
        time.sleep(3)
    fail(f"Timed out waiting for {url}")


def wait_container_health(container_name: str, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        status = result.stdout.strip()
        if result.returncode == 0 and status == "healthy":
            log(f"Container healthy: {container_name}")
            return
        if result.returncode == 0 and not status:
            log(f"Container has no healthcheck, continuing: {container_name}")
            return
        time.sleep(3)
    fail(f"Timed out waiting for container health: {container_name}")


def deploy_vespa_schema() -> None:
    if not VESPA_APP.exists():
        fail(f"Vespa application package not found: {VESPA_APP}")
    target = "/tmp/rcarcnra-vespa-app"
    run(["docker", "exec", "vespa", "rm", "-rf", target])
    run(["docker", "cp", str(VESPA_APP), f"vespa:{target}"])
    run(["docker", "exec", "vespa", "vespa", "deploy", target, "--wait", "120"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a reproducible local setup for the RAG chatbot.")
    parser.add_argument("--skip-build", action="store_true", help="Do not run docker compose build.")
    parser.add_argument("--skip-ollama", action="store_true", help="Do not prepare/pull the Ollama model.")
    parser.add_argument("--skip-vespa-deploy", action="store_true", help="Do not deploy Vespa schemas.")
    parser.add_argument("--skip-model-downloads", action="store_true", help="Do not pre-download embedding/reranker models.")
    parser.add_argument("--skip-index", action="store_true", help="Do not index docs/forms/videos.")
    parser.add_argument("--skip-app-start", action="store_true", help="Do not start API and UI at the end.")
    parser.add_argument("--strict-keys", action="store_true", help="Fail when main provider API keys are missing.")
    parser.add_argument("--wait-seconds", type=int, default=240, help="Healthcheck timeout for services.")
    args = parser.parse_args()

    ensure_env_file()
    env = merged_env()
    warn_or_fail_missing_keys(env, strict=args.strict_keys)
    ensure_sources()

    if not args.skip_build:
        run(["docker", "compose", "build"], env=env)

    run(["docker", "compose", "up", "-d", *CORE_SERVICES], env=env)
    wait_http("http://localhost:19071/state/v1/health", timeout_seconds=args.wait_seconds)
    for container in CORE_SERVICES:
        wait_container_health(container, timeout_seconds=args.wait_seconds)

    if not args.skip_ollama:
        log("Preparing Ollama model. If this fails, run 'docker compose exec ollama ollama signin' and retry.")
        run(["docker", "compose", "--profile", "models", "run", "--rm", "prepare-ollama-cloud-model"], env=env)

    if not args.skip_vespa_deploy:
        deploy_vespa_schema()

    if not args.skip_model_downloads:
        for service in MODEL_SERVICES:
            run(["docker", "compose", "--profile", "index", "run", "--rm", service], env=env)

    if not args.skip_index:
        log("Indexing may take several minutes on first run.")
        for service in INDEX_SERVICES:
            run(["docker", "compose", "--profile", "index", "run", "--rm", service], env=env)

    if not args.skip_app_start:
        run(["docker", "compose", "up", "-d", "api", "ui"], env=env)
        wait_http("http://localhost:8000/health", timeout_seconds=args.wait_seconds)

    log("Setup complete.")
    log("UI: http://localhost:5173")
    log("API health: http://localhost:8000/health")
    log("Vespa health: http://localhost:19071/state/v1/health")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
