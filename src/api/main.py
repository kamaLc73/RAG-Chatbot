from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger as loguru_logger

from .config import settings
from .database import init_db
from .routers import auth, chat, conversations
from .routers.admin import conversations as admin_conversations
from .routers.admin import evaluation, intents, kb, stats, users
from ..config.logger import is_logger_initialized, setup_logger
from ..config.settings import LOGS_DIR


class LoguruForwardHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        loguru_logger.opt(exception=record.exc_info).log(level, record.getMessage())


def configure_logging() -> None:
    if not is_logger_initialized():
        setup_logger(log_dir=LOGS_DIR, source="api")
    logging.basicConfig(
        handlers=[LoguruForwardHandler()],
        level=logging.INFO,
        force=True,
    )


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.rag = None
    app.state.audio_transcriber = None
    await init_db()

    if settings.load_rag_on_startup:
        from src.chatbot.rag_pipeline import RAGPipeline

        app.state.rag = RAGPipeline()

    if settings.load_audio_on_startup:
        from src.audio.transcriber import AudioTranscriber

        app.state.audio_transcriber = AudioTranscriber(
            model_size=settings.audio_model_size,
            language=settings.audio_language,
        )

    yield


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(users.router)
app.include_router(admin_conversations.router)
app.include_router(kb.router)
app.include_router(intents.router)
app.include_router(evaluation.router)
app.include_router(stats.router)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "database": "configured",
        "rag_loaded": getattr(app.state, "rag", None) is not None,
        "audio_loaded": getattr(app.state, "audio_transcriber", None) is not None,
    }
