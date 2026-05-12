"""
src/audio/transcriber.py
=========================
Transcription audio → texte via Whisper (local, GPU-accelerated).

Usage autonome :
    transcriber = AudioTranscriber()
    text = transcriber.transcribe(audio_bytes)   # bytes depuis st.audio_input()
    text = transcriber.transcribe("path/to/file.wav")

Usage depuis app.py :
    @st.cache_resource
    def load_transcriber():
        return AudioTranscriber()

Design :
    - Modèle chargé une seule fois (singleton via @st.cache_resource)
    - Compatible avec le modèle Whisper déjà utilisé dans fetch_transcripts.py
    - Accepte bytes (st.audio_input) ou chemin fichier
    - Détection automatique GPU (RTX 3060 → cuda)
    - Nettoyage automatique des fichiers temporaires
"""

import io
import os
import re
import sys
import tempfile
from pathlib import Path

from loguru import logger

# ── Vocabulaire domaine injecté dans Whisper (initial_prompt) ─────────────────
# Whisper utilise ce texte comme contexte de départ pour biaiser la reconnaissance
# vers les termes métier RCAR/CNRA plutôt que leurs homophones courants.
# Ex : "RECORE" vs "record", "RCAR" vs "r'car", "CNRA" vs "sénar"…
DOMAIN_INITIAL_PROMPT = (
    "RCAR, CNRA, RECORE, CDG, FRAM, CRAC, retraite, pension, cotisation, "
    "allocation de retraite, affilié, non-titulaire, titulaire, rente viagère, "
    "invalidité, vieillesse, décès, pécule, liquidation, rachat, Dahir, "
    "branche épargne prévoyance, Caisse de Dépôt et de Gestion, "
    "collectivité locale, accident du travail, rente AT."
)

# ── Corrections post-transcription (homophones connus) ───────────────────────
# Dictionnaire général : {transcription_erronee_minuscule: terme_correct}
# Ajouter ici tout nouveau conflit détecté — solution générale et extensible.
DOMAIN_CORRECTIONS: dict = {
    # RECORE (produit CNRA) ← confondu avec le mot courant "record"
    "record":   "RECORE",
    "décor":    "RECORE",
    "recor":    "RECORE",
    "le core":  "RECORE",
    "re-core":  "RECORE",
    "re core":  "RECORE",
    # RCAR ← variantes phonétiques
    "r car":    "RCAR",
    "r-car":    "RCAR",
    "ercar":    "RCAR",
    "ar car":   "RCAR",
    # CNRA ← variantes phonétiques
    "c n r a":  "CNRA",
    "sénar":    "CNRA",
    "senra":    "CNRA",
    # FRAM
    "frame":    "FRAM",
    # CRAC ← confondu avec "crack"
    "crack":    "CRAC",
    # CDG
    "c d g":    "CDG",
    "cédégé":   "CDG",
    # Dahir
    "da hier":  "Dahir",
    "dahier":   "Dahir",
}

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

# Modèle Whisper chargé une seule fois pour toute la session
_whisper_model = None


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_whisper_model(model_size: str = "small") -> object:
    """Charge le modèle Whisper une seule fois (singleton global)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper non installé. Lance : pip install openai-whisper")

    device = _get_device()
    logger.info("Chargement Whisper '{}' sur {}...", model_size, device)
    _whisper_model = whisper.load_model(model_size, device=device)
    logger.info("Whisper prêt ({} | {})", model_size, device.upper())
    return _whisper_model


class AudioTranscriber:
    """
    Transcripteur audio → texte basé sur Whisper.

    Conçu pour s'intégrer dans l'app Streamlit sans aucune modification
    du RAGPipeline : il retourne simplement un str prêt pour rag.query().

    Paramètres :
        model_size   : "tiny" | "small" (défaut) | "medium"
                       small = bon équilibre vitesse/précision sur RTX 3060
        language     : "fr" forcé par défaut (évite l'auto-détection coûteuse)
                       Mettre None pour laisser Whisper détecter la langue.
    """

    def __init__(self, model_size: str = "small", language: str = "fr"):
        self.model_size = model_size
        self.language = language
        self.device = _get_device()
        self._model = _load_whisper_model(model_size)

    # ── API publique ──────────────────────────────────────────────────────────

    def transcribe(self, audio_input: bytes | str | Path) -> str:
        """
        Transcrit un audio en texte.

        Args:
            audio_input : bytes (depuis st.audio_input()),
                          str ou Path (chemin vers fichier audio)

        Returns:
            Texte transcrit, stripped. Chaîne vide si échec.
        """
        if isinstance(audio_input, (str, Path)):
            return self._transcribe_file(str(audio_input))

        if isinstance(audio_input, bytes):
            return self._transcribe_bytes(audio_input)

        # Streamlit audio_input retourne un UploadedFile (file-like)
        if hasattr(audio_input, "read"):
            return self._transcribe_bytes(audio_input.read())

        logger.error("Type audio non supporté : {}", type(audio_input))
        return ""

    @property
    def ready(self) -> bool:
        return self._model is not None

    # ── Implémentation interne ────────────────────────────────────────────────

    def _transcribe_file(self, path: str) -> str:
        """Transcrit depuis un chemin fichier."""
        try:
            result = self._model.transcribe(
                path,
                language=self.language,
                fp16=(self.device == "cuda"),
                verbose=False,
            )
            text = (result.get("text") or "").strip()
            logger.info("Transcription OK ({} chars) : {}", len(text), text[:60])
            return text
        except Exception as exc:
            logger.error("Erreur transcription fichier {}: {}", path, exc)
            return ""

    def _transcribe_bytes(self, audio_bytes: bytes) -> str:
        """
        Transcrit depuis des bytes bruts.
        Écrit dans un fichier temporaire (Whisper exige un fichier sur disque),
        puis nettoie automatiquement.
        """
        if not audio_bytes:
            return ""

        # Détection du format à partir du magic bytes
        suffix = _detect_audio_suffix(audio_bytes)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, prefix="stt_"
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            return self._transcribe_file(tmp_path)

        except Exception as exc:
            logger.error("Erreur transcription bytes : {}", exc)
            return ""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_audio_suffix(data: bytes) -> str:
    """
    Détecte le format audio depuis les magic bytes.
    Streamlit audio_input génère du .wav (PCM) ou .webm selon le navigateur.
    Fallback sur .wav (format universel supporté par Whisper).
    """
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return ".mp3"
    if data[:4] == b"fLaC":
        return ".flac"
    # Streamlit génère souvent du webm/opus dans les navigateurs modernes
    return ".webm"