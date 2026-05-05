"""
src/youtube/fetch_transcripts.py
=================================
Fetches video metadata + transcripts from the CNRA-RCAR YouTube channel.

Usage:
    python src/youtube/fetch_transcripts.py
    python src/youtube/fetch_transcripts.py --max-videos 10  # test avec 10 vidéos
    python src/youtube/fetch_transcripts.py --channel-url https://www.youtube.com/@CNRA_RCAR/videos

Stratégie de langue:
    1. Transcript français (fr) si disponible
    2. Transcript arabe (ar) traduit en français via YouTube
    3. N'importe quel transcript disponible (la langue est notée dans les métadonnées)
    Le modèle BAAI/bge-m3 étant multilingue, même les transcripts arabes non traduits
    fonctionnent bien pour la recherche sémantique depuis des requêtes en français.

Sortie:
    data/youtube/transcripts.json  ← persisté incrémentalement
"""

import argparse
import json
import sys
import time
import os
import random
import subprocess
from pathlib import Path

# Configuration du PATH pour inclure ffmpeg localement pour Windows
if os.name == 'nt':
    ffmpeg_dir = r"C:\Users\kamal\AppData\Local\Microsoft\WinGet\Links"
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_dir

from loguru import logger

# Ajout du dossier src au path si lancé en direct
src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from config.settings import BASE_DIR, LOGS_DIR
    from config.logger import setup_logger
    setup_logger(LOGS_DIR, source="youtube_fetch")
except Exception:
    BASE_DIR = Path(__file__).resolve().parents[2]
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOGS_DIR / "youtube_fetch.log", level="INFO")

CHANNEL_URL = "https://www.youtube.com/@CNRA_RCAR/videos"
OUTPUT_DIR = BASE_DIR / "data" / "youtube"
AUDIO_DIR = OUTPUT_DIR / "audios"
PREFERRED_LANGS = ["fr", "ar"]  # Priorité : français d'abord


# ─────────────────────────────────────────────────────────────────────────────
# Listing des vidéos via yt-dlp
# ─────────────────────────────────────────────────────────────────────────────

def fetch_channel_videos(channel_url: str = CHANNEL_URL, max_videos: int = 0) -> list[dict]:
    """
    Liste toutes les vidéos de la chaîne via yt-dlp (sans API key).
    Retourne une liste de dicts: video_id, title, url, duration, upload_date.
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp non installé. Lance: pip install yt-dlp")
        return []

    ydl_opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
    }

    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            if not info:
                logger.error("yt-dlp n'a retourné aucune info pour: {}", channel_url)
                return []

            entries = info.get("entries") or []
            if max_videos > 0:
                entries = entries[:max_videos]

            for entry in entries:
                if not entry or not entry.get("id"):
                    continue
                videos.append({
                    "video_id": entry["id"],
                    "title": entry.get("title", ""),
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                    "duration": entry.get("duration"),
                    "upload_date": entry.get("upload_date", ""),
                    "description": (entry.get("description") or "")[:500],
                })

        logger.info("Chaîne: {} vidéos trouvées", len(videos))
    except Exception as exc:
        logger.error("Erreur listing chaîne: {}", exc)

    return videos


# ─────────────────────────────────────────────────────────────────────────────
# Récupération des transcripts
# ─────────────────────────────────────────────────────────────────────────────

def fetch_transcript(video_id: str, preferred_langs: list[str] = PREFERRED_LANGS, max_retries: int = 3) -> dict | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        logger.error("Installe: pip install youtube-transcript-api")
        return None

    def segments_to_text(segments) -> str:
        return " ".join([seg.text for seg in segments]).strip()

    api = YouTubeTranscriptApi()

    for attempt in range(1, max_retries + 1):
        try:
            # 🔥 1. Essayer de récupérer le transcript
            # On demande d'abord les langues préférées sinon ça fallback sur les autres
            transcript_list = YouTubeTranscriptApi().list(video_id)
            
            transcript_obj = None
            # Chercher d'abord dans nos langues préférées
            for lang in preferred_langs:
                try:
                    transcript_obj = transcript_list.find_transcript([lang])
                    break
                except:
                    continue
            
            # Sinon on prend ce qu'il y a
            if not transcript_obj:
                # Cherche n'importe quelle langue générée ou manuelle
                transcript_obj = next(iter(transcript_list))
                
            data = transcript_obj.fetch()
            
            return {
                "text": segments_to_text(data),
                "language": transcript_obj.language_code,
                "translated": False,
            }

        except Exception as exc:
            error_str = str(exc).lower()
            
            # Gestion des blocages d'IP (429, Too Many Requests)
            if "too many requests" in error_str or "rate limit" in error_str or "429" in error_str or "urllib.error.httperror" in error_str:
                if attempt < max_retries:
                    # Exponential backoff avec un jitter (aléatoire) pour éviter de spammer
                    wait_time = (2 ** attempt) + random.uniform(3, 7)
                    logger.warning(f"Blocage IP / Rate limit (429) YouTube pour {video_id}. Attente {wait_time:.2f}s avant réessai ({attempt}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Echec définitif API après {max_retries} tentatives pour {video_id}: {exc}")
                    return None
            else:
                # Pas de sous-titres trouvés ou vidéo bloquée (vidéo privée, pas de paroles, etc.)
                logger.debug("Erreur API ou Pas de transcript pour {} : {}", video_id, exc)
                return None

    return None

# ─────────────────────────────────────────────────────────────────────────────
# Fallback Whisper
# ─────────────────────────────────────────────────────────────────────────────
_whisper_model = None

def load_whisper():
    global _whisper_model
    if not _whisper_model:
        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Chargement du modèle Whisper (medium) sur le device: {device}...")
        _whisper_model = whisper.load_model("small", device=device)
    return _whisper_model

def download_and_transcribe_audio(video_id: str) -> str | None:
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp non installé.")
        return None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(AUDIO_DIR / f"{video_id}.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as e:
        logger.error("Impossible de télécharger l'audio pour {}: {}", video_id, e)
        return None

    # Trouver le fichier téléchargé
    downloaded_file = None
    for ext in ["webm", "m4a", "mp3", "wav"]:
        tmp_path = str(AUDIO_DIR / f"{video_id}.{ext}")
        if os.path.exists(tmp_path):
            downloaded_file = tmp_path
            break
            
    if not downloaded_file:
        logger.error("Fichier audio non trouvé sur le disque pour {}", video_id)
        return None

    # Conversion en wav
    wav_file = downloaded_file.rsplit(".", 1)[0] + "_processed.wav"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", downloaded_file,
            "-ar", "16000",
            "-ac", "1",
            wav_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        logger.error("Erreur ffmpeg pour {}: {}", video_id, e)
        if os.path.exists(downloaded_file): os.remove(downloaded_file)
        return None

    # Transcrire
    try:
        result = load_whisper().transcribe(wav_file)
        text = result.get("text", "").strip()
    except Exception as e:
        logger.error("Erreur Whisper pour {}: {}", video_id, e)
        text = None
        
    # Nettoyage
    for f in [downloaded_file, wav_file]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    return text

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_transcripts(
    channel_url: str = CHANNEL_URL,
    output_dir: Path = OUTPUT_DIR,
    max_videos: int = 0,
    delay: float = 1.5,
) -> list[dict]:
    """
    Récupère tous les transcripts de la chaîne et sauvegarde en JSON.
    La sauvegarde est incrémentale : relancer ne re-télécharge pas ce qui existe déjà.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "transcripts.json"

    # Charger l'existant pour reprise incrémentale
    existing: dict[str, dict] = {}
    if output_file.exists():
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            existing = {v["video_id"]: v for v in data if isinstance(v, dict)}
            logger.info("Reprise : {} transcripts déjà en cache", len(existing))
        except Exception as exc:
            logger.warning("Cache illisible, repartir de zéro : {}", exc)

    videos = fetch_channel_videos(channel_url, max_videos)
    if not videos:
        logger.error("Aucune vidéo récupérée.")
        return []

    results_dict: dict[str, dict] = existing.copy()
    new_count = 0

    for i, video in enumerate(videos):
        vid_id = video["video_id"]

        # Critères de skip: Existe ET a un transcript ET transcript est assez long
        if vid_id in results_dict:
            rec = results_dict[vid_id]
            t_text = rec.get("transcript", "")
            if rec.get("has_transcript") and t_text and len(t_text.split()) > 20:
                logger.debug("Skip: Vidéo déjà traitée avec transcript complet: {}", vid_id)
                continue

        logger.info("[{}/{}] Traitement: {}", i + 1, len(videos), video["title"][:70])
        
        # 1. API YouTube
        transcript_data = fetch_transcript(vid_id)
        
        source = "youtube_api"
        text = transcript_data["text"] if transcript_data else None
        
        # 2. Fallback Whisper si insuffisant
        if not text or len(text.split()) < 20:
            logger.warning("Transcript API absent ou trop court (<20 mots) pour {}. Bascule sur Whisper...", vid_id)
            whisper_text = download_and_transcribe_audio(vid_id)
            if whisper_text and len(whisper_text.split()) >= 20:
                text = whisper_text
                source = "whisper"
                transcript_data = {
                    "language": "unknown (whisper)",
                    "translated": False,
                    "original_lang": "audio"
                }
            else:
                logger.error("Échec du fallback Whisper pour {}", vid_id)

        # Enregistrement
        record = {
            **video,
            "transcript": text,
            "source": source if text else None,
            "transcript_language": transcript_data.get("language") if transcript_data else None,
            "transcript_translated": transcript_data.get("translated", False) if transcript_data else False,
            "transcript_original_lang": transcript_data.get("original_lang") if transcript_data else None,
            "has_transcript": bool(text and len(text.split()) >= 20),
        }
        results_dict[vid_id] = record
        new_count += 1
        
        preview = (text[:60].replace('\n', ' ') + '...') if text else "AUCUN TEXTE"
        logger.info("Résultat {}: {} | Source: {} | Preview: {}", vid_id, "SUCCES" if record["has_transcript"] else "ECHEC", source, preview)

        # Sauvegarde (on réécrit tout le dictionnaire en liste JSON, ça overwrite proprement)
        output_file.write_text(
            json.dumps(list(results_dict.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Anti-ban : attendre entre les requêtes, même en cas de succès, pour ne pas trigger les limites de l'API YouTube
        actual_delay = delay + random.uniform(0.5, 2.0)
        time.sleep(actual_delay)

        if i < len(videos) - 1:
            # Délai random pour contrer le ban IP (429)
            time.sleep(delay + random.uniform(1.0, 3.0))

    final_results = list(results_dict.values())
    with_transcript = sum(1 for r in final_results if r.get("has_transcript"))
    logger.info(
        "Terminé: {}/{} vidéos avec transcript ({} traitées lors de cette session)",
        with_transcript, len(final_results), new_count,
    )
    logger.info("Fichier: {}", output_file)
    return final_results

# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch YouTube transcripts for CNRA-RCAR channel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python src/youtube/fetch_transcripts.py
  python src/youtube/fetch_transcripts.py --max-videos 10
  python src/youtube/fetch_transcripts.py --channel-url https://www.youtube.com/@CNRA_RCAR/videos
        """,
    )
    parser.add_argument("--channel-url", default=CHANNEL_URL, help="URL de la chaîne YouTube")
    parser.add_argument(
        "--max-videos", type=int, default=0,
        help="Nombre max de vidéos (0 = toutes)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Délai entre requêtes en secondes (défaut: 1.5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Dossier de sortie pour transcripts.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = fetch_all_transcripts(
        channel_url=args.channel_url,
        output_dir=Path(args.output_dir),
        max_videos=args.max_videos,
        delay=args.delay,
    )
    print(f"\nRésultat: {sum(1 for r in results if r.get('has_transcript'))}/{len(results)} vidéos avec transcript")