import html
import os
import time
import re

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from loguru import logger

from chatbot.rag_pipeline import RAGPipeline
from audio.transcriber import AudioTranscriber
from config.logger import setup_logger
from config.settings import LOGS_DIR

st.set_page_config(
    page_title="Assistant RCAR/CNRA",
    page_icon="🌿",
    layout="centered",
)

if "logger_initialized" not in st.session_state:
    setup_logger(log_dir=LOGS_DIR, source="chatbot")
    st.session_state["logger_initialized"] = True


# ── Constantes UI par organisme ───────────────────────────────────────────────
ORG_LABELS = {
    "all":  "RCAR & CNRA",
    "rcar": "RCAR",
    "cnra": "CNRA",
}
ORG_TITLES = {
    "all":  "Assistant RCAR / CNRA",
    "rcar": "Assistant RCAR",
    "cnra": "Assistant CNRA",
}
ORG_SUBTITLES = {
    "all":  "Interface Chatbot — sites RCAR et CNRA",
    "rcar": "Régime Collectif d'Allocation de Retraite",
    "cnra": "Caisse Nationale de Retraites et d'Assurances",
}
ORG_WELCOME = {
    "all":  "Bonjour. Je suis votre assistant RCAR/CNRA. Posez-moi une question sur vos droits, démarches ou formulaires.",
    "rcar": "Bonjour. Je suis votre assistant RCAR. Je réponds à vos questions sur le régime de retraite complémentaire.",
    "cnra": "Bonjour. Je suis votre assistant CNRA. Je réponds à vos questions sur les rentes, assurances et prestations.",
}


def get_welcome_message(org: str = "all") -> str:
    return ORG_WELCOME.get(org, ORG_WELCOME["all"])


# ── Session state ─────────────────────────────────────────────────────────────
if "theme"      not in st.session_state: st.session_state["theme"] = "dark"
if "selected_org" not in st.session_state: st.session_state["selected_org"] = "all"
if "history"    not in st.session_state:
    st.session_state["history"] = [{"type": "bot", "content": get_welcome_message("all"), "videos": [], "forms": []}]
if "processing"        not in st.session_state: st.session_state["processing"] = False
if "show_typewriter"   not in st.session_state: st.session_state["show_typewriter"] = False
if "typewriter_message" not in st.session_state: st.session_state["typewriter_message"] = ""
if "rag_preload_attempted" not in st.session_state: st.session_state["rag_preload_attempted"] = False
if "rag_preload_error"     not in st.session_state: st.session_state["rag_preload_error"] = ""
if "transcriber_preload_attempted" not in st.session_state: st.session_state["transcriber_preload_attempted"] = False
if "transcriber_preload_error"     not in st.session_state: st.session_state["transcriber_preload_error"] = ""
if "last_audio_id"         not in st.session_state: st.session_state["last_audio_id"] = None
if "voice_query"           not in st.session_state: st.session_state["voice_query"] = ""


def toggle_theme() -> None:
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"


# ── Couleurs selon thème ──────────────────────────────────────────────────────
dark = st.session_state["theme"] == "dark"

title_color     = "#ffffff"        if dark else "#0f5132"
subtitle_color  = "#9dd7be"        if dark else "#1f6e4a"
bg_color        = "#0f1f1b"        if dark else "#f6fbf8"
user_msg_bg     = "#1f7a5c"        if dark else "#0f7a5a"
bot_msg_bg      = "#eef5f1"        if dark else "#e6f1ec"
bot_msg_color   = "#1a2a25"        if dark else "#17332a"
spinner_color   = "#d8f3e8"        if dark else "#0f5132"

# Cartes vidéo
video_card_bg           = "#162a23"  if dark else "#e8f5ee"
video_card_border       = "#2a5043"  if dark else "#a8d5bc"
video_card_title_color  = "#9dd7be"  if dark else "#0f5132"
video_card_text_color   = "#c5ddd5"  if dark else "#2d5a42"
video_badge_bg          = "#1f4a39"  if dark else "#c5e8d4"
video_badge_color       = "#9dd7be"  if dark else "#0a3f27"

# Cartes formulaire — ton bleu/indigo distinct des vidéos vertes
form_card_bg            = "#162030"  if dark else "#eaf0f9"
form_card_border        = "#2a3f5a"  if dark else "#a8c0e0"
form_card_title_color   = "#9bbde8"  if dark else "#1a4a8a"
form_card_text_color    = "#b8cfe8"  if dark else "#2d4a6a"
form_badge_bg           = "#1e3355"  if dark else "#c5d8f4"
form_badge_color        = "#9bbde8"  if dark else "#0a2f6a"
form_org_cnra_color     = "#e8b89b"  if dark else "#a03010"  # orange pour CNRA
form_org_rcar_color     = "#9bbde8"  if dark else "#1a4a8a"  # bleu pour RCAR


theme_css = f"""
<style>
    .stApp {{ background-color: {bg_color}; }}
    .stTextInput > div > div > input {{
        background-color: {"#1f2f2b" if dark else "#ffffff"};
        color: {"#ffffff" if dark else "#000000"};
        border: 1px solid {"#35564d" if dark else "#b7d0c4"};
    }}
    .stTextInput > div > div > input::placeholder {{
        color: {"#9aa9a3" if dark else "#6b7f75"} !important;
    }}
    .stTextInput * {{ color: {"#ffffff" if dark else "#000000"} !important; }}
    .stTextInput > label {{ color: {"#ffffff" if dark else "#000000"} !important; font-weight: 500; }}
    .stForm {{ background-color: transparent; border: none; border-radius: 8px; }}
    .stFormSubmitButton > button {{
        background-color: {"#1f7a5c" if dark else "#0f5132"};
        color: white; border: 1px solid transparent; border-radius: 16px;
        padding: 10px 18px; font-weight: 600; box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: all 0.2s ease-in-out;
        white-space: nowrap;
        min-width: 110px;
    }}
    .stFormSubmitButton > button:hover {{
        filter: brightness(1.08);
        transform: translateY(-1px);
    }}
    .stButton > button {{
        background-color: {"#325248" if dark else "#5f7a70"};
        color: white; border: 1px solid {"#35564d" if dark else "#b7d0c4"};
        border-radius: 16px; padding: 10px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: all 0.2s ease-in-out;
    }}
    .stButton > button:hover {{
        filter: brightness(1.08);
        transform: translateY(-1px);
    }}
    button[kind="primary"] {{
        background-color: {"#1f7a5c" if dark else "#0f5132"};
        border: 1px solid transparent;
    }}
    .stMarkdown p {{ color: {"#ffffff" if dark else "#000000"}; }}

    div[data-testid="stAudioInput"] {{
        max-width: 220px;
        width: 100%;
        margin-left: auto;
        margin-right: 0;
        min-height: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        background-color: transparent;
        border: none;
        box-shadow: none;
        padding: 0;
    }}
    div[data-testid="stAudioInput"] > div {{
        height: 42px;
        min-height: 42px;
        display: flex;
        align-items: center;
    }}
    div[data-testid="stAudioInput"] button {{
        min-height: 42px;
        height: 42px;
        padding: 0 12px;
    }}
    div[data-testid="stAudioInput"] * {{
        color: inherit;
    }}

    div[data-testid="stAudioInput"] {{
        position: relative;
        z-index: 20;
    }}

    /* popup menu */
    div[data-testid="stAudioInput"] [role="menu"] {{
        background: #1A2433 !important;
        border: 1px solid #2E4057 !important;
        border-radius: 14px !important;
        padding: 6px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.45) !important;
        min-width: 180px !important;
    }}

    /* menu buttons */
    div[data-testid="stAudioInput"] [role="menu"] button {{
        background: transparent !important;
        color: #F4F7FB !important;
        border-radius: 10px !important;
        transition: all 0.15s ease-in-out !important;
        font-weight: 500 !important;
    }}

    /* hover */
    div[data-testid="stAudioInput"] [role="menu"] button:hover {{
        background: #24364D !important;
        color: #38BDF8 !important;
    }}

    /* timer text */
    div[data-testid="stAudioInput"] span {{
        color: #DCE7F3 !important;
    }}

    /* dots/options button */
    div[data-testid="stAudioInput"] button[kind="secondary"] {{
        background: #132235 !important;
        border: 1px solid #28435E !important;
        color: #F4F7FB !important;
    }}

    /* mic/play buttons */
    div[data-testid="stAudioInput"] button {{
        color: #F4F7FB !important;
    }}

    /* fix clipping */
    div[data-testid="stAudioInput"] > div {{
        overflow: visible !important;
    }}

    /* ensure popover visible */
    [data-baseweb="popover"] {{
        z-index: 99999 !important;
    }}
    .stSpinner {{ display: none !important; }}
    .custom-spinner {{ display:flex;flex-direction:column;justify-content:center;align-items:center;margin:20px 0; }}
    .custom-spinner .spinner-ring {{
        width:34px;height:34px;
        border:4px solid {"#35564d" if dark else "#b7d0c4"};
        border-top:4px solid {"#d8f3e8" if dark else "#0f5132"};
        border-radius:50%;animation:spin 1s linear infinite;
    }}
    @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}

    .user-message {{
        background-color:{user_msg_bg};color:white;padding:14px 18px;
        border-radius:16px 16px 0 16px;margin-bottom:16px;
        width:fit-content;min-width:30%;max-width:70%;
        word-break:break-word;white-space:pre-line;margin-left:auto;text-align:right;display:block;
    }}
    .bot-message {{
        background-color:{bot_msg_bg};color:{bot_msg_color};
        padding:10px 12px;border-radius:12px 12px 12px 0;
        margin-bottom:8px;max-width:70%;word-break:break-word;white-space:pre-line;
    }}
    .bot-message * {{ color:{bot_msg_color} !important; }}
    .typewriter-message {{
        background-color:{bot_msg_bg};color:{bot_msg_color};
        padding:10px 12px;border-radius:12px 12px 12px 0;
        margin-bottom:8px;max-width:70%;word-break:break-word;white-space:pre-line;
        border-right:2px solid {bot_msg_color};
        animation:blink-caret 0.75s step-end infinite;
    }}
    .typewriter-message * {{ color:{bot_msg_color} !important; }}
    @keyframes blink-caret {{ from,to{{border-color:transparent}} 50%{{border-color:{bot_msg_color}}} }}
    .message-separator {{ margin:15px 0; }}
    .form-container {{ margin-top:30px;padding-top:20px;background-color:{bg_color}; }}

    /* ── Cartes vidéo ──────────────────────────────────────────────────────── */
    .video-suggestions {{ margin-top:12px;max-width:70%; }}
    .video-suggestions-label {{
        font-size:0.8rem;font-weight:600;color:{video_badge_color};
        background-color:{video_badge_bg};display:inline-block;
        padding:2px 10px;border-radius:999px;margin-bottom:8px;
    }}
    .video-card {{
        display:flex;gap:10px;align-items:flex-start;
        background-color:{video_card_bg};border:1px solid {video_card_border};
        border-radius:10px;padding:10px;margin-bottom:8px;
        text-decoration:none;transition:opacity .15s;
    }}
    .video-card:hover {{ opacity:0.85; }}
    .video-card img {{ width:100px;height:56px;object-fit:cover;border-radius:6px;flex-shrink:0; }}
    .video-card-info {{ flex:1;min-width:0; }}
    .video-card-title {{
        font-size:0.85rem;font-weight:600;color:{video_card_title_color};margin-bottom:4px;
        display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
    }}
    .video-card-excerpt {{
        font-size:0.75rem;color:{video_card_text_color};
        display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;opacity:0.85;
    }}
    .video-card-icon {{ font-size:0.75rem;color:{video_card_text_color};margin-top:4px;opacity:0.7; }}

    /* ── Cartes formulaire ─────────────────────────────────────────────────── */
    .form-suggestions {{ margin-top:10px;max-width:70%; }}
    .form-suggestions-label {{
        font-size:0.8rem;font-weight:600;color:{form_badge_color};
        background-color:{form_badge_bg};display:inline-block;
        padding:2px 10px;border-radius:999px;margin-bottom:8px;
    }}
    .form-card {{
        display:flex;gap:10px;align-items:flex-start;
        background-color:{form_card_bg};border:1px solid {form_card_border};
        border-radius:10px;padding:10px 12px;margin-bottom:8px;
        text-decoration:none;transition:opacity .15s;
    }}
    .form-card:hover {{ opacity:0.88; }}
    .form-card-icon-box {{
        width:36px;height:36px;border-radius:8px;display:flex;align-items:center;
        justify-content:center;flex-shrink:0;font-size:1.2rem;
        background-color:{form_badge_bg};
    }}
    .form-card-info {{ flex:1;min-width:0; }}
    .form-card-header {{ display:flex;align-items:center;gap:6px;margin-bottom:3px; }}
    .form-org-badge {{
        font-size:0.68rem;font-weight:700;padding:1px 7px;border-radius:999px;
    }}
    .form-org-cnra {{ background-color:#4a2000;color:{form_org_cnra_color}; }}
    .form-org-rcar {{ background-color:#0a1f40;color:{form_org_rcar_color}; }}
    .form-card-title {{
        font-size:0.83rem;font-weight:600;color:{form_card_title_color};
        display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
    }}
    .form-card-category {{ font-size:0.72rem;color:{form_card_text_color};opacity:0.85;margin-top:3px; }}
    .form-card-actions {{ display:flex;gap:8px;margin-top:6px;flex-wrap:wrap; }}
    .form-action-btn {{
        font-size:0.7rem;padding:2px 8px;border-radius:6px;
        text-decoration:none;font-weight:500;display:inline-block;
    }}
    .form-action-download {{
        background-color:{form_badge_bg};color:{form_badge_color};
        border:1px solid {form_card_border};
    }}
    .form-action-page {{
        background-color:transparent;color:{form_card_text_color};
        border:1px solid {form_card_border};opacity:0.8;
    }}
    .form-action-btn:hover {{ opacity:0.8; }}

    .bot-message a, .typewriter-message a {{
    color: {"#4db88a" if dark else "#0f5132"};
    text-decoration: underline;
    font-weight: 500;
    }}
    .bot-message a:hover, .typewriter-message a:hover {{
        opacity: 0.8;
    }}
    .bot-message strong, .typewriter-message strong {{
        font-weight: 700;
        color: {bot_msg_color};
    }}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Organisme")
    org = st.session_state["selected_org"]
    for org_key in ("all", "rcar", "cnra"):
        if st.button(
            ORG_LABELS[org_key],
            key=f"org_btn_{org_key}",
            use_container_width=True,
            type="primary" if st.session_state["selected_org"] == org_key else "secondary",
        ):
            if st.session_state["selected_org"] != org_key:
                st.session_state["selected_org"] = org_key
                st.session_state["history"] = [{
                    "type": "bot",
                    "content": get_welcome_message(org_key),
                    "videos": [],
                    "forms": [],
                }]
                st.rerun()

    st.markdown("---")
    st.markdown("### Paramètres")
    if st.button("Mode sombre" if dark else "Mode clair", key="theme_toggle"):
        toggle_theme()
        st.rerun()
    st.markdown("---")
    st.markdown("### Réinitialisation")
    if st.button("Effacer l'historique", type="secondary"):
        st.session_state["history"] = [{"type": "bot", "content": get_welcome_message(st.session_state["selected_org"]), "videos": [], "forms": []}]
        st.session_state["show_typewriter"] = False
        st.session_state["typewriter_message"] = ""
        st.session_state["processing"] = False
        st.success("Historique effacé")
        time.sleep(0.7)
        st.rerun()


# ── Titre ─────────────────────────────────────────────────────────────────────
org = st.session_state["selected_org"]
st.markdown(
    f"<h1 style='color:{title_color};margin-bottom:0.2rem;'>{ORG_TITLES[org]}</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='font-size:1rem;color:{subtitle_color};margin-bottom:1.2rem;'>{ORG_SUBTITLES[org]}</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag_pipeline() -> RAGPipeline:
    with st.spinner("Initialisation de l'assistant RCAR/CNRA..."):
        return RAGPipeline()


@st.cache_resource
def load_transcriber() -> AudioTranscriber:
    return AudioTranscriber(model_size="small", language="fr")


def preload_transcriber() -> None:
    if st.session_state["transcriber_preload_attempted"]:
        return
    st.session_state["transcriber_preload_attempted"] = True
    try:
        with st.spinner("Préchargement du module vocal..."):
            load_transcriber()
    except Exception as exc:
        st.session_state["transcriber_preload_error"] = str(exc)
        logger.error("Erreur préchargement Whisper: {}", exc)


def preload_rag_pipeline() -> None:
    if st.session_state["rag_preload_attempted"]:
        return
    st.session_state["rag_preload_attempted"] = True
    try:
        load_rag_pipeline()
    except Exception as exc:
        st.session_state["rag_preload_error"] = str(exc)
        logger.error("Erreur préchargement RAG: {}", exc)


def render_plain_text(text: str) -> str:
    if not text:
        return ""
    # Headers → texte simple
    text = re.sub(r'#{1,6}\s*', '', text)
    # Code inline
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text, flags=re.DOTALL)
    # Séparateurs
    text = re.sub(r'---+', '─' * 30, text)
    # Liens Markdown [texte](url) → HTML cliquable (avant l'escape)
    text = re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        r'<LINK_START href="\2" target="_blank" rel="noopener noreferrer">\1<LINK_END>',
        text,
    )
    # Escape HTML (protège le reste du texte)
    text = html.escape(text)
    # Restaure les balises <a> après l'escape
    text = text.replace('&lt;LINK_START ', '<a ')
    text = text.replace('&gt;', '>', 1) if '&lt;LINK_START' in text else text
    text = re.sub(r'&lt;LINK_START (.*?)&gt;', r'<a \1>', text)
    text = text.replace('&lt;LINK_END&gt;', '</a>')
    # Gras **texte** → <strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Italique *texte* → <em>  (après le gras pour éviter les conflits)
    text = re.sub(r'\*([^\*]+)\*', r'<em>\1</em>', text)
    # Sauts de ligne
    text = text.replace("\n", "<br>")
    return text
def render_form_cards(forms: list[dict]) -> str:
    if not forms:
        return ""

    cards = '<div class="form-suggestions"><div class="form-suggestions-label">📄 Formulaires disponibles</div>'

    for f in forms:
        title     = html.escape(f.get("title",    "Formulaire"))
        org       = html.escape(f.get("org",      ""))
        category  = html.escape(f.get("category", ""))
        pdf_url   = html.escape(f.get("pdf_url",  "#"))
        org_class = "form-org-cnra" if org == "CNRA" else "form-org-rcar"

        actions = (
            f'<div class="form-card-actions">'
            f'<a href="{pdf_url}" target="_blank" rel="noopener noreferrer" class="form-action-btn form-action-download">'
            f'⬇ Télécharger PDF</a></div>'
        ) if pdf_url != "#" else ""

        cards += (
            f'<div class="form-card">'
            f'<div class="form-card-icon-box">📄</div>'
            f'<div class="form-card-info">'
            f'<div class="form-card-header"><span class="form-org-badge {org_class}">{org}</span></div>'
            f'<div class="form-card-title">{title}</div>'
            f'<div class="form-card-category">{category}</div>'
            f'{actions}'
            f'</div>'
            f'</div>'
        )

    cards += "</div>"
    return cards


def render_video_cards(videos: list[dict]) -> str:
    if not videos:
        return ""

    cards = '<div class="video-suggestions">'
    cards += '<div class="video-suggestions-label">Videos suggerees</div>'

    for v in videos:
        title = html.escape(v.get("title", "Video CNRA/RCAR"), quote=True)
        url = html.escape(v.get("url", "#"), quote=True)
        video_id = v.get("video_id", "")
        thumbnail = html.escape(
            v.get("thumbnail_url") or f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
            quote=True,
        )
        excerpt = html.escape((v.get("excerpt") or v.get("description") or "")[:120], quote=True)
        excerpt_html = f"<div class='video-card-excerpt'>{excerpt}...</div>" if excerpt else ""

        cards += (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="video-card">'
            f'<img src="{thumbnail}" alt="" loading="lazy" onerror="this.style.display=\'none\'">'
            f'<div class="video-card-info">'
            f'<div class="video-card-title">{title}</div>'
            f'{excerpt_html}'
            f'<div class="video-card-icon">Voir sur YouTube</div>'
            f'</div>'
            f'</a>'
        )

    cards += "</div>"
    return cards


def stream_typewriter(text: str, placeholder, delay: float = 0.012) -> None:
    if not text:
        placeholder.markdown("<div class='bot-message'></div>", unsafe_allow_html=True)
        return
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(
            f"<div class='typewriter-message'>{render_plain_text(displayed)}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(delay)
    placeholder.markdown(
        f"<div class='bot-message'>{render_plain_text(text)}</div>",
        unsafe_allow_html=True,
    )


preload_rag_pipeline()
preload_transcriber()

# ── Historique ────────────────────────────────────────────────────────────────
with st.container():
    for i, message in enumerate(st.session_state["history"]):
        is_last = i == len(st.session_state["history"]) - 1

        if message["type"] == "user":
            st.markdown(
                f"<div class='user-message'>{render_plain_text(message['content'])}</div>",
                unsafe_allow_html=True,
            )
        elif message["type"] == "bot":
            should_animate = (
                st.session_state["show_typewriter"]
                and is_last
                and message["content"] == st.session_state["typewriter_message"]
                and bool(message["content"])
            )

            if should_animate:
                ph = st.empty()
                stream_typewriter(message["content"], ph)
                st.session_state["show_typewriter"] = False
                st.session_state["typewriter_message"] = ""
            else:
                is_typing = is_last and st.session_state["processing"] and not message["content"]
                cls = "typewriter-message" if is_typing else "bot-message"
                content = render_plain_text(message["content"]) if message["content"] else "..."
                st.markdown(f"<div class='{cls}'>{content}</div>", unsafe_allow_html=True)

            # Cartes vidéo
            if message.get("videos") and message["content"] and not (is_last and st.session_state["processing"]):
                html_v = render_video_cards(message["videos"])
                if html_v:
                    st.markdown(html_v, unsafe_allow_html=True)

            # Cartes formulaire
            if message.get("forms") and message["content"] and not (is_last and st.session_state["processing"]):
                html_f = render_form_cards(message["forms"])
                if html_f:
                    st.markdown(html_f, unsafe_allow_html=True)

            if not is_last:
                st.markdown("<div class='message-separator'></div>", unsafe_allow_html=True)


# ── Spinner de génération ─────────────────────────────────────────────────────
if st.session_state["processing"]:
    _, c2, _ = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            f"""<div class="custom-spinner">
                <div class="spinner-ring"></div>
                <div style="color:{spinner_color};font-weight:bold;font-size:1.05rem;margin-top:10px;">
                    Génération de la réponse...
                </div></div>""",
            unsafe_allow_html=True,
        )

# ── Zone de saisie ────────────────────────────────────────────────────────────
st.markdown('<div class="form-container">', unsafe_allow_html=True)

# ⚠️  CORRECTION : audio_input EN DEHORS des colonnes et AVANT le formulaire
# — exactement comme l'ancienne version qui fonctionnait.
# Le placer dans une colonne ou après un st.form corrompt son blob URL au rerun.
audio_value = st.audio_input("Message vocal", key="mic_input", label_visibility="collapsed")

# Formulaire texte
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Votre question", key="input_field",
        label_visibility="collapsed",
        placeholder="Tapez votre question ici...",
    )
    c1, c2 = st.columns([1, 6])
    with c1:
        submit = st.form_submit_button(
            "Envoyer",
            disabled=st.session_state["processing"],
        )

st.markdown("</div>", unsafe_allow_html=True)

# ── Traitement audio ──────────────────────────────────────────────────────────
if audio_value is not None:
    audio_id = hash(audio_value.read())
    audio_value.seek(0)
    if audio_id != st.session_state["last_audio_id"] and not st.session_state["processing"]:
        st.session_state["last_audio_id"] = audio_id
        with st.spinner("Transcription en cours..."):
            transcribed = load_transcriber().transcribe(audio_value.read())
        if transcribed:
            st.session_state["processing"] = True
            st.session_state["show_typewriter"] = False
            st.session_state["history"].extend([
                {"type": "user", "content": transcribed, "videos": [], "forms": []},
                {"type": "bot",  "content": "",          "videos": [], "forms": []},
            ])
            st.rerun()
        else:
            st.warning("Aucun texte détecté — réessayez.")

# ── Soumission texte ──────────────────────────────────────────────────────────
if submit and user_input and not st.session_state["processing"]:
    st.session_state["processing"] = True
    st.session_state["show_typewriter"] = False
    st.session_state["history"].extend([
        {"type": "user", "content": user_input,  "videos": [], "forms": []},
        {"type": "bot",  "content": "",           "videos": [], "forms": []},
    ])
    st.rerun()

# ── Génération ────────────────────────────────────────────────────────────────
if (
    st.session_state["history"]
    and st.session_state["history"][-1]["type"] == "bot"
    and st.session_state["history"][-1]["content"] == ""
    and st.session_state["processing"]
):
    user_msg = next(
        (m for m in reversed(st.session_state["history"]) if m["type"] == "user"),
        None,
    )
    if user_msg:
        try:
            rag    = load_rag_pipeline()
            result = rag.query(query=user_msg["content"], org=st.session_state["selected_org"])
            response = result.get("response") or str(result)
            videos   = result.get("videos") or []
            forms    = result.get("forms")  or []
        except Exception as exc:
            logger.error("Erreur génération: {}", exc)
            response = "Une erreur est survenue. Veuillez réessayer."
            videos, forms = [], []

        st.session_state["history"][-1]["content"] = response
        st.session_state["history"][-1]["videos"]  = videos
        st.session_state["history"][-1]["forms"]   = forms
        st.session_state["show_typewriter"]         = True
        st.session_state["typewriter_message"]      = response
        st.session_state["processing"]              = False
        st.rerun()
