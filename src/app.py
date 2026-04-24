import html
import os
import time
import re

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from loguru import logger

from chatbot.rag_pipeline import RAGPipeline
from config.logger import setup_logger
from config.settings import LOGS_DIR


# Page config
st.set_page_config(
    page_title="Assistant RCAR/CNRA",
    page_icon="🌿",
    layout="centered",
)

if "logger_initialized" not in st.session_state:
    setup_logger(log_dir=LOGS_DIR, source="chatbot")
    st.session_state["logger_initialized"] = True


def get_welcome_message() -> str:
    return (
        "Bonjour. Je suis votre assistant RCAR/CNRA. "
        "Posez-moi une question sur les documents et informations du projet."
    )


# Initial state
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

if "history" not in st.session_state:
    st.session_state["history"] = [{"type": "bot", "content": get_welcome_message()}]

if "processing" not in st.session_state:
    st.session_state["processing"] = False

if "show_typewriter" not in st.session_state:
    st.session_state["show_typewriter"] = False

if "typewriter_message" not in st.session_state:
    st.session_state["typewriter_message"] = ""

if "rag_preload_attempted" not in st.session_state:
    st.session_state["rag_preload_attempted"] = False

if "rag_preload_error" not in st.session_state:
    st.session_state["rag_preload_error"] = ""


def toggle_theme() -> None:
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"


# Theme variables adapted for RCAR/CNRA
if st.session_state["theme"] == "dark":
    title_color = "#ffffff"
    subtitle_color = "#9dd7be"
    bg_color = "#0f1f1b"
    user_msg_bg = "#1f7a5c"
    bot_msg_bg = "#eef5f1"
    bot_msg_color = "#1a2a25"
    spinner_color = "#d8f3e8"
    info_bg = "#18312a"
    info_color = "#d8f3e8"
else:
    title_color = "#0f5132"
    subtitle_color = "#1f6e4a"
    bg_color = "#f6fbf8"
    user_msg_bg = "#0f7a5a"
    bot_msg_bg = "#e6f1ec"
    bot_msg_color = "#17332a"
    spinner_color = "#0f5132"
    info_bg = "#e5f0ea"
    info_color = "#1f5a41"


theme_css = f"""
<style>
    .stApp {{
        background-color: {bg_color};
    }}

    .stTextInput > div > div > input {{
        background-color: {"#1f2f2b" if st.session_state["theme"] == "dark" else "#ffffff"};
        color: {"#ffffff" if st.session_state["theme"] == "dark" else "#000000"};
        border: 1px solid {"#35564d" if st.session_state["theme"] == "dark" else "#b7d0c4"};
    }}

    .stTextInput > div > div > input::placeholder {{
        color: {"#9aa9a3" if st.session_state["theme"] == "dark" else "#6b7f75"} !important;
        opacity: 1;
    }}

    .stTextInput > div > div > div[data-testid="InputInstructions"] {{
        color: {"#9aa9a3" if st.session_state["theme"] == "dark" else "#6b7f75"} !important;
    }}

    .stTextInput * {{
        color: {"#ffffff" if st.session_state["theme"] == "dark" else "#000000"} !important;
    }}

    .stTextInput > label {{
        color: {"#ffffff" if st.session_state["theme"] == "dark" else "#000000"} !important;
        font-weight: 500;
    }}

    .stForm {{
        background-color: transparent;
        border: none;
        border-radius: 8px;
        padding: {"10px" if st.session_state["theme"] == "light" else "0px"};
    }}

    .stFormSubmitButton > button {{
        background-color: {"#1f7a5c" if st.session_state["theme"] == "dark" else "#0f5132"};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }}

    .stFormSubmitButton > button:hover {{
        background-color: {"#176147" if st.session_state["theme"] == "dark" else "#0a3f27"};
    }}

    .stButton > button {{
        background-color: {"#325248" if st.session_state["theme"] == "dark" else "#5f7a70"};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }}

    .stButton > button:hover {{
        background-color: {"#40695d" if st.session_state["theme"] == "dark" else "#4e6860"};
    }}

    .css-1d391kg {{
        background-color: {"#172a24" if st.session_state["theme"] == "dark" else "#edf5f1"};
    }}

    .stMarkdown p {{
        color: {"#ffffff" if st.session_state["theme"] == "dark" else "#000000"};
    }}

    .stSpinner {{
        display: none !important;
    }}

    .custom-spinner {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }}

    .custom-spinner .spinner-ring {{
        width: 34px;
        height: 34px;
        border: 4px solid {"#35564d" if st.session_state["theme"] == "dark" else "#b7d0c4"};
        border-top: 4px solid {"#d8f3e8" if st.session_state["theme"] == "dark" else "#0f5132"};
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}

    @keyframes spin {{
        100% {{ transform: rotate(360deg); }}
    }}

    .user-message {{
        background-color: {user_msg_bg};
        color: white;
        padding: 14px 18px;
        font-size: 1.05rem;
        border-radius: 16px 16px 0 16px;
        margin-bottom: 16px;
        width: fit-content;
        min-width: 30%;
        max-width: 70%;
        word-break: break-word;
        white-space: pre-line;
        margin-left: auto;
        text-align: right;
        display: block;
    }}

    .bot-message {{
        background-color: {bot_msg_bg};
        color: {bot_msg_color};
        padding: 10px 12px;
        border-radius: 12px 12px 12px 0;
        margin-bottom: 8px;
        max-width: 70%;
        align-self: flex-start;
        word-break: break-word;
        white-space: pre-line;
    }}

    .bot-message * {{
        color: {bot_msg_color} !important;
    }}

    .message-separator {{
        margin: 15px 0;
    }}

    .form-container {{
        margin-top: 30px;
        padding-top: 20px;
        background-color: {bg_color};
    }}

    .typewriter-message {{
        background-color: {bot_msg_bg};
        color: {bot_msg_color};
        padding: 10px 12px;
        border-radius: 12px 12px 12px 0;
        margin-bottom: 8px;
        max-width: 70%;
        align-self: flex-start;
        word-break: break-word;
        white-space: pre-line;
        border-right: 2px solid {bot_msg_color};
        animation: blink-caret 0.75s step-end infinite;
    }}

    .typewriter-message * {{
        color: {bot_msg_color} !important;
    }}

    @keyframes blink-caret {{
        from, to {{ border-color: transparent; }}
        50% {{ border-color: {bot_msg_color}; }}
    }}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Parametres")
    theme_label = "Mode sombre" if st.session_state["theme"] == "dark" else "Mode clair"
    if st.button(theme_label, key="theme_toggle", help="Basculer entre le mode sombre et clair"):
        toggle_theme()
        st.rerun()

    st.markdown("---")
    st.markdown("### Reinitialisation")
    if st.button("Effacer l'historique", help="Reinitialise la conversation", type="secondary"):
        st.session_state["history"] = [{"type": "bot", "content": get_welcome_message()}]
        st.session_state["show_typewriter"] = False
        st.session_state["typewriter_message"] = ""
        st.session_state["processing"] = False
        st.success("Historique efface")
        time.sleep(0.7)
        st.rerun()


st.markdown(
    f"<h1 style='text-align: left; color: {title_color}; margin-bottom: 0.2rem;'>Assistant RCAR/CNRA</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:left;font-size:1rem;color:{subtitle_color};margin-bottom:1.2rem;'>"
    "Interface Chatbot pour les sites RCAR et CNRA"
    "</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag_pipeline() -> RAGPipeline:
    with st.spinner("Initialisation de l'assistant RCAR/CNRA..."):
        return RAGPipeline()


def preload_rag_pipeline() -> None:
    """Initialize embeddings/vectorstore at startup to reduce first-question latency."""
    if st.session_state["rag_preload_attempted"]:
        return

    st.session_state["rag_preload_attempted"] = True
    try:
        load_rag_pipeline()
        st.session_state["rag_preload_error"] = ""
        logger.info("Pipeline RAG precharge au demarrage de l'application.")
    except Exception as exc:
        st.session_state["rag_preload_error"] = str(exc)
        logger.error("Echec du prechargement RAG au demarrage: {}", exc)


def render_plain_text(text: str) -> str:
    """Convertit le texte en HTML lisible, en nettoyant le Markdown résiduel."""
    if not text:
        return ""
    
    # Supprimer le Markdown résiduel que le LLM aurait quand même généré
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)   # **gras** → texte
    text = re.sub(r'\*(.*?)\*', r'\1', text)         # *italique* → texte
    text = re.sub(r'#{1,6}\s*', '', text)            # ### Titre → Titre
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text, flags=re.DOTALL)  # `code` → texte
    text = re.sub(r'---+', '─' * 30, text)           # --- → ligne lisible
    
    # Échapper le HTML puis remettre les sauts de ligne
    text = html.escape(text)
    text = text.replace("\n", "<br>")
    return text


def stream_typewriter(text: str, placeholder, delay: float = 0.012) -> None:
    """Render the latest bot response progressively."""
    if not text:
        placeholder.markdown("<div class='bot-message'></div>", unsafe_allow_html=True)
        return

    displayed_text = ""
    for char in text:
        displayed_text += char
        safe_partial = render_plain_text(displayed_text)
        placeholder.markdown(
            f"<div class='typewriter-message'>{safe_partial}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(delay)

    safe_text = render_plain_text(text)
    placeholder.markdown(
        f"<div class='bot-message'>{safe_text}</div>",
        unsafe_allow_html=True,
    )


preload_rag_pipeline()


chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state["history"]):
        message_container = st.container()

        with message_container:
            if message["type"] == "user":
                user_content = render_plain_text(message["content"])
                st.markdown(
                    f"<div class='user-message'>{user_content}</div>",
                    unsafe_allow_html=True,
                )
            elif message["type"] == "bot":
                is_last_message = i == len(st.session_state["history"]) - 1
                should_animate = (
                    st.session_state["show_typewriter"]
                    and is_last_message
                    and message["content"] == st.session_state["typewriter_message"]
                    and bool(message["content"])
                )

                if should_animate:
                    response_placeholder = st.empty()
                    stream_typewriter(message["content"], response_placeholder)
                    st.session_state["show_typewriter"] = False
                    st.session_state["typewriter_message"] = ""
                else:
                    is_typing_message = (
                        is_last_message
                        and st.session_state["processing"]
                        and message["content"] == ""
                    )
                    bot_class = "typewriter-message" if is_typing_message else "bot-message"
                    if message["content"]:
                        bot_content = render_plain_text(message["content"])
                    else:
                        bot_content = "..."

                    st.markdown(
                        f"<div class='{bot_class}'>{bot_content}</div>",
                        unsafe_allow_html=True,
                    )

                if i < len(st.session_state["history"]) - 1:
                    st.markdown("<div class='message-separator'></div>", unsafe_allow_html=True)


st.markdown('<div class="form-container">', unsafe_allow_html=True)

if st.session_state["processing"]:
    spinner_col1, spinner_col2, spinner_col3 = st.columns([1, 2, 1])
    with spinner_col2:
        spinner_html = f"""
        <div class="custom-spinner" style="text-align: center; margin: 20px 0;">
            <div class="spinner-ring"></div>
            <div style="color:{spinner_color};font-weight:bold;font-size:1.05rem;margin-top:10px;">
                Generation de la reponse...
            </div>
        </div>
        """
        st.markdown(spinner_html, unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    st.markdown(
        f"<label style='color: {title_color}; font-weight: 500; margin-bottom: 8px; display: block;'>Votre question :</label>",
        unsafe_allow_html=True,
    )

    user_input = st.text_input(
        "Votre question",
        key="input_field",
        label_visibility="collapsed",
        placeholder="Tapez votre question ici...",
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        submit = st.form_submit_button(
            "Envoyer",
            help="Envoyer votre question",
            disabled=st.session_state["processing"],
        )
    with col2:
        st.write("")

st.markdown("</div>", unsafe_allow_html=True)


if submit and user_input and not st.session_state["processing"]:
    st.session_state["processing"] = True
    st.session_state["show_typewriter"] = False
    st.session_state["typewriter_message"] = ""

    user_message = {
        "type": "user",
        "content": user_input,
        "timestamp": time.time(),
    }

    bot_message = {
        "type": "bot",
        "content": "",
        "timestamp": time.time(),
    }

    st.session_state["history"].extend([user_message, bot_message])
    st.rerun()


if (
    st.session_state["history"]
    and st.session_state["history"][-1]["type"] == "bot"
    and st.session_state["history"][-1]["content"] == ""
    and st.session_state["processing"]
):
    user_message = None
    for msg in reversed(st.session_state["history"]):
        if msg["type"] == "user":
            user_message = msg
            break

    if user_message:
        last_question = user_message["content"]

        try:
            rag = load_rag_pipeline()
            result = rag.query(query=last_question)
            logger.info(result) 
            if isinstance(result, dict):
                response = result.get("response") or result.get("answer") or str(result)
            else:
                response = str(result)

            logger.info("Reponse UI recue | chars={}", len(response or ""))
        except Exception as e:
            logger.error("Erreur lors de la generation de la reponse : {}", e)
            response = "Une erreur est survenue lors de la generation de la reponse. Veuillez reessayer."

        st.session_state["history"][-1]["content"] = response
        st.session_state["show_typewriter"] = True
        st.session_state["typewriter_message"] = response
        st.session_state["processing"] = False
        st.rerun()