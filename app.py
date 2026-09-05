"""Presentation-ready Streamlit interface for the psychotherapy assistant."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from chatbot_demo import PsychotherapyChatbot


st.set_page_config(
    page_title="Mend | Reflective support",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600&display=swap');
    :root { --ink:#17312b; --muted:#69807a; --mint:#dff3eb; --cream:#fbfaf7; --accent:#1d806b; }
    .stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stSidebar"] { background:#17312b; }
    [data-testid="stSidebar"] * { color:#edf8f3 !important; }
    .hero { padding:1.7rem 0 1rem; }
    .eyebrow { color:var(--accent); font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.72rem; }
    h1, h2, h3 { font-family:'Playfair Display',serif !important; color:var(--ink) !important; }
    .hero h1 { font-size:3.1rem; line-height:1.05; margin:.3rem 0 .7rem; }
    .hero p { color:var(--muted); font-size:1.05rem; max-width:670px; }
    .status { display:inline-flex; align-items:center; gap:.45rem; background:#e8f6ed; color:#246b4e;
      border-radius:999px; padding:.38rem .75rem; font-size:.8rem; font-weight:600; }
    .dot { width:8px; height:8px; border-radius:50%; background:#38a169; display:inline-block; }
    .welcome { background:linear-gradient(135deg,#e8f7f0,#f3f8ed); border:1px solid #cde9dd;
      border-radius:20px; padding:1.1rem 1.25rem; margin-bottom:1rem; }
    .welcome strong { color:var(--ink); }
    .metric { background:#fff; border:1px solid #e6ebe6; border-radius:14px; padding:.9rem 1rem; }
    .metric-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--ink); font-size:1.25rem; font-weight:700; margin-top:.2rem; }
    .disclaimer { color:#b7cbc3; font-size:.76rem; line-height:1.5; margin-top:1.5rem; }
    .stChatMessage { border-radius:16px; }
    div[data-testid="stChatInput"] { padding-bottom:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_bot() -> PsychotherapyChatbot:
    dataset = Path(__file__).resolve().parent / "base_dataset.jsonl"
    return PsychotherapyChatbot(dataset_path=str(dataset) if dataset.exists() else None)


if "bot" not in st.session_state:
    st.session_state.bot = new_bot()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome. I am here to offer a calm space to reflect. "
                "What has been on your mind lately?"
            ),
        }
    ]
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {"emotion": "neutral", "behaviour": "Conversational_Act"}
if "last_response" not in st.session_state:
    st.session_state.last_response = st.session_state.messages[0]["content"]


def submit_message(text: str) -> None:
    text = text.strip()
    if not text:
        return
    response, meta = st.session_state.bot.generate_response(text)
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "crisis": meta["crisis_flag"] == "true"}
    )
    st.session_state.last_meta = meta
    st.session_state.last_response = response


with st.sidebar:
    st.markdown("## Mend")
    st.caption("Reflective support, grounded in empathy")
    st.markdown("---")
    st.markdown("### Session signals")
    meta = st.session_state.last_meta
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Affect", meta.get("emotion", "neutral").replace("_", " ").title())
    with col2:
        st.metric("Intent", meta.get("behaviour", "Conversation").replace("_", " ").title())
    st.markdown("---")
    st.markdown("### Try a demo prompt")
    demo_prompts = [
        "I have been feeling overwhelmed by work and cannot switch my mind off.",
        "I keep telling myself that I am a failure.",
        "I went for a walk today and it actually helped a little.",
    ]
    for prompt in demo_prompts:
        if st.button(prompt, use_container_width=True):
            submit_message(prompt)
            st.rerun()
    st.markdown("---")
    if st.button("Start a new session", use_container_width=True):
        st.session_state.bot = new_bot()
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome back. What feels most important to talk about today?"}
        ]
        st.session_state.last_meta = {"emotion": "neutral", "behaviour": "Conversational_Act"}
        st.session_state.last_response = st.session_state.messages[0]["content"]
        st.rerun()
    export = st.session_state.bot.export_chat_session()
    st.download_button(
        "Download session JSON",
        data=json.dumps(export, indent=2),
        file_name="mend-session.json",
        mime="application/json",
        use_container_width=True,
    )
    st.markdown(
        '<div class="disclaimer">Mend is a demonstration of an AI support tool, not a '
        'licensed therapist. It does not diagnose, prescribe, or replace professional care. '
        'If you may be in immediate danger, contact local emergency services now.</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero"><div class="eyebrow">AI-assisted reflection</div>'
    '<h1>A quieter place to<br>sort through the day.</h1>'
    '<p>Mend listens without judgement, reflects what it hears, and suggests a small next step '
    'using a CBT-inspired conversation style.</p><span class="status"><span class="dot"></span>'
    'Private demo session · local processing</span></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.7, 1], gap="large")
with left:
    st.markdown(
        '<div class="welcome"><strong>A note before we begin</strong><br>'
        'You can share as much or as little as feels comfortable. Try naming one feeling, '
        'one thought, or one thing you need today.</div>',
        unsafe_allow_html=True,
    )
    for message in st.session_state.messages:
        avatar = "🟢" if message["role"] == "assistant" else "🙂"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("crisis"):
                st.warning("If you are in immediate danger, call your local emergency number or go to the nearest emergency department.")

    prompt = st.chat_input("Share what is on your mind...")
    if prompt:
        submit_message(prompt)
        st.rerun()

with right:
    st.markdown("### How this demo works")
    st.markdown(
        "Mend combines an empathetic response layer with lightweight, transparent "
        "affect and behaviour signals. The signals are there to support the conversation, "
        "not to label you."
    )
    st.markdown('<div class="metric"><div class="metric-label">Conversation turns</div>'
                f'<div class="metric-value">{max(0, (len(st.session_state.messages) - 1) // 2)}</div></div>',
                unsafe_allow_html=True)
    st.markdown("")
    st.markdown("### Latest AI reflection")
    st.markdown(
        f'<div class="welcome"><strong>Mend says</strong><br><br>{st.session_state.last_response}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Safety first")
    st.info(
        "Crisis language activates an urgent safety response with crisis resources. "
        "For a live product, connect these flows to verified local services and clinical review."
    )
