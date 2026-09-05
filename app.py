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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --ink:#132b27; --muted:#647873; --cream:#f5f7f5; --accent:#167a68; --line:#dfe8e3; }
    .stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:rgba(245,247,245,.82); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#102f2a 0%,#18463c 100%); border-right:0; }
    [data-testid="stSidebar"] * { color:#effaf5 !important; }
    [data-testid="stSidebar"] .stCaption { color:#a6c6ba !important; }
    [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.13); }
    [data-testid="stSidebar"] button { border:1px solid rgba(255,255,255,.16); background:rgba(255,255,255,.08); }
    [data-testid="stSidebar"] button:hover { border-color:#8de1c2; background:rgba(141,225,194,.16); }
    [data-testid="stSidebar"] [data-testid="stMetric"] { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.1);
      border-radius:14px; padding:.65rem .7rem; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color:#a6c6ba !important; font-size:.68rem; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#f4fffa !important; font-size:1rem; }
    .brand { display:flex; align-items:center; gap:.65rem; margin:.15rem 0 1.5rem; }
    .brand-mark { display:grid; place-items:center; width:34px; height:34px; border-radius:11px;
      background:#8de1c2; color:#10392f !important; font-weight:700; font-size:1.1rem; }
    .brand-name { font-size:1.35rem; font-weight:700; letter-spacing:-.04em; }
    .hero { padding:2.2rem 0 1.25rem; }
    .eyebrow { color:var(--accent); font-weight:700; letter-spacing:.15em; text-transform:uppercase; font-size:.68rem; }
    h1, h2, h3 { font-family:'Playfair Display',serif !important; color:var(--ink) !important; letter-spacing:-.02em; }
    .hero h1 { font-size:3.35rem; line-height:1.04; margin:.45rem 0 .75rem; }
    .hero p { color:var(--muted); font-size:1.05rem; line-height:1.6; max-width:660px; }
    .status { display:inline-flex; align-items:center; gap:.45rem; background:#e3f6ed; color:#216b55;
      border:1px solid #c9e9db; border-radius:999px; padding:.42rem .8rem; font-size:.76rem; font-weight:700; }
    .dot { width:8px; height:8px; border-radius:50%; background:#25a875; display:inline-block; box-shadow:0 0 0 4px #c8efdf; }
    .welcome { background:linear-gradient(135deg,#e3f6ed,#f0f8ee); border:1px solid #cce6d9;
      border-radius:18px; padding:1.1rem 1.25rem; margin-bottom:1.2rem; color:#36594f; line-height:1.55; }
    .welcome strong { color:var(--ink); }
    .metric { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; box-shadow:0 8px 24px rgba(20,55,45,.04); }
    .metric-label { color:var(--muted); font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
    .metric-value { color:var(--ink); font-size:1.3rem; font-weight:700; margin-top:.25rem; }
    .disclaimer { color:#a9c7bb; font-size:.73rem; line-height:1.55; margin-top:1.4rem; }
    [data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:18px; padding:.9rem 1rem; margin:.7rem 0;
      background:#fff; box-shadow:0 5px 18px rgba(20,55,45,.035); }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { color:#173a32 !important; font-size:.96rem; line-height:1.65; }
    div[data-testid="stChatInput"] { padding:1rem 0 1.3rem; }
    div[data-testid="stChatInput"] > div { border:1px solid #bfd8cc; border-radius:18px; background:#fff; box-shadow:0 8px 25px rgba(20,55,45,.08); }
    div[data-testid="stChatInput"] textarea { color:#173a32 !important; }
    .stInfo { border-radius:14px; }
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
    st.markdown(
        '<div class="brand"><div class="brand-mark">✦</div><div class="brand-name">Mend</div></div>',
        unsafe_allow_html=True,
    )
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
