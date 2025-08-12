import os, json, tempfile
import streamlit as st

from rag_core import (
    load_pdfs_as_docs, chunk_texts, build_or_update_chroma, load_chroma,
    rag_answer, rag_answer_exhaustive, summarize, quiz,
    init_logger, log_interaction
)

# ---------- Config ----------
BOT_NAME = "IndiGo RAG Bot"
HERO_IMG = "https://img.freepik.com/free-vector/chatbot-chat-message-vectorart_78370-4104.jpg?semt=ais_hybrid&w=740&q=80"

# ---------- Page setup ----------
st.set_page_config(page_title=f"{BOT_NAME} (Chroma + Groq)", page_icon="✈️", layout="wide")

# ---------- Light Blue / Deep Blue with Yellow Accent Theme ----------
PALETTE = {
    "bg1": "#e6f2ff", "bg2": "#b3d9ff", "card": "#ffffff", "text": "#0d1b2a",
    "muted": "#4f5d75", "brand": "#0077b6", "brand2": "#00b4d8", "accent": "#ffcc00",
    "success": "#10b981", "danger": "#ef4444",
}
THEME_CSS = f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  background: linear-gradient(120deg, {PALETTE['bg1']}, {PALETTE['bg2']});
  color: {PALETTE['text']};
}}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
.block-container {{
  padding-top: 1.2rem; padding-bottom: 2rem;
  background: {PALETTE['card']};
  border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,.15);
  border: 1px solid rgba(0,0,0,.08);
}}

[data-testid="stSidebar"] {{ background: {PALETTE['brand']}; }}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{ color: #fff !important; }}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput input, [data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stFileUploader, [data-testid="stSidebar"] .stSlider {{
  background: #ffffff !important; color: {PALETTE['text']} !important;
  border: 1px solid {PALETTE['brand2']} !important; border-radius: 10px !important;
}}
[data-testid="stSidebar"] [role="listbox"] *, [data-testid="stSidebar"] [data-baseweb="select"] * {{
  color: {PALETTE['text']} !important;
}}
[data-testid="stSidebar"] svg {{ color: {PALETTE['text']} !important; fill: {PALETTE['text']} !important; }}

div[data-testid="stSlider"] label, div[data-testid="stSlider"] span,
div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {{ color: {PALETTE['text']} !important; }}
div[data-baseweb="slider"] [role="slider"]{{ background:#f59e0b !important; border:2px solid {PALETTE['text']} !important; }}
div[data-baseweb="slider"] .rc-slider-track, div[data-baseweb="slider"] .rc-slider-rail {{ background:#0ea5e9 !important; }}

[data-testid="stSidebar"] .stButton > button {{
  background: linear-gradient(90deg, #ef4444, #f97316);
  color: white; border: 0; padding: .55rem 1rem; border-radius: 10px; font-weight: 600;
  box-shadow: 0 6px 18px rgba(0,0,0,.15);
}}
[data-testid="stSidebar"] .stButton > button:hover {{ filter: brightness(1.05); }}

h1, h2, h3, h4 {{ color: {PALETTE['text']}; }}
input, textarea, .stTextInput > div > div > input {{
  background: {PALETTE['bg1']} !important; color: {PALETTE['text']} !important;
  border: 1px solid {PALETTE['brand']} !important; border-radius: 10px !important;
}}
.stSelectbox, .stNumberInput, .stFileUploader, .stSlider, .stRadio, .stCheckbox label {{ color: {PALETTE['text']} !important; }}

.stButton > button {{
  background: linear-gradient(90deg, {PALETTE['brand']}, {PALETTE['brand2']});
  color: white; border: 0; padding: .55rem 1rem; border-radius: 10px; font-weight: 600;
  box-shadow: 0 6px 18px rgba(0,0,0,.15);
}}
.stButton > button:hover {{ filter: brightness(1.05); }}
[data-testid="baseButton-secondary"] {{ background: {PALETTE['accent']} !important; color: {PALETTE['text']} !important; border: none !important; }}

.streamlit-expanderHeader {{ color: {PALETTE['text']} !important; }}
code, pre {{ background: {PALETTE['bg1']} !important; color: {PALETTE['text']} !important; }}

.chat-bubble-user {{ background: rgba(0,183,255,.12); border: 1px solid {PALETTE['brand2']};
  padding: 12px 14px; border-radius: 12px; margin: 8px 0; color: {PALETTE['text']}; }}
.chat-bubble-assistant {{ background: rgba(255,204,0,.10); border: 1px solid {PALETTE['accent']};
  padding: 12px 14px; border-radius: 12px; margin: 8px 0; color: {PALETTE['text']}; }}

.hero-img {{
  width: 92px; height: 92px; border-radius: 18px; object-fit: cover;
  border: 2px solid {PALETTE['brand2']}; box-shadow: 0 10px 30px rgba(0,0,0,.15);
}}
.hero-title {{ font-size: 26px; font-weight: 800; letter-spacing:.2px; }}
.hero-sub {{ color: {PALETTE['muted']}; font-size: 13px; }}
.hr-soft {{ border-top: 1px solid rgba(0,0,0,.08); margin: .5rem 0 1rem; }}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------- Hero (top of app) ----------
st.markdown(
    f"""
    <div style="text-align: center; padding: 12px 6px 4px;">
        <img src="{HERO_IMG}" alt="Bot Image" style="width: 200px; border-radius: 16px; margin-bottom: 10px; box-shadow: 0 10px 30px rgba(0,0,0,.20);" />
        <h1 style="margin-bottom: 0;">✈️ {BOT_NAME}</h1>
        <p style="color: #274c77; font-size: 15px;">Multi-PDF → Chroma → Q&A / Chat / Summary / Quiz (Groq)</p>
    </div>
    <hr class="hr-soft">
    """,
    unsafe_allow_html=True
)

# ---------- Session defaults ----------
if "vs_ready" not in st.session_state: st.session_state.vs_ready = False
if "messages" not in st.session_state: st.session_state.messages = []
if "nav" not in st.session_state: st.session_state.nav = "📥 Build Index"
if "persist_dir" not in st.session_state: st.session_state.persist_dir = "chroma_db"
if "collection" not in st.session_state: st.session_state.collection = "indigo_docs"
if "top_k" not in st.session_state: st.session_state.top_k = 12
if "temperature" not in st.session_state: st.session_state.temperature = 0.1
if "max_tokens" not in st.session_state: st.session_state.max_tokens = 2048
if "username" not in st.session_state: st.session_state.username = "user"

# ---------- Logger (SQLite) ----------
init_logger("rag_logs.db")

# ---------- Sidebar (settings form) ----------
with st.sidebar:
    st.header("Settings")
    with st.form("settings_form"):
        groq_key_in = st.text_input("GROQ API Key (session only)", type="password", value=os.environ.get("GROQ_API_KEY",""))
        model_choice = st.selectbox("Groq model", ["llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b","llama-3.1-70b-versatile","llama-3.1-8b-instant"], index=0)

        persist_dir = st.text_input("Chroma persist dir", value=st.session_state.persist_dir)
        collection  = st.text_input("Collection name", value=st.session_state.collection)
        top_k       = st.slider("Top-K retrieved chunks", 2, 60, st.session_state.top_k)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, st.session_state.temperature)
        st.caption(f"Temp value: **{st.session_state.get('temperature', temperature):.2f}**")
        max_tokens  = st.slider("Max tokens", 256, 4096, st.session_state.max_tokens, step=64)
        st.caption(f"Max tokens value: **{st.session_state.get('max_tokens', max_tokens)}**")

        nav = st.radio(
            "Go to",
            ["📥 Build Index", "❓ Q&A", "💬 Document Chat", "📝 Summarize", "🧩 Quiz"],
            index=["📥 Build Index","❓ Q&A","💬 Document Chat","📝 Summarize","🧩 Quiz"].index(st.session_state.nav)
        )

        submitted = st.form_submit_button("Apply")
        if submitted:
            if groq_key_in:
                os.environ["GROQ_API_KEY"] = groq_key_in.strip()
            os.environ["GROQ_MODEL_OVERRIDE"] = model_choice

            st.session_state.persist_dir = persist_dir
            st.session_state.collection  = collection
            st.session_state.top_k       = top_k
            st.session_state.temperature = temperature
            st.session_state.max_tokens  = max_tokens
            st.session_state.nav         = nav

    st.caption("Key status: " + ("✅ set" if os.environ.get("GROQ_API_KEY") else "❌ missing"))
    st.caption("Embeddings: " + os.environ.get("EMBED_MODEL", "thenlper/gte-large") + " (GPU if available)")

# ---------- Read active settings ----------
persist_dir = st.session_state.persist_dir
collection  = st.session_state.collection
top_k       = st.session_state.top_k
temperature = st.session_state.temperature
max_tokens  = st.session_state.max_tokens
active_page = st.session_state.nav

def need_vs():
    if not st.session_state.vs_ready:
        st.warning("No vector DB loaded yet. Build or Load an index first.")
        st.stop()

# ---------- Pages ----------
if active_page == "📥 Build Index":
    st.subheader("Upload PDFs → Build/Update Chroma Index")
    files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    reset_index = st.checkbox("Reset index (rebuild from scratch)", value=False, help="Deletes existing Chroma data in the chosen persist dir before indexing.")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Build / Update Index", use_container_width=True):
            if not files:
                st.error("Please upload at least one PDF")
            else:
                with st.spinner("Indexing..."):
                    paths=[]
                    for f in files:
                        p = os.path.join(tempfile.gettempdir(), f.name)
                        with open(p,"wb") as w: w.write(f.getbuffer())
                        paths.append(p)
                    docs = load_pdfs_as_docs(paths)
                    chunks = chunk_texts(docs, chunk_size=1800, chunk_overlap=150)
                    _ = build_or_update_chroma(chunks, persist_dir=persist_dir, collection_name=collection, reset=reset_index)
                    st.session_state.vs_ready = True
                st.success(f"Indexed {len(chunks)} chunks → {persist_dir}")

    with colB:
        if st.button("Load Existing Index", use_container_width=True):
            try:
                _ = load_chroma(persist_dir=persist_dir, collection_name=collection)
                st.session_state.vs_ready = True
                st.success("Loaded existing Chroma index.")
            except Exception as e:
                st.error(f"Failed to load: {e}")

elif active_page == "❓ Q&A":
    need_vs()
    st.subheader("Ask a question (grounded in your PDFs)")
    q = st.text_input("Your question", placeholder="e.g., Full procedure to publish a contract workspace")
    detailed = st.checkbox("Detailed (multi-pass map-reduce)", value=True)
    batch_size = st.slider("Batch size (Detailed mode)", 3, 12, 6)

    if st.button("Answer", use_container_width=True):
        with st.spinner("Thinking..."):
            vs = load_chroma(persist_dir=persist_dir, collection_name=collection)
            try:
                if detailed:
                    out = rag_answer_exhaustive(
                        vs, q, k=max(24, top_k*2), batch=batch_size,
                        temperature=0.0, max_tokens=min(1200, max_tokens)
                    )
                    mode = "qna_exhaustive"
                else:
                    out = rag_answer(vs, q, k=top_k, temperature=temperature, max_tokens=max_tokens)
                    mode = "qna"

                st.markdown("### Answer")
                st.write(out["answer"])
                with st.expander("Citations"):
                    for c in out["citations"]:
                        st.write(f"[{c['slot']}] {c['source']} (p.{c['page']})")

                log_interaction(
                    session_id="ui", user=st.session_state.username,
                    question=q, mode=mode,
                    model=os.environ.get("GROQ_MODEL_OVERRIDE","llama-3.3-70b-versatile"),
                    top_k=top_k, max_tokens=max_tokens,
                    temperature=(temperature if mode=="qna" else 0.0),
                    answer=out["answer"], citations=out["citations"], error=None
                )
            except Exception as e:
                st.error(str(e))
                log_interaction(
                    session_id="ui", user=st.session_state.username,
                    question=q, mode="qna" if not detailed else "qna_exhaustive",
                    model=os.environ.get("GROQ_MODEL_OVERRIDE","llama-3.3-70b-versatile"),
                    top_k=top_k, max_tokens=max_tokens,
                    temperature=(temperature if not detailed else 0.0),
                    answer="", citations=[], error=str(e)
                )

elif active_page == "💬 Document Chat":
    need_vs()
    st.subheader("Document Chat")

    user_msg = st.text_input("Message", placeholder="Ask a follow-up question…")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        send = st.button("Send", use_container_width=True)
    with col2:
        if st.button("Clear Chat (session)"):
            st.session_state.messages = []
            st.success("Cleared in-memory chat history.")
    with col3:
        chat_history_txt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Download Chat (TXT)", chat_history_txt, file_name="chat_history.txt")

    if send and user_msg.strip():
        st.session_state.messages.append({"role":"user","content":user_msg})
        convo = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
        q2 = f"Conversation so far:\n{convo}\n\nUser's latest question: {user_msg}"
        with st.spinner("Thinking..."):
            vs = load_chroma(persist_dir=persist_dir, collection_name=collection)
            try:
                out = rag_answer(vs, q2, k=top_k, temperature=temperature, max_tokens=max_tokens)
                st.session_state.messages.append({"role":"assistant","content":out["answer"]})
                log_interaction(
                    session_id="ui", user=st.session_state.username,
                    question=user_msg, mode="chat",
                    model=os.environ.get("GROQ_MODEL_OVERRIDE","llama-3.3-70b-versatile"),
                    top_k=top_k, max_tokens=max_tokens, temperature=temperature,
                    answer=out["answer"], citations=out["citations"], error=None
                )
            except Exception as e:
                st.error(str(e))
                log_interaction(
                    session_id="ui", user=st.session_state.username,
                    question=user_msg, mode="chat",
                    model=os.environ.get("GROQ_MODEL_OVERRIDE","llama-3.3-70b-versatile"),
                    top_k=top_k, max_tokens=max_tokens, temperature=temperature,
                    answer="", citations=[], error=str(e)
                )

    st.markdown("#### Recent Chat")
    for m in st.session_state.messages[-12:]:
        if m["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {m['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'><b>Assistant:</b> {m['content']}</div>", unsafe_allow_html=True)

elif active_page == "📝 Summarize":
    need_vs()
    st.subheader("Summarize (chapter/page-level brief)")
    topic = st.text_input("(Optional) Topic/file hint", value="overview")
    if st.button("Summarize", use_container_width=True):
        with st.spinner("Summarizing..."):
            vs = load_chroma(persist_dir=persist_dir, collection_name=collection)
            s = summarize(vs, topic_hint=topic, k=20, max_tokens=max_tokens)
            st.markdown("### Summary")
            st.write(s)
            log_interaction(
                session_id="ui", user=st.session_state.username,
                question=f"[SUMMARY] {topic}", mode="summary",
                model=os.environ.get("GROQ_MODEL_OVERRIDE","llama-3.3-70b-versatile"),
                top_k=20, max_tokens=max_tokens, temperature=0.2,
                answer=s, citations=[], error=None
            )

elif active_page == "🧩 Quiz":
    need_vs()
    st.subheader("Generate MCQ quiz")
    topic = st.text_input("(Optional) Topic hint", placeholder="e.g., PR→PO flow")
    num = st.slider("Number of questions", 3, 15, 5)
    if st.button("Create Quiz", use_container_width=True):
        with st.spinner("Generating..."):
            vs = load_chroma(persist_dir=persist_dir, collection_name=collection)
            items = quiz(vs, topic_hint=topic, num=num)
            if isinstance(items, list) and items and isinstance(items[0], dict) and "question" in items[0]:
                for i, qx in enumerate(items, start=1):
                    st.markdown(f"**Q{i}. {qx['question']}**")
                    for opt in qx["options"]:
                        st.write(f"- {opt}")
                    st.caption(f"**Answer:** {qx['answer']} — {qx.get('why','')}")
            else:
                st.write("Model returned non-JSON output:")
                st.code(items, language="json")
            log_interaction(
                session_id="ui", user=st.session_state.username,
                question=f"[QUIZ] {topic} (n={num})", mode="quiz",
                model=os.environ.get("GROQ_MODEL_OVERRIDE","llama-3.3-70b-versatile"),
                top_k=top_k, max_tokens=max_tokens, temperature=0.2,
                answer=json.dumps(items, ensure_ascii=False), citations=[], error=None
            )
