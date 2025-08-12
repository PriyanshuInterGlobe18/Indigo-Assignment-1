import os, json, requests, sqlite3, time, random, hashlib, shutil, re
from typing import List

# ───────────────────── LangSmith tracking (kept as you wanted) ─────────────────────
# NOTE: If you push to a public repo, rotate this key later.
os.environ["LANGCHAIN_API_KEY"]    = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "Indigo-RAG-Streamlit"

from langsmith import traceable  # noqa: F401

# Runtime safety to avoid GPU meta-tensor issues in some envs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ───────────────────── Config ─────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "thenlper/gte-large")
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_OVERRIDE", "llama-3.3-70b-versatile")
DEFAULT_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192"
]
_env_candidates = [m.strip() for m in os.getenv("GROQ_MODEL_CANDIDATES", "").split(",") if m.strip()]
GROQ_MODEL_CANDIDATES = [GROQ_MODEL_PRIMARY] + [m for m in (_env_candidates or DEFAULT_CANDIDATES) if m != GROQ_MODEL_PRIMARY]
GROQ_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# ───────────────────── Embeddings / Chroma ─────────────────────
def _pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def get_embeddings():
    device = _pick_device()
    model_name = os.getenv("EMBED_MODEL", EMBED_MODEL)
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": 64}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

def chunk_texts(docs, chunk_size=1800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def load_pdfs_as_docs(file_paths: List[str]):
    from langchain_core.documents import Document
    out = []
    for path in file_paths:
        r = PdfReader(path)
        for i, page in enumerate(r.pages, start=1):
            t = (page.extract_text() or "").strip()
            if t:
                out.append(Document(page_content=" ".join(t.split()), metadata={"source": os.path.basename(path), "page": i}))
    return out

def _safe_rmtree(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def build_or_update_chroma(chunks, persist_dir="chroma_db", collection_name="docs", reset: bool = False):
    emb = get_embeddings()
    os.makedirs(persist_dir, exist_ok=True)
    if reset:
        for fn in os.listdir(persist_dir):
            _safe_rmtree(os.path.join(persist_dir, fn))
    vs = Chroma.from_documents(chunks, emb, persist_directory=persist_dir, collection_name=collection_name)
    vs.persist()
    return vs

def load_chroma(persist_dir="chroma_db", collection_name="docs"):
    emb = get_embeddings()
    return Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=emb)

# ───────────────────── Groq HTTP client ─────────────────────
def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(12, 2 ** attempt) + random.random())

def _groq_post(payload: dict, api_key: str, base_url: str):
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    return requests.post(url, headers=headers, json=payload, timeout=60)

def groq_complete(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    base_url = os.getenv("GROQ_BASE_URL", GROQ_BASE)
    primary = os.getenv("GROQ_MODEL_OVERRIDE", GROQ_MODEL_PRIMARY)
    candidates = [primary] + [m for m in GROQ_MODEL_CANDIDATES if m != primary]
    last_err = None

    for model in candidates:
        payload = {
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": "You are a concise, grounded assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        for attempt in range(6):
            try:
                r = _groq_post(payload, api_key, base_url)
                if r.status_code == 429:
                    try:
                        j = r.json()
                        emsg = (j.get("error") or {}).get("message", "")
                        ecode = (j.get("error") or {}).get("code", "")
                    except Exception:
                        emsg, ecode = r.text, ""
                    if "tokens per day" in emsg.lower() or ecode == "rate_limit_exceeded":
                        last_err = f"TPD exhausted for `{model}`: {emsg}"
                        break
                    _sleep_backoff(attempt); last_err = emsg; continue
                if r.status_code in (500, 502, 503, 504):
                    _sleep_backoff(attempt); last_err = f"HTTP {r.status_code}"; continue

                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except requests.RequestException as e:
                s = str(e)
                if any(code in s for code in ["429", "500", "502", "503", "504"]):
                    _sleep_backoff(attempt); last_err = s; continue
                if "Unknown request URL" in s or "unknown_url" in s:
                    raise RuntimeError(f"Groq base URL looks wrong: {base_url}. Expected 'https://api.groq.com/openai/v1'. Original error: " + s)
                raise
        continue
    raise RuntimeError(f"All Groq models exhausted/limited. Last error: {last_err}")

# ───────────────────── Helpers ─────────────────────
def _format_citations(docs) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source","unknown"); pg = d.metadata.get("page","-")
        parts.append(f"[S{i}] ({src} p.{pg})\n{d.page_content}")
    return "\n\n".join(parts)

def _coerce_json_array(raw: str):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    start = s.find("["); end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = s[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    s2 = s.replace("'", '"')
    start = s2.find("["); end = s2.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = s2[start:end+1]
        return json.loads(snippet)
    raise ValueError("Could not coerce JSON array from model output.")

# ───────────────────── RAG tasks ─────────────────────
def rag_answer(vs, question: str, k: int = 5, temperature: float = 0.2, max_tokens: int = 512):
    docs = vs.similarity_search(question, k=k)
    ctx = _format_citations(docs)
    prompt = f"""Answer strictly from the context. If unsure, say you don't know.
Be EXHAUSTIVE: include every step, rule, exception, and prerequisite you find.
Cite sources inline like [S1], [S2] (they map to (source,page)). Use numbered lists.

Question: {question}

Context:
{ctx}

Answer (exhaustive with citations):"""
    ans = groq_complete(prompt, temperature=temperature, max_tokens=max_tokens)
    cites = [{"slot": f"S{i+1}", "source": d.metadata.get("source"), "page": d.metadata.get("page")} for i,d in enumerate(docs)]
    return {"answer": ans, "citations": cites}

def rag_answer_exhaustive(vs, question: str, k: int = 40, batch: int = 8,
                          temperature: float = 0.0, max_tokens: int = 900):
    docs = vs.similarity_search(question, k=k)
    partials = []
    for i in range(0, len(docs), batch):
        group = docs[i:i+batch]
        ctx = _format_citations(group)
        prompt = f"""You will answer a complex question using ONLY this batch of context.
List EVERY relevant step/detail you can find. Use [S#] citations for each bullet.

Question: {question}

Context:
{ctx}

Partial answer (bulleted with citations):"""
        partials.append(groq_complete(prompt, temperature=temperature, max_tokens=max_tokens))
    merged_ctx = "\n\n".join([f"[P{i+1}] {p}" for i, p in enumerate(partials)])
    final_prompt = f"""You are merging partial answers into ONE exhaustive answer.
Remove duplicates, keep ordering logical, and preserve [S#] citations from partials.
If something conflicts, note both and cite them.

Question: {question}

Partials:
{merged_ctx}

Final exhaustive answer (numbered list with [S#] citations):"""
    final = groq_complete(final_prompt, temperature=0.0, max_tokens=1200)
    cites = [{"slot": f"S{i+1}", "source": d.metadata.get("source"), "page": d.metadata.get("page")}
             for i, d in enumerate(docs[:min(15, len(docs))])]
    return {"answer": final, "citations": cites}

def summarize(vs, topic_hint: str = "overview", k: int = 20, max_tokens: int = 700):
    docs = vs.similarity_search(topic_hint, k=k)
    ctx = _format_citations(docs)
    prompt = f"""Summarize the following content clearly and concisely as bullet points with key steps/terms.

Context:
{ctx}

Summary:"""
    return groq_complete(prompt, temperature=0.2, max_tokens=max_tokens)

def quiz(vs, topic_hint: str = "", num: int = 5, k: int = 15, max_tokens: int = 900):
    q = topic_hint if topic_hint.strip() else "important concepts and procedures"
    docs = vs.similarity_search(q, k=k)
    ctx = _format_citations(docs)

    prompt = f"""Generate {num} MCQs from the context. 
Rules:
- USE ONLY the context.
- Output MUST be a JSON array (no markdown, no prose), exactly like:
[
  {{"question":"...","options":["A","B","C","D"],"answer":"A","why":"1-line rationale"}},
  ...
]
- "options" must have 4 strings. "answer" must be exactly one of the options.

Context:
{ctx}
"""
    raw = groq_complete(prompt, temperature=0.0, max_tokens=max_tokens)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    try:
        data = _coerce_json_array(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return [{"raw": raw}]

# ───────────────────── SQLite logger ─────────────────────
def init_logger(db_path: str = "rag_logs.db"):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER,
        session_id TEXT,
        user TEXT,
        question TEXT,
        mode TEXT,
        model TEXT,
        top_k INTEGER,
        max_tokens INTEGER,
        temperature REAL,
        answer TEXT,
        citations TEXT,
        error TEXT
    )""")
    con.commit(); con.close()

def log_interaction(session_id: str, user: str, question: str, mode: str,
                    model: str, top_k: int, max_tokens: int, temperature: float,
                    answer: str, citations: list, error: str | None = None,
                    db_path: str = "rag_logs.db"):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO interactions
        (ts, session_id, user, question, mode, model, top_k, max_tokens, temperature, answer, citations, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(time.time()), session_id, user, question, mode, model, top_k, max_tokens, float(temperature),
        answer, json.dumps(citations or []), error
    ))
    con.commit(); con.close()
