# IndiGo RAG Bot (VS Code Ready)

Run your multi‑PDF → Chroma → Groq Q&A/Chat/Summary/Quiz app **locally** in VS Code.

## Quick Start

```bash
# 1) Create & activate a virtual env (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Add your keys
cp .env.example .env
# Edit .env and paste GROQ_API_KEY

# 4) Run
streamlit run app.py
```

Open http://localhost:8501

# ✈️ IndiGo RAG Chatbot – AI-Powered Procurement Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![LangChain](https://img.shields.io/badge/AI-LangChain-00b3b3)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> An AI-powered chatbot designed for **IndiGo Airlines’ procurement process**, capable of answering document-based queries, summarizing sourcing manuals, and generating quiz questions.  
> Built with **LangChain**, **RAG (Retrieval-Augmented Generation)**, **ChromaDB**, and **Streamlit** — integrated with **Groq’s LLaMA3** for lightning-fast inference.

---

## 🎥 Live Demo

| Feature                | Preview |
|------------------------|---------|
| **PDF Upload & Query** | ![Upload & Ask](docs/gifs/upload_and_ask.gif) |
| **Summarization**      | ![Summarization](docs/gifs/summarization.gif) |
| **Quiz Generation**    | ![Quiz Generation](docs/gifs/quiz_generation.gif) |

*(GIFs recorded at 720p for clarity — place them in `docs/gifs/` in your repo)*

---

## 🚀 Features (Detailed)

### 📄 1. Document Upload & Search  
Easily upload PDF documents (such as the *IndiGo Sourcing Training Manual*).  
The chatbot indexes the content using embeddings so you can **search instantly** for any keyword, phrase, or concept.

---

### 🔍 2. Retrieval-Augmented Generation (RAG)  
The chatbot uses **RAG** to pull relevant document sections before answering.  
This ensures that every answer is:
- Contextually accurate.
- Grounded in your document’s actual content.
- Transparent, with relevant references.

---

### 📝 3. Intelligent Summarization  
Upload a lengthy procurement guide and get:
- Bullet-point summaries for quick reading.
- Condensed key points without losing meaning.
Perfect for **briefing managers or training staff**.

---

### 🎯 4. Quiz Generation for Training  
Automatically generate quiz questions from uploaded documents.  
Ideal for:
- Staff onboarding.
- Compliance training.
- Refreshing team knowledge on procedures.

---

### ⚡ 5. Groq LLaMA3 Integration (Switchable Models)  
The chatbot uses **Groq’s LLaMA3 models** by default, offering:
- Lightning-fast inference speeds.
- Low latency for a smoother UX.
You can **switch to other available Groq models** from the sidebar dropdown for testing or performance comparison.

---

### 🌐 6. Public Sharing via Cloudflare Tunnel  
Easily run the app on **Google Colab** and share a **public link** with your team.  
The app even displays a **big, bold, clickable link** in Colab for instant access.

---

### 🛡️ 7. Role-Based and Multi-Action UI  
The chatbot supports:
- **Query Mode** – Ask document-based questions.
- **Summarize Mode** – Condense lengthy docs.
- **Quiz Mode** – Create training questions.

---

## 🛠️ Tech Stack (Detailed)

- **Python 3.10+** – Core programming language for the application.
- **[Streamlit](https://streamlit.io/)** – Interactive frontend framework to build the chatbot UI quickly and with minimal code.
- **[LangChain](https://www.langchain.com/)** – Orchestration layer for:
  - Chaining prompts.
  - Handling retrieval.
  - Managing summarization and quiz generation pipelines.
- **[ChromaDB](https://www.trychroma.com/)** – Vector database for storing and retrieving document embeddings.
- **[Sentence-Transformers](https://www.sbert.net/)** – Used for creating high-quality text embeddings.  
  *Embedder:* `"all-MiniLM-L6-v2"` (fast & lightweight, ~384-dim vector size).  
- **[Groq API](https://groq.com/)** – Provides access to LLaMA3 models for ultra-low latency inference.  
  *Model Parameters:* temperature = `0.0` (factual responses), max_tokens = `1024` (long answers).
- **[pypdf](https://pypi.org/project/pypdf/)** – Extracts text from uploaded PDFs.
- **Google Colab + Cloudflare Tunnel** – Deployment for demos without local setup.
- **[LangSmith](https://smith.langchain.com/)** – Optional integration for tracing, debugging, and tracking chatbot runs.

---

## 📂 Project Structure

