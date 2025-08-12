# IndiGo RAG Bot (VS Code Ready)

Run your multi‑PDF → Chroma → Groq Q&A/Chat/Summary/Quiz app **locally** in VS Code.

## Quick Start


---

````markdown
# ✈️ IndiGo RAG Chatbot – Document Q&A

A **Retrieval-Augmented Generation (RAG)** based chatbot designed for querying PDF documents with precision.  
Powered by **Groq LLaMA-3**, **LangChain**, **ChromaDB**, and **Streamlit** — built for fast, contextual responses.

---

## 📽 Demo Video  
**[▶ Watch the Video](https://drive.google.com/file/d/1prWTHqUt76tMGLHutOZElbROh1Peq2nI/view?usp=sharing)**  

---

## 🚀 Features
- 📄 Upload multiple PDF documents
- 🔍 Query using natural language
- 🧠 Uses **Groq LLaMA-3** for lightning-fast responses
- 🗄 Document embeddings stored in **ChromaDB**
- 📊 Summarization & MCQ quiz generation
- 🎨 Modern **Streamlit** UI

---

## ⚡ Run the Chatbot

You can run this project in **two ways**:

---

### **Option 1 – Run in Google Colab (Cloud)**
No installation required! Run directly in the browser.

1. Open the Colab Notebook:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_NOTEBOOK_LINK_HERE)
   
2. Upload your PDF files inside the notebook.

3. Enter your **GROQ_API_KEY** when prompted.

4. Click **Run All** — the Streamlit app will launch in Colab.

---

### **Option 2 – Run Locally in VS Code**
Run directly on your system for full control.

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PriyanshuInterGlobe18/Indigo-Chatbot-Assignment-1.git
cd Indigo-Chatbot-Assignment-1
````

#### 2️⃣ Create & Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv .venv
```

**For Windows:**

```bash
.venv\Scripts\activate
```

**For macOS/Linux:**

```bash
source .venv/bin/activate
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Add Your API Key

```bash
cp .env.example .env
```

Edit `.env` and add:

```
GROQ_API_KEY=your_api_key_here
```

#### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Indigo-Chatbot-Assignment-1/
│── vscode/               # VS Code project files
│── images/               # Screenshots for README
│── app.py                 # Streamlit app
│── rag_core.py            # RAG logic
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── .env.example           # Example environment variables
```

---

---


```


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

