# ✈️ IndiGo RAG Chatbot – Document Q&A

A **Retrieval-Augmented Generation (RAG)** based chatbot designed for querying PDF documents with precision.  
Powered by **Groq LLaMA-3**, **LangChain**, **ChromaDB**, and **Streamlit** — built for **fast, contextual responses**.

---

## 📽 Demo Video  
**[▶ Watch the Video](https://drive.google.com/file/d/1prWTHqUt76tMGLHutOZElbROh1Peq2nI/view?usp=sharing)**  

---

## 🚀 Features & Functionalities
- 📄 **Multiple PDF Uploads** – Supports uploading and processing multiple PDFs at once.
- 🔍 **Natural Language Querying** – Ask questions as if chatting with a human.
- 🧠 **Groq LLaMA-3** – Super-fast large language model for instant answers.
- 🗄 **Vector Search via ChromaDB** – Efficient document retrieval for accurate responses.
- 📊 **Summarization** – Quickly get concise document overviews.
- ❓ **MCQ Quiz Generation** – Automatically generate quizzes from document content.
- 🎨 **Modern Streamlit UI** – Simple, interactive, and responsive interface.
- 🌐 **Two Deployment Options** – Run on **Google Colab** (cloud) or **VS Code** (local).

---

## 🛠 Tech Stack
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![LangChain](https://img.shields.io/badge/AI-LangChain-00b3b3)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA--3-orange)

---

## ⚡ How to Run & Deploy

You can run this project in **two ways**:

---

### **Option 1 – ☁ Run in Google Colab (Cloud)**
No installation required! Run directly in the browser.

1. Open the Colab Notebook:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_NOTEBOOK_LINK_HERE)
   
2. Upload your PDF files inside the notebook.

3. Enter your **GROQ_API_KEY** when prompted.

4. Click **Run All** — the Streamlit app will launch in Colab using **Cloudflare Tunnel**.

---

### **Option 2 – 🖥 Run Locally in VS Code**
Run the chatbot locally for full control over the environment.

#### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/PriyanshuInterGlobe18/Indigo-Chatbot-Assignment-1.git
cd Indigo-Chatbot-Assignment-1

````

#### **2️⃣ Create & Activate a Virtual Environment** (recommended)

```bash
python -m venv .venv
```

* **Windows:**

  ```bash
  .venv\Scripts\activate
  ```
* **macOS/Linux:**

  ```bash
  source .venv/bin/activate
  ```

#### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

#### **4️⃣ Add API Keys**

```bash
cp .env.example .env
```

* Open the `.env` file and paste your **GROQ\_API\_KEY**.

#### **5️⃣ Run the Application**

```bash
streamlit run app.py
```

---

## 📂 **Project Structure**

```
Indigo-Chatbot-Assignment-1/
│── vscode/             # VS Code project files
│── images/             # Screenshots for README
│── app.py               # Streamlit app
│── rag_core.py          # RAG logic
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation
│── .env.example         # Example environment variables
```

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

```


Do you want me to do that next?
```
