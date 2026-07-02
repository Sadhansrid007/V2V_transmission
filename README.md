An intelligent, completely local **Retrieval-Augmented Generation (RAG)** application. Type a question or request a summary of a video, and a local LLM will analyze the transcript text, explain the answer, and seamlessly cut and stitch the corresponding video highlights side-by-side in a web dashboard.

---

## 🌟 Features
* **🧠 100% Local Intelligence:** Powered by Ollama (`llama3.2:1b`)—no external API keys or cloud dependencies required.
* **🔍 Semantic Video Search:** Uses ChromaDB to understand the *meaning* behind your queries, bypassing simple keyword matching.
* **✂️ Dynamic Slicing & Stitching:** Orchestrates MoviePy v2 to instantly crop video frames using precise timestamp structures.
* **📱 Split-Screen UI:** Provides a beautiful, parallel Streamlit interface featuring text explanations alongside live video playback.

---

## 🛠️ The Tech Stack
* **Frontend:** Streamlit (Wide-mode UI Layout)
* **Vector Engine:** ChromaDB (Persistent local embeddings storage)
* **LLM Orchestrator:** Ollama & LangChain
* **Video Automation:** MoviePy v2 (`.subclipped()` engine)

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/Sadhansrid007/V2V_transmission.git]
cd V2V_transmission
```

### 2. Install Dependencies
```bash
pip install streamlit chromadb langchain ollama moviepy
```

### 3. Initialize Your Local Model
```bash
ollama pull llama3.2:1b
```
### 4. Fire Up the Dashboard!
```bash
streamlit run app.py
```



