# 📄 RAG PDF QA System
🚀 A complete RAG system with local embedding + FAISS retrieval + LLM generation
An AI-powered course assistant system based on Large Language Models (LLM), designed to answer machine learning course questions using Retrieval-Augmented Generation (RAG).

## 📌 Project Overview

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline for course question answering.

It retrieves relevant content from lecture materials using semantic search, and generates accurate answers with an LLM. The system also includes fallback mechanisms to ensure robustness when retrieval fails.


## 🚀 Features

- 🔍 Semantic retrieval using embeddings (vector search)
- 🧠 RAG-based question answering (Retrieval + Generation)
- ✏️ Query rewriting to improve retrieval quality
- 🛡️ Fallback mechanism for low-quality retrieval results
- 🌐 English materials → Chinese answers (translation support)
- 💻 Interactive UI built with Streamlit
- ⚡ Caching for repeated queries


## 🧰 Tech Stack

- Python
- Streamlit (UI)
- DashScope API (LLM: Qwen)
- FAISS (vector store)
- LangChain (retrieval pipeline)
- Sentence-Transformers (embedding model)


## ⚙️ Pipeline

```text
User Query → Query Rewrite → Retrieval → (Fallback Check) → LLM Generation → Answer
```


## 📁 Project Structure

```
pdf_ai_project/
├── app.py                # Streamlit frontend
├── rag.py                # Core RAG logic
├── rebuild_faiss.py      # Rebuild vector index
├── requirements.txt
├── faiss_index/
└── README.md
```

---

## ⚠️ First Run Notice

The embedding model will be automatically downloaded on first run (~90MB).  
Please wait for the download to complete.


## 🔑 API Key Setup

This project relies on DashScope for large language model inference.  
Please configure your API key before running the application.

```Windows (CMD)
set DASHSCOPE_API_KEY=your_api_key
```

## 🚀 Getting Started
1. Clone the repository
```
git clone <your-repo-url>
cd pdf_ai_project
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the application
```
streamlit run app.py
```

## 🔄 Rebuild FAISS Index (Optional)
If you update your PDF data or change the embedding model:
```Bash
python rebuild_faiss.py
```

## 💡 Future Improvements
Multi-document upload support
Chat history memory
Streaming output
Web deployment

## 📬 Contact
1572408266@qq.com
