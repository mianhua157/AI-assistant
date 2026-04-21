# 🧠 Hybrid RAG System for Machine Learning Q&A
🚀 A complete RAG system with local embedding + FAISS retrieval + LLM generation
An AI-powered course assistant system based on Large Language Models (LLM), designed to answer machine learning course questions using Retrieval-Augmented Generation (RAG).

## 📌 Project Overview

This project implements a hybrid Retrieval-Augmented Generation (RAG) system for machine learning question answering.

Unlike basic RAG pipelines, this system introduces:

- Hybrid Knowledge Base (PDF + Structured Wiki)
- Controlled Retrieval Strategy
- Query Rewriting (EN ↔ CN)
- Task-aware Generation (Definition vs Comparison)

## 🚀 Key Features

### 1️⃣ Hybrid Knowledge Base

- 📄 Raw PDFs → complete but noisy
- 📖 Wiki Pages → structured, LLM-generated summaries

👉 Combined into a unified vector store

### 2️⃣ Structured Wiki Generation

Each concept is converted into a structured markdown page with:

- Definition
- Training Process
- Prediction Process
- Common Methods
- Relationships

👉 Improves retrieval alignment and answer quality

### 3️⃣ Controlled Retrieval Pipeline (Core Innovation)

Instead of naive top-k retrieval:

- Query Rewrite
English → Chinese / bilingual query
- Dynamic k
Definition: k=3
Comparison: k=8
- Document Selection
Definition → 1 wiki + optional raw
Comparison → 2 wiki + raw supplement

👉 Enables precision + coverage balance
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
