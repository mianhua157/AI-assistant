# 📄 RAG PDF QA System

An AI-powered course assistant system based on Large Language Models (LLM), designed to answer machine learning course questions using Retrieval-Augmented Generation (RAG).

---

## 📌 Project Overview

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline for course question answering.

It retrieves relevant content from lecture materials using semantic search, and generates accurate answers with an LLM. The system also includes fallback mechanisms to ensure robustness when retrieval fails.

---

## 🚀 Features

- 🔍 Semantic retrieval using embeddings (vector search)
- 🧠 RAG-based question answering (Retrieval + Generation)
- ✏️ Query rewriting to improve retrieval quality
- 🛡️ Fallback mechanism for low-quality retrieval results
- 🌐 English materials → Chinese answers (translation support)
- 💻 Interactive UI built with Streamlit
- ⚡ Caching for repeated queries

---

## 🧰 Tech Stack

- Python
- Streamlit (UI)
- DashScope API (LLM: Qwen)
- FAISS (vector store)
- LangChain (retrieval pipeline)
- Sentence-Transformers (embedding model)

---

## ⚙️ Pipeline

```text
User Query → Query Rewrite → Retrieval → (Fallback Check) → LLM Generation → Answer
```

---

## 📁 Project Structure

pdf_ai_project/
├── app.py                # Streamlit frontend
├── rag.py                # Core RAG logic
├── rebuild_faiss.py      # Rebuild vector index
├── requirements.txt
├── faiss_index/
└── README.md


