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

- Query Rewrite:
English → Chinese / bilingual query
- Dynamic k:
Definition: k=3,
Comparison: k=8
- Document Selection:
Definition → 1 wiki + optional raw, 
Comparison → 2 wiki + raw supplement

👉 Enables precision + coverage balance

### 4️⃣ Task-aware Prompt Engineering

Different prompts for different question types:

Definition: concise, no expansion

Comparison: structured, table output

Fallback: allow controlled knowledge completion

### 5️⃣ Token Optimization

- Reduced chunk size (800 → 500)
- Limited wiki generation (33 → 3 topics)
- Context truncation (600 chars/doc)
- Dynamic retrieval size

👉 ~70%+ token reduction


## 🏗️ System Architecture

```mermaid
graph TD
    A[User Query] --> B[Query Rewrite (EN/CN)]
    B --> C[Vector Search (FAISS)]
    C --> D[Controlled Retrieval]
    D --> E[LLM Generation]
    E --> F[Structured Answer]
User Query → Query Rewrite (EN/CN) → Vector Search (FAISS) → Controlled Retrieval → LLM Generation → Structured Answer
```

## 📂 Project Structure
```bash
pdf_ai_project/
├── raw/                    # Original PDFs
├── wiki/                   # Structured knowledge base
├── faiss_index/            # Vector database
├── build_wiki.py           # Generate wiki pages
├── build_vectorstore.py    # Build vector DB
├── rag.py                  # Retrieval + generation logic
├── app.py                  # Streamlit UI
└── .env                    # API keys
```
## ▶️ How to Run

 Step 1: Generate wiki knowledge python build_wiki.py  
 
 Step 2: Build vector database python build_vectorstore.py
 
 Step 3: Launch app streamlit run app.py

## 📊 Example Capabilities

✔ Definition questions

✔ Concept comparison

✔ Cross-language queries

✔ Structured explanations

## 💡 Highlights

Hybrid retrieval improves answer accuracy

Controlled pipeline avoids irrelevant context

Task-aware generation improves readability

Engineering-focused optimization (token, latency)

## 🧠 Future Improvements

Reranker integration (e.g. cross-encoder)

Better source filtering

Multi-hop reasoning

Evaluation metrics (EM / F1)

## 📬 Contact

1572408266@qq.com
