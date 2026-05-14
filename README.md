# RAG PDF QA System

A course-assistant demo built with Retrieval-Augmented Generation (RAG), local embeddings, FAISS retrieval, and DashScope/Qwen generation.

## Project Overview

This project is designed to answer machine learning course questions using lecture materials and a local vector index.

The pipeline combines:

- semantic retrieval over course materials
- query rewriting for better search quality
- LLM answer generation
- a fallback path when retrieval quality is too low
- a Streamlit demo UI

## Features

- Semantic retrieval with local embeddings
- FAISS-based vector search
- RAG question answering
- Query rewriting for short English questions
- Fallback answering when retrieval is weak
- Streamlit web interface
- Support for course-oriented Q&A workflows

## Tech Stack

- Python
- Streamlit
- DashScope / Qwen
- FAISS
- LangChain community loaders and splitters
- Sentence-Transformers

## Pipeline

```text
User Query -> Query Rewrite -> Retrieval -> Fallback Check -> LLM Generation -> Answer
```

## Project Structure

```text
pdf_ai_project/
|-- app.py
|-- rag.py
|-- build_vectorstore.py
|-- build_wiki.py
|-- run_demo.bat
|-- run_demo.ps1
|-- requirements.txt
|-- raw/
|-- wiki/
|-- faiss_index/
`-- README.md
```

## First Run Notice

On first run, the embedding model may be downloaded automatically. This can take a little while depending on your network and Python environment.

## API Key Setup

This project uses DashScope for model inference. Make sure `DASHSCOPE_API_KEY` is available before running the app.

Example:

```env
DASHSCOPE_API_KEY=your_api_key
```

You can store it in a local `.env` file. A sample file is included as `.env.example`.

## One-Click Demo Start

If you are on Windows, you can start the demo by double-clicking:

```text
run_demo.bat
```

Or run it manually in PowerShell:

```powershell
./run_demo.ps1
```

The startup script will:

1. Check Python
2. Check or create `.env`
3. Prompt for `DASHSCOPE_API_KEY` if it is missing
4. Install dependencies from `requirements.txt`
5. Build the FAISS index automatically if it is missing
6. Start the Streamlit app

## Getting Started

1. Clone the repository

```bash
git clone <your-repo-url>
cd pdf_ai_project
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Build the vector index if needed

```bash
python build_vectorstore.py
```

4. Start the app

```bash
streamlit run app.py
```

## Rebuild FAISS Index

If you update the PDFs, wiki files, or embedding workflow, rebuild the vector index:

```bash
python build_vectorstore.py
```

## Future Improvements

- Multi-document upload support
- Chat history memory
- Streaming output
- Web deployment
- Agent-based tutoring features

## Contact

`1572408266@qq.com`
