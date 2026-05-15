# RAG PDF QA System

A course-assistant demo built with Retrieval-Augmented Generation (RAG), local embeddings, FAISS retrieval, and DashScope/Qwen generation.

## Project Overview

This project is designed to answer machine learning course questions using lecture materials and a local vector index.

The pipeline combines:

- semantic retrieval over course materials
- query rewriting for better search quality
- intent routing for task-aware handling
- retrieval planning before search execution
- coverage checks before grounded answer generation
- LLM answer generation
- a fallback path when retrieval quality is too low
- a Streamlit demo UI

## Features

- Semantic retrieval with local embeddings
- FAISS-based vector search
- RAG question answering
- Intent-aware retrieval planning
- Raw/wiki mixed-source balancing
- Coverage-aware refusal for weak retrieval
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
User Query
-> Intent Router
-> Retrieval Planner
-> Plan Executor
-> Coverage Checker
-> Prompt Builder
-> LLM Generation
-> Answer
```

## Retrieval Planner

Instead of using one fixed retrieval strategy for every question, the system now builds a small retrieval plan before searching.

Different intents use different retrieval behavior:

- `definition`: small wiki support plus a small amount of raw evidence
- `comparison`: query decomposition plus reranking to retrieve both concepts
- `summary`: broader raw coverage so chapter-level summaries are less fragmentary
- `quiz`: diversified retrieval so generated questions cover multiple concepts
- `diagnosis`: prioritize coverage inspection before direct answer generation

This makes the pipeline easier to debug and explain because the app can show:

- detected intent
- planned queries
- raw/wiki document balance
- coverage status
- retrieved evidence

## Project Structure

```text
pdf_ai_project/
|-- app.py
|-- eval_agent.py
|-- llm_client.py
|-- rag.py
|-- build_vectorstore.py
|-- build_wiki.py
|-- run_demo.bat
|-- run_demo.ps1
|-- requirements.txt
|-- eval/
|   |-- eval_questions.jsonl
|   |-- judge_prompt.txt
|   `-- runs/
|-- raw/
|-- wiki/
|-- faiss_index/
`-- README.md
```

## First Run Notice

On first run, the embedding model may need to be available locally before the app can run.

If your environment cannot reach Hugging Face, set `EMBEDDING_MODEL_PATH` in `.env` to a local `sentence-transformers/all-MiniLM-L6-v2` snapshot directory.

## API Key Setup

This project uses DashScope for model inference. Make sure `DASHSCOPE_API_KEY` is available before running the app.

Example:

```env
DASHSCOPE_API_KEY=your_api_key
```

You can store it in a local `.env` file. A sample file is included as `.env.example`.

Optional offline setting:

```env
EMBEDDING_MODEL_PATH=C:\Users\yourname\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\<snapshot-id>
```

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

If the embedding model is not already cached, the startup may still fail in a restricted network environment. In that case, point `EMBEDDING_MODEL_PATH` to a local model snapshot first.

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

## Evaluation Agent

The project now includes a lightweight evaluation agent for regression testing.

It can:

- read questions from `eval/eval_questions.jsonl`
- batch-run `ask_rag()` against the local vector store
- record answers, rewritten queries, intents, coverage status, and retrieved sources
- compute simple retrieval metrics such as wiki hit, raw hit, keyword hit, and latency
- optionally call an LLM-as-judge with `eval/judge_prompt.txt`
- generate run artifacts under `eval/runs/<timestamp>/`

Run a basic evaluation first:

```bash
python eval_agent.py --limit 5
```

Run the judged version after the basic run looks healthy:

```bash
python eval_agent.py --limit 5 --judge
```

Each run produces:

- `results.jsonl`: one JSON record per case
- `summary.json`: aggregate metrics for the run
- `report.md`: human-readable summary, failure analysis, and case details

This is especially useful for tracking:

- which intents are performing well
- whether retrieval is pulling wiki/raw evidence as expected
- which cases are failing because of coverage or routing
- whether bad cases are hard negatives or true course-coverage gaps

## Future Improvements

- Multi-document upload support
- Chat history memory
- Streaming output
- Web deployment
- Agent-based tutoring features

## Contact

`1572408266@qq.com`
