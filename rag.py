"""
RAG 核心逻辑：检索、问题改写与回答生成。
"""

import os
import pickle
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional

import dashscope
import faiss
import numpy as np
from dotenv import load_dotenv


load_dotenv()

DEBUG = False


def get_dashscope_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    return api_key


class LocalEmbeddings:
    """使用本地 SentenceTransformer 生成 embedding。"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer

            print("Loading embedding model (may download on first run)...")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_embedding_dimension()
        except Exception as exc:
            raise RuntimeError(f"加载本地 embedding 模型失败：{exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()


class SimpleFAISSRetriever:
    """简单的 FAISS 检索器。"""

    def __init__(self, index_path: str = "faiss_index"):
        index_file = Path(index_path) / "index.faiss"
        pkl_file = Path(index_path) / "index.pkl"

        if not index_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                f"未找到向量库文件：{index_file} 或 {pkl_file}。请先运行 python build_vectorstore.py"
            )

        self.index = faiss.read_index(str(index_file))

        with open(pkl_file, "rb") as file:
            data = pickle.load(file)
            if isinstance(data, tuple) and len(data) == 2:
                self.docstore, self.index_to_docstore_id = data
            else:
                self.docstore = {}
                self.index_to_docstore_id = {}

        self.embeddings = LocalEmbeddings()

    def similarity_search(self, query: str, k: int = 5) -> list[Any]:
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        _, indices = self.index.search(
            query_embedding_np,
            k=min(k, len(self.index_to_docstore_id)),
        )

        docs = []
        for idx in indices[0]:
            if idx < 0:
                continue
            doc_id = self.index_to_docstore_id.get(idx)
            if doc_id is not None and doc_id in self.docstore:
                docs.append(self.docstore[doc_id])

        return docs


def load_vectorstore(index_path: str = "faiss_index") -> SimpleFAISSRetriever:
    return SimpleFAISSRetriever(index_path)


def format_sources(docs: list[Any]) -> list[dict[str, Any]]:
    sources = []
    for i, doc in enumerate(docs, start=1):
        preview = doc.page_content[:150].strip().replace("\n", " ")
        if len(preview) == 150:
            preview += "..."
        sources.append(
            {
                "id": i,
                "content": doc.page_content.strip(),
                "preview": preview,
                "source": doc.metadata.get("source", "unknown"),
                "type": doc.metadata.get("type", "raw"),
            }
        )
    return sources


def _call_generation(prompt: str, *, temperature: float = 0, max_tokens: Optional[int] = None):
    kwargs: dict[str, Any] = {
        "model": "qwen-plus",
        "prompt": prompt,
        "api_key": get_dashscope_api_key(),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return dashscope.Generation.call(**kwargs)


def rewrite_query_to_chinese(query: str) -> str:
    if not re.search(r"[a-zA-Z]", query):
        return query

    if len(query.split()) > 5:
        return query

    prompt = f"""把下面的机器学习相关问题翻译成中文，只输出翻译结果：

问题：{query}

翻译：
"""

    try:
        response = _call_generation(prompt, temperature=0)
        if response.status_code == HTTPStatus.OK:
            translated = response.output.text.strip()
            if translated.startswith("翻译："):
                translated = translated[3:].strip()
            return translated or query
    except Exception:
        if DEBUG:
            print("query 中文改写失败，回退原问题")

    return query


def rewrite_query_bilingual(query: str) -> str:
    if not re.search(r"[a-zA-Z]", query):
        return query

    if len(query.split()) > 5:
        return query

    translated = rewrite_query_to_chinese(query)
    if translated == query:
        return query
    return f"{query} {translated}"


def ask_fallback(question: str, fallback_reason: str = "检索资料不足") -> dict[str, Any]:
    if DEBUG:
        print(f"触发 fallback：{fallback_reason}")

    prompt = f"""你是一个机器学习助教。课程资料不足时，可以基于已有知识做合理补充。

规则：
1. 不要编造不存在的概念。
2. 回答要清晰、准确、简洁。
3. 如果资料不足，请直接承认资料覆盖有限。
4. 对于“what is / 什么是”类问题，优先给核心定义。

问题：{question}

回答：
"""

    try:
        response = _call_generation(prompt, temperature=0.1, max_tokens=500)
        if response.status_code == HTTPStatus.OK:
            return {
                "answer": response.output.text,
                "sources": [],
                "doc_count": 0,
                "fallback_used": True,
                "intent": "fallback",
            }
        answer = f"请求失败：{response.code} - {response.message}"
    except Exception as exc:
        answer = f"请求失败：{exc}"

    return {
        "answer": answer,
        "sources": [],
        "doc_count": 0,
        "fallback_used": True,
        "intent": "fallback",
    }


def detect_intent(query: str) -> str:
    normalized = query.lower()
    if normalized.startswith("what is") or normalized.startswith("什么是") or "定义" in normalized:
        return "definition"
    if any(token in normalized for token in ["compare", "对比", "比较", "vs", "difference"]):
        return "compare"
    return "general"


def select_docs(candidates: list[Any], intent: str) -> list[Any]:
    wiki_docs = [doc for doc in candidates if doc.metadata.get("type") == "wiki"]
    raw_docs = [doc for doc in candidates if doc.metadata.get("type") == "raw_pdf"]

    docs: list[Any] = []

    if intent == "compare":
        docs.extend(wiki_docs[:2])
    elif wiki_docs:
        docs.append(wiki_docs[0])

    if intent != "compare" and raw_docs:
        docs.append(raw_docs[0])

    docs.extend(raw_docs[1:3])
    return docs


def build_prompt(question: str, docs: list[Any], intent: str) -> str:
    context = "\n\n".join(doc.page_content[:600] for doc in docs)

    if intent == "compare":
        return f"""你是一个机器学习助教，请基于检索到的课程资料回答问题。

规则：
1. 优先使用提供的资料。
2. 回答要清晰、结构化。
3. 比较类问题请明确列出异同点。
4. 如果资料不完整，可以做有限的合理补充，并说明边界。

资料：
{context}

问题：{question}

回答：
"""

    return f"""你是一个机器学习助教，请基于检索到的课程资料回答问题。

规则：
1. 优先使用提供的资料。
2. 不要编造不存在的概念。
3. 回答要清晰、结构化。
4. 如果资料不完整，可以做有限的合理补充，并说明边界。
5. 对于“what is / 什么是”类问题，优先给核心定义。

资料：
{context}

问题：{question}

回答：
"""


def ask_rag(question: str, vectorstore: Optional[SimpleFAISSRetriever] = None) -> dict[str, Any]:
    retriever = vectorstore or load_vectorstore()

    rewritten_query = rewrite_query_bilingual(question)
    intent = detect_intent(rewritten_query)

    if intent == "definition":
        candidates = retriever.similarity_search(rewritten_query, k=3)
    elif intent == "compare":
        candidates = retriever.similarity_search(rewritten_query, k=8)
    else:
        candidates = retriever.similarity_search(rewritten_query, k=6)

    docs = select_docs(candidates, intent)

    if not docs:
        return ask_fallback(question, "暂无相关检索资料")

    total_content_len = sum(len(doc.page_content) for doc in docs)
    if total_content_len < 100:
        return ask_fallback(question, "检索资料内容过少")

    prompt = build_prompt(question, docs, intent)

    try:
        response = _call_generation(prompt, temperature=0)
        if response.status_code == HTTPStatus.OK:
            return {
                "answer": response.output.text,
                "sources": format_sources(docs),
                "doc_count": len(docs),
                "fallback_used": False,
                "intent": intent,
            }
        answer = f"请求失败：{response.code} - {response.message}"
    except Exception as exc:
        answer = f"请求失败：{exc}"

    return {
        "answer": answer,
        "sources": format_sources(docs),
        "doc_count": len(docs),
        "fallback_used": False,
        "intent": intent,
    }
