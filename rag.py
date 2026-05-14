from __future__ import annotations

import copy
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

from coverage_checker import check_coverage
from embeddings_utils import DEFAULT_EMBEDDING_MODEL, load_sentence_transformer
from intent_router import IntentResult, route_intent
from prompts import select_prompt
from retrieval_executor import execute_retrieval_plan, normalize_doc_type
from retrieval_planner import RetrievalPlan, create_retrieval_plan


load_dotenv()

DEBUG = False


def get_dashscope_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Please set DASHSCOPE_API_KEY before running the demo.")
    return api_key


class LocalEmbeddings:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        try:
            self.model = load_sentence_transformer(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load local embedding model: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()


class SimpleFAISSRetriever:
    def __init__(self, index_path: str = "faiss_index"):
        index_file = Path(index_path) / "index.faiss"
        pkl_file = Path(index_path) / "index.pkl"

        if not index_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                f"Vector index files were not found at {index_file} and {pkl_file}. "
                "Please run python build_vectorstore.py first."
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

    def _search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        return self.index.search(
            query_embedding_np,
            k=min(k, len(self.index_to_docstore_id)),
        )

    def similarity_search(self, query: str, k: int = 5) -> list[Any]:
        distances, indices = self._search(query, k)
        docs: list[Any] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.index_to_docstore_id.get(idx)
            if doc_id is None or doc_id not in self.docstore:
                continue
            doc = copy.deepcopy(self.docstore[doc_id])
            doc.metadata = dict(getattr(doc, "metadata", {}))
            doc.metadata["retrieval_distance"] = float(distance)
            doc.metadata["chunk_type"] = normalize_doc_type(doc.metadata.get("type"))
            docs.append(doc)
        return docs

    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Any, float]]:
        distances, indices = self._search(query, k)
        results: list[tuple[Any, float]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.index_to_docstore_id.get(idx)
            if doc_id is None or doc_id not in self.docstore:
                continue
            results.append((self.docstore[doc_id], float(distance)))
        return results


def load_vectorstore(index_path: str = "faiss_index") -> SimpleFAISSRetriever:
    return SimpleFAISSRetriever(index_path)


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

    if len(query.split()) > 8:
        return query

    prompt = f"""Translate the following machine learning question into Chinese.
Return only the translated question.

Question: {query}

Chinese:"""

    try:
        response = _call_generation(prompt, temperature=0)
        if response.status_code == HTTPStatus.OK:
            return response.output.text.strip() or query
    except Exception:
        if DEBUG:
            print("Failed to rewrite query to Chinese. Falling back to original query.")

    return query


def rewrite_query_bilingual(query: str) -> str:
    if not re.search(r"[a-zA-Z]", query):
        return query

    translated = rewrite_query_to_chinese(query)
    if translated == query:
        return query
    return f"{query} {translated}"


def _prepend_retrieval_query_variant(
    queries: list[str],
    rewritten_query: str,
    original_question: str,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for candidate in (rewritten_query, original_question, *queries):
        value = candidate.strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)

    return ordered


def format_sources(docs: list[Any]) -> list[dict[str, Any]]:
    sources = []
    for i, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {})
        preview = doc.page_content[:180].strip().replace("\n", " ")
        if len(doc.page_content.strip()) > 180:
            preview += "..."

        sources.append(
            {
                "id": i,
                "content": doc.page_content.strip(),
                "preview": preview,
                "source": metadata.get("source", "unknown"),
                "type": metadata.get("chunk_type", normalize_doc_type(metadata.get("type"))),
                "page": metadata.get("page"),
                "score": metadata.get("retrieval_score"),
                "query": metadata.get("retrieval_query"),
            }
        )
    return sources


def build_context(docs: list[Any], limit_per_doc: int = 700) -> str:
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source", "unknown")
        doc_type = metadata.get("chunk_type", normalize_doc_type(metadata.get("type")))
        score = metadata.get("retrieval_score", 0.0)
        content = doc.page_content[:limit_per_doc]
        context_blocks.append(
            f"[{i}] source={source} type={doc_type} score={score}\n{content}"
        )
    return "\n\n".join(context_blocks)


def build_coverage_refusal(question: str, intent: str, coverage_reason: str) -> str:
    return (
        "课程资料对这个问题的覆盖不足，当前不建议直接给出确定答案。\n\n"
        f"- intent: {intent}\n"
        f"- question: {question}\n"
        f"- reason: {coverage_reason}\n\n"
        "如果你愿意，我可以继续帮你：\n"
        "1. 改写问题，让它更贴近课程资料里的表述\n"
        "2. 仅基于已覆盖的部分给一个保守总结\n"
        "3. 说明还缺哪类资料"
    )


def ask_fallback(question: str, fallback_reason: str, intent: str) -> dict[str, Any]:
    prompt = f"""你是一个机器学习课程助教。课程资料不足时，可以基于已有知识做有限补充。
问题：{question}

当前 intent：{intent}

要求：
1. 先明确说明课程资料覆盖不足。
2. 再给出尽可能稳妥的解释。
3. 不要编造不存在的课程细节。
4. 用中文回答。"""

    try:
        response = _call_generation(prompt, temperature=0.1, max_tokens=600)
        if response.status_code == HTTPStatus.OK:
            answer = response.output.text
        else:
            answer = f"Request failed: {response.code} - {response.message}"
    except Exception as exc:
        answer = f"Request failed: {exc}"

    return {
        "answer": answer,
        "sources": [],
        "doc_count": 0,
        "fallback_used": True,
        "intent": intent,
        "intent_reason": fallback_reason,
    }


def generate_answer(question: str, docs: list[Any], intent: str) -> str:
    prompt_template = select_prompt(intent)
    context = build_context(docs)
    prompt = prompt_template.format(question=question, context=context)

    response = _call_generation(prompt, temperature=0, max_tokens=900)
    if response.status_code == HTTPStatus.OK:
        return response.output.text
    return f"Request failed: {response.code} - {response.message}"


def _build_result(
    *,
    answer: str,
    intent_result: IntentResult,
    retrieval_query_input: str,
    plan: RetrievalPlan,
    docs: list[Any],
    coverage: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    fallback_used: bool,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": format_sources(docs),
        "doc_count": len(docs),
        "fallback_used": fallback_used,
        "intent": intent_result.intent,
        "intent_reason": intent_result.reason,
        "retrieval_query_input": retrieval_query_input,
        "queries": plan.queries,
        "plan": plan.to_dict(),
        "coverage": coverage,
        "execution_trace": execution_trace,
    }


def ask_rag(question: str, vectorstore: Optional[SimpleFAISSRetriever] = None) -> dict[str, Any]:
    retriever = vectorstore or load_vectorstore()
    intent_result = route_intent(question)
    rewritten_query = rewrite_query_bilingual(question)
    plan = create_retrieval_plan(question, intent_result)
    plan.queries = _prepend_retrieval_query_variant(plan.queries, rewritten_query, question)
    execution = execute_retrieval_plan(plan, retriever)
    docs = execution.docs

    coverage_report = check_coverage(question, docs, plan)
    coverage = coverage_report.to_dict()

    if not docs:
        result = ask_fallback(question, "No relevant material was retrieved.", intent_result.intent)
        result["plan"] = plan.to_dict()
        result["coverage"] = coverage
        result["execution_trace"] = execution.trace
        result["retrieval_query_input"] = rewritten_query
        result["queries"] = plan.queries
        return result

    if plan.need_coverage_check and not coverage_report.can_answer and intent_result.intent != "diagnosis":
        return _build_result(
            answer=build_coverage_refusal(question, intent_result.intent, coverage_report.reason),
            intent_result=intent_result,
            retrieval_query_input=rewritten_query,
            plan=plan,
            docs=docs,
            coverage=coverage,
            execution_trace=execution.trace,
            fallback_used=False,
        )

    total_content_len = sum(len(doc.page_content) for doc in docs)
    if total_content_len < 120 and intent_result.intent != "diagnosis":
        result = ask_fallback(
            question,
            "Retrieved content was too short for a grounded answer.",
            intent_result.intent,
        )
        result["plan"] = plan.to_dict()
        result["coverage"] = coverage
        result["execution_trace"] = execution.trace
        result["retrieval_query_input"] = rewritten_query
        result["queries"] = plan.queries
        return result

    try:
        answer = generate_answer(question, docs, intent_result.intent)
    except Exception as exc:
        answer = f"Request failed: {exc}"

    return _build_result(
        answer=answer,
        intent_result=intent_result,
        retrieval_query_input=rewritten_query,
        plan=plan,
        docs=docs,
        coverage=coverage,
        execution_trace=execution.trace,
        fallback_used=False,
    )
