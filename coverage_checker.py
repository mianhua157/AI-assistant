from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from retrieval_planner import RetrievalPlan


QUESTION_STOPWORDS = {
    "what",
    "what's",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "how",
    "does",
    "do",
    "did",
    "why",
    "explain",
    "define",
    "meaning",
    "summary",
    "summarize",
    "compare",
    "difference",
    "versus",
    "vs",
    "quiz",
    "plan",
    "study",
    "course",
    "material",
    "coverage",
    "什么是",
    "啥是",
    "定义",
    "解释一下",
    "解释",
    "介绍一下",
    "总结",
    "概括",
    "归纳",
    "区别",
    "不同",
    "对比",
    "比较",
    "帮我",
    "生成",
    "复习",
    "学习计划",
    "资料里",
    "课程里",
    "有没有",
    "没讲",
    "没提到",
    "覆盖",
}


QUESTION_PHRASE_PREFIXES = [
    "what is",
    "what's",
    "meaning of",
    "define",
    "explain",
    "什么是",
    "啥是",
    "定义",
    "解释一下",
    "介绍一下",
    "资料里有没有",
    "课程里有没有",
]


@dataclass
class CoverageReport:
    status: str
    can_answer: bool
    reason: str
    matched_terms: list[str]
    total_terms: int
    strong_doc_count: int
    raw_doc_count: int
    wiki_doc_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _extract_question_terms(question: str) -> list[str]:
    normalized = _normalize_text(question)
    for phrase in QUESTION_PHRASE_PREFIXES:
        normalized = normalized.replace(phrase, " ")
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+|[\u4e00-\u9fff]{2,}", normalized)
    filtered = [token for token in tokens if len(token) > 1 and token not in QUESTION_STOPWORDS]
    seen: set[str] = set()
    deduped: list[str] = []
    for token in filtered:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def check_coverage(question: str, docs: list[Any], plan: RetrievalPlan) -> CoverageReport:
    if not docs:
        return CoverageReport(
            status="uncovered",
            can_answer=False,
            reason="No relevant documents were retrieved from the course material.",
            matched_terms=[],
            total_terms=0,
            strong_doc_count=0,
            raw_doc_count=0,
            wiki_doc_count=0,
        )

    coverage_query = " ".join(plan.queries) if plan.queries else question
    terms = _extract_question_terms(coverage_query)
    joined_content = "\n".join(_normalize_text(doc.page_content[:1500]) for doc in docs)
    matched_terms = [term for term in terms if term in joined_content]

    strong_docs = [doc for doc in docs if float(doc.metadata.get("retrieval_score", 0.0)) >= plan.min_score]
    raw_doc_count = sum(1 for doc in docs if doc.metadata.get("chunk_type") == "raw")
    wiki_doc_count = sum(1 for doc in docs if doc.metadata.get("chunk_type") == "wiki")

    required_docs = 1 if plan.intent == "definition" else 2
    if plan.intent in {"summary", "quiz", "study_plan"}:
        required_docs = 3

    if not strong_docs:
        return CoverageReport(
            status="uncovered",
            can_answer=False,
            reason="Retrieved chunks are too weakly related to support a grounded answer.",
            matched_terms=matched_terms,
            total_terms=len(terms),
            strong_doc_count=0,
            raw_doc_count=raw_doc_count,
            wiki_doc_count=wiki_doc_count,
        )

    if len(strong_docs) < required_docs:
        return CoverageReport(
            status="partial",
            can_answer=False,
            reason="Some relevant material was found, but not enough independent evidence was retrieved.",
            matched_terms=matched_terms,
            total_terms=len(terms),
            strong_doc_count=len(strong_docs),
            raw_doc_count=raw_doc_count,
            wiki_doc_count=wiki_doc_count,
        )

    if terms and len(matched_terms) / len(terms) < 0.4:
        return CoverageReport(
            status="partial",
            can_answer=False,
            reason="The retrieved context overlaps with the question only loosely, so it may be a hard-negative match.",
            matched_terms=matched_terms,
            total_terms=len(terms),
            strong_doc_count=len(strong_docs),
            raw_doc_count=raw_doc_count,
            wiki_doc_count=wiki_doc_count,
        )

    if plan.intent != "definition" and raw_doc_count == 0:
        return CoverageReport(
            status="partial",
            can_answer=False,
            reason="Only wiki-style support was found; the answer lacks enough raw course evidence.",
            matched_terms=matched_terms,
            total_terms=len(terms),
            strong_doc_count=len(strong_docs),
            raw_doc_count=raw_doc_count,
            wiki_doc_count=wiki_doc_count,
        )

    return CoverageReport(
        status="covered",
        can_answer=True,
        reason="The retrieved set is sufficiently grounded for answer generation.",
        matched_terms=matched_terms,
        total_terms=len(terms),
        strong_doc_count=len(strong_docs),
        raw_doc_count=raw_doc_count,
        wiki_doc_count=wiki_doc_count,
    )
