from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from intent_router import IntentResult


@dataclass
class RetrievalPlan:
    intent: str
    queries: list[str]
    raw_top_k: int = 3
    wiki_top_k: int = 1
    fetch_k: int = 12
    use_mmr: bool = True
    use_rerank: bool = False
    need_coverage_check: bool = True
    min_score: float = 0.2
    source_filter: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def total_top_k(self) -> int:
        return self.raw_top_k + self.wiki_top_k

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_top_k"] = self.total_top_k
        return payload


def _clean_query_fragment(text: str) -> str:
    return text.strip(" ：:，,？?。.!！;；\"'`()[]{}")


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        candidate = value.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def build_queries(question: str, intent: str) -> list[str]:
    question = question.strip()
    queries = [question]

    if intent == "comparison":
        lowered = question.lower()
        separators = ["和", "与", "vs", "versus", "区别", "difference between", "compare"]
        for separator in separators:
            if separator.lower() in lowered:
                parts = re.split(separator, question, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    left = _clean_query_fragment(parts[0])
                    right = _clean_query_fragment(parts[1])
                    right = re.sub(
                        r"^(有什么区别|有什么不同|区别是什么|difference|compare)\s*",
                        "",
                        right,
                        flags=re.IGNORECASE,
                    )
                    right = re.sub(
                        r"\s*(有什么区别|有什么不同|区别是什么|difference|compare)\s*$",
                        "",
                        right,
                        flags=re.IGNORECASE,
                    )
                    queries = [left, _clean_query_fragment(right), question, f"{left} {right} difference"]
                    return _dedupe_keep_order(queries)

    if intent == "definition":
        stripped = question
        for phrase in ["什么是", "啥是", "what is", "define", "meaning of", "解释一下", "介绍一下"]:
            stripped = re.sub(phrase, "", stripped, flags=re.IGNORECASE).strip()
        stripped = _clean_query_fragment(stripped)
        if stripped and stripped != question:
            queries.append(stripped)
    elif intent == "summary":
        queries.extend([f"{question} main points", f"{question} chapter summary"])
    elif intent == "quiz":
        queries.extend([f"{question} key concepts", f"{question} exam points"])
    elif intent == "diagnosis":
        queries.extend([f"{question} coverage", f"{question} course material"])
    elif intent == "study_plan":
        queries.extend([f"{question} roadmap", f"{question} key concepts"])
    else:
        queries.append(question)

    return _dedupe_keep_order(queries)


def create_retrieval_plan(question: str, intent_result: IntentResult) -> RetrievalPlan:
    intent = intent_result.intent
    queries = build_queries(question, intent)

    if intent == "definition":
        return RetrievalPlan(
            intent=intent,
            queries=queries,
            raw_top_k=2,
            wiki_top_k=2,
            fetch_k=10,
            use_mmr=True,
            use_rerank=False,
            need_coverage_check=True,
            min_score=0.18,
            reason="Definition questions benefit from a compact wiki definition plus a small amount of grounded raw evidence.",
        )

    if intent == "comparison":
        return RetrievalPlan(
            intent=intent,
            queries=queries,
            raw_top_k=3,
            wiki_top_k=2,
            fetch_k=18,
            use_mmr=True,
            use_rerank=True,
            need_coverage_check=True,
            min_score=0.18,
            reason="Comparison questions should decompose the concepts, retrieve both sides, and rerank to avoid over-focusing on one term.",
        )

    if intent == "summary":
        return RetrievalPlan(
            intent=intent,
            queries=queries,
            raw_top_k=6,
            wiki_top_k=2,
            fetch_k=24,
            use_mmr=True,
            use_rerank=False,
            need_coverage_check=False,
            min_score=0.14,
            reason="Summaries need broader raw coverage so the answer reflects the course structure rather than a single chunk.",
        )

    if intent == "quiz":
        return RetrievalPlan(
            intent=intent,
            queries=queries,
            raw_top_k=5,
            wiki_top_k=2,
            fetch_k=22,
            use_mmr=True,
            use_rerank=False,
            need_coverage_check=True,
            min_score=0.14,
            reason="Quiz generation needs multiple raw chunks and diversified retrieval so questions cover several concepts.",
        )

    if intent == "diagnosis":
        return RetrievalPlan(
            intent=intent,
            queries=queries,
            raw_top_k=4,
            wiki_top_k=2,
            fetch_k=20,
            use_mmr=False,
            use_rerank=True,
            need_coverage_check=True,
            min_score=0.2,
            reason="Diagnosis is not direct QA; it should inspect whether the course material really covers the request before answering.",
        )

    if intent == "study_plan":
        return RetrievalPlan(
            intent=intent,
            queries=queries,
            raw_top_k=5,
            wiki_top_k=2,
            fetch_k=20,
            use_mmr=True,
            use_rerank=False,
            need_coverage_check=False,
            min_score=0.14,
            reason="Study plans need broader coverage and concept ordering, so the planner keeps more raw context than normal QA.",
        )

    return RetrievalPlan(
        intent="general_qa",
        queries=queries,
        raw_top_k=3,
        wiki_top_k=1,
        fetch_k=12,
        use_mmr=True,
        use_rerank=False,
        need_coverage_check=True,
        min_score=0.18,
        reason="General QA keeps a modest mix of raw evidence and wiki support while avoiding unnecessary context.",
    )
