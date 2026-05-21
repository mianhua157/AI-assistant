from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from retrieval_planner import RetrievalPlan


@dataclass
class RetrievalExecutionResult:
    docs: list[Any]
    trace: list[dict[str, Any]]
    raw_candidates: int
    wiki_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_doc_type(value: str | None) -> str:
    if value == "wiki":
        return "wiki"
    return "raw"


def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _query_terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+|[\u4e00-\u9fff]{2,}", normalized)
    return [token for token in tokens if len(token) > 1]


def _overlap_score(query: str, doc_text: str) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    lowered_doc = _normalize_text(doc_text)
    hits = sum(1 for term in terms if term in lowered_doc)
    return hits / len(terms)


def annotate_doc(doc: Any, *, query: str, distance: float) -> Any:
    annotated = copy.deepcopy(doc)
    annotated.metadata = dict(getattr(annotated, "metadata", {}))
    annotated.metadata["chunk_type"] = normalize_doc_type(annotated.metadata.get("type"))
    annotated.metadata["retrieval_query"] = query
    annotated.metadata["retrieval_distance"] = round(distance, 6)
    annotated.metadata["retrieval_score"] = round(_distance_to_score(distance), 6)
    annotated.metadata["overlap_score"] = round(_overlap_score(query, annotated.page_content[:1200]), 6)
    return annotated


def deduplicate_docs(docs: list[Any]) -> list[Any]:
    best_by_key: dict[tuple[Any, ...], Any] = {}
    for doc in docs:
        metadata = getattr(doc, "metadata", {})
        key = (
            metadata.get("source", "unknown"),
            metadata.get("page"),
            metadata.get("chunk_type", normalize_doc_type(metadata.get("type"))),
            doc.page_content[:200],
        )
        current = best_by_key.get(key)
        if current is None or metadata.get("retrieval_score", 0.0) > current.metadata.get("retrieval_score", 0.0):
            best_by_key[key] = doc
    return list(best_by_key.values())


def rerank_docs(query_bundle: str, docs: list[Any]) -> list[Any]:
    reranked: list[Any] = []
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}))
        semantic_score = float(metadata.get("retrieval_score", 0.0))
        overlap_score = _overlap_score(query_bundle, doc.page_content[:1600])
        type_bonus = 0.03 if metadata.get("chunk_type") == "raw" else 0.0
        final_score = semantic_score * 0.7 + overlap_score * 0.3 + type_bonus
        metadata["rerank_score"] = round(final_score, 6)
        doc.metadata = metadata
        reranked.append(doc)
    return sorted(
        reranked,
        key=lambda item: item.metadata.get("rerank_score", item.metadata.get("retrieval_score", 0.0)),
        reverse=True,
    )


def diversify_docs(docs: list[Any], limit: int) -> list[Any]:
    if limit <= 0:
        return []

    diversified: list[Any] = []
    used_sources: set[str] = set()

    primary = sorted(
        docs,
        key=lambda item: item.metadata.get("rerank_score", item.metadata.get("retrieval_score", 0.0)),
        reverse=True,
    )

    for doc in primary:
        source = str(doc.metadata.get("source", "unknown"))
        if source in used_sources:
            continue
        diversified.append(doc)
        used_sources.add(source)
        if len(diversified) >= limit:
            return diversified

    for doc in primary:
        if len(diversified) >= limit:
            break
        if doc not in diversified:
            diversified.append(doc)

    return diversified


def execute_retrieval_plan(plan: RetrievalPlan, retriever: Any) -> RetrievalExecutionResult:
    raw_candidates: list[Any] = []
    wiki_candidates: list[Any] = []
    trace: list[dict[str, Any]] = []

    for query in plan.queries:
        results = retriever.similarity_search_with_score(query, k=plan.fetch_k)
        annotated_docs = [annotate_doc(doc, query=query, distance=distance) for doc, distance in results]

        raw_hits = 0
        wiki_hits = 0
        for doc in annotated_docs:
            chunk_type = doc.metadata.get("chunk_type", "raw")
            if chunk_type == "wiki":
                wiki_candidates.append(doc)
                wiki_hits += 1
            else:
                raw_candidates.append(doc)
                raw_hits += 1

        trace.append(
            {
                "query": query,
                "raw_hits": raw_hits,
                "wiki_hits": wiki_hits,
                "total_hits": len(annotated_docs),
            }
        )

    raw_unique = deduplicate_docs(raw_candidates)
    wiki_unique = deduplicate_docs(wiki_candidates)

    if plan.use_rerank:
        bundle = " ".join(plan.queries)
        raw_unique = rerank_docs(bundle, raw_unique)
        wiki_unique = rerank_docs(bundle, wiki_unique)
    else:
        raw_unique = sorted(raw_unique, key=lambda item: item.metadata.get("retrieval_score", 0.0), reverse=True)
        wiki_unique = sorted(wiki_unique, key=lambda item: item.metadata.get("retrieval_score", 0.0), reverse=True)

    if plan.use_mmr:
        selected_raw = diversify_docs(raw_unique, plan.raw_top_k)
        selected_wiki = diversify_docs(wiki_unique, plan.wiki_top_k)
    else:
        selected_raw = raw_unique[: plan.raw_top_k]
        selected_wiki = wiki_unique[: plan.wiki_top_k]

    final_docs = selected_raw + selected_wiki
    if plan.intent in {"definition", "comparison", "general_qa"}:
        final_docs = selected_wiki + selected_raw

    return RetrievalExecutionResult(
        docs=final_docs,
        trace=trace,
        raw_candidates=len(raw_unique),
        wiki_candidates=len(wiki_unique),
    )
