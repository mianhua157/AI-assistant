import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from coverage_checker import check_coverage
from intent_router import route_intent
from rag import load_vectorstore, rewrite_query_bilingual
from retrieval_executor import execute_retrieval_plan
from retrieval_planner import RetrievalPlan, create_retrieval_plan


BASE_DIR = Path(__file__).parent
DEFAULT_CASES_PATH = BASE_DIR / "eval" / "intent_eval.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "eval" / "retrieval_planner_eval.csv"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def create_baseline_plan(question: str) -> RetrievalPlan:
    return RetrievalPlan(
        intent="general_qa",
        queries=[question],
        raw_top_k=4,
        wiki_top_k=1,
        fetch_k=12,
        use_mmr=False,
        use_rerank=False,
        need_coverage_check=False,
        min_score=0.0,
        reason="Baseline fixed retrieval without planner decomposition or coverage gating.",
    )


def summarize_sources(docs: list[Any], limit: int = 5) -> str:
    parts = []
    for doc in docs[:limit]:
        source = doc.metadata.get("source", "unknown")
        chunk_type = doc.metadata.get("chunk_type", "raw")
        page = doc.metadata.get("page")
        score = doc.metadata.get("retrieval_score", "")
        parts.append(f"{chunk_type}:{source}:page={page}:score={score}")
    return " | ".join(parts)


def evaluate_case(question: str, retriever) -> dict[str, Any]:
    intent_result = route_intent(question)
    baseline_plan = create_baseline_plan(question)
    rewritten_query = rewrite_query_bilingual(question)
    planned = create_retrieval_plan(question, intent_result)
    if rewritten_query not in planned.queries:
        planned.queries = [rewritten_query, *planned.queries]

    baseline_execution = execute_retrieval_plan(baseline_plan, retriever)
    planned_execution = execute_retrieval_plan(planned, retriever)
    planned_coverage = check_coverage(question, planned_execution.docs, planned)

    return {
        "question": question,
        "predicted_intent": intent_result.intent,
        "intent_reason": intent_result.reason,
        "rewritten_query": rewritten_query,
        "baseline_queries": " | ".join(baseline_plan.queries),
        "planned_queries": " | ".join(planned.queries),
        "baseline_doc_count": len(baseline_execution.docs),
        "planned_doc_count": len(planned_execution.docs),
        "baseline_raw_candidates": baseline_execution.raw_candidates,
        "baseline_wiki_candidates": baseline_execution.wiki_candidates,
        "planned_raw_candidates": planned_execution.raw_candidates,
        "planned_wiki_candidates": planned_execution.wiki_candidates,
        "planned_use_rerank": planned.use_rerank,
        "planned_need_coverage_check": planned.need_coverage_check,
        "planned_coverage_status": planned_coverage.status,
        "planned_coverage_reason": planned_coverage.reason,
        "planned_reason": planned.reason,
        "baseline_sources": summarize_sources(baseline_execution.docs),
        "planned_sources": summarize_sources(planned_execution.docs),
        "manual_retrieval_relevance": "",
        "manual_coverage_judgement": "",
        "notes": "",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="Path to a JSONL evaluation dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the CSV report.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N cases.")
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    retriever = load_vectorstore()
    rows = [evaluate_case(case["question"], retriever) for case in cases]
    write_csv(rows, args.output)

    print(f"Wrote retrieval planner evaluation CSV to: {args.output}")
    print(f"Evaluated {len(rows)} cases.")


if __name__ == "__main__":
    main()
