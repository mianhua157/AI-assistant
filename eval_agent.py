from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rag import ask_rag, load_vectorstore


BASE_DIR = Path(__file__).parent
DEFAULT_QUESTIONS_PATH = BASE_DIR / "eval" / "eval_questions.jsonl"
DEFAULT_JUDGE_PROMPT_PATH = BASE_DIR / "eval" / "judge_prompt.txt"
DEFAULT_RUNS_DIR = BASE_DIR / "eval" / "runs"
JUDGE_SCORE_KEYS = [
    "answer_correctness",
    "source_relevance",
    "faithfulness",
    "language_consistency",
    "hallucination_risk",
]


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def normalize_rag_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "doc_count": result.get("doc_count", 0),
            "rewritten_query": result.get("retrieval_query_input", ""),
            "queries": result.get("queries", []),
            "intent": result.get("intent", ""),
            "intent_reason": result.get("intent_reason", ""),
            "coverage": result.get("coverage", {}),
            "fallback_used": result.get("fallback_used", False),
            "plan": result.get("plan", {}),
        }

    if isinstance(result, tuple):
        if len(result) == 2:
            answer, docs = result
            return {
                "answer": answer,
                "sources": docs,
                "doc_count": len(docs) if isinstance(docs, list) else 0,
                "rewritten_query": "",
                "queries": [],
                "intent": "",
                "intent_reason": "",
                "coverage": {},
                "fallback_used": False,
                "plan": {},
            }
        if len(result) >= 3:
            answer, docs, rewritten_query = result[:3]
            return {
                "answer": answer,
                "sources": docs,
                "doc_count": len(docs) if isinstance(docs, list) else 0,
                "rewritten_query": rewritten_query,
                "queries": [],
                "intent": "",
                "intent_reason": "",
                "coverage": {},
                "fallback_used": False,
                "plan": {},
            }

    return {
        "answer": str(result),
        "sources": [],
        "doc_count": 0,
        "rewritten_query": "",
        "queries": [],
        "intent": "",
        "intent_reason": "",
        "coverage": {},
        "fallback_used": False,
        "plan": {},
    }


def format_source(source: Any, max_chars: int = 400) -> dict[str, Any]:
    if isinstance(source, dict):
        content = str(source.get("content", ""))
        return {
            "type": source.get("type", "unknown"),
            "source": source.get("source", "unknown"),
            "page": source.get("page"),
            "score": source.get("score"),
            "query": source.get("query"),
            "preview": content[:max_chars].replace("\n", " "),
        }

    metadata = dict(getattr(source, "metadata", {}))
    content = str(getattr(source, "page_content", ""))
    return {
        "type": metadata.get("type", metadata.get("chunk_type", "unknown")),
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page"),
        "score": metadata.get("retrieval_score"),
        "query": metadata.get("retrieval_query"),
        "preview": content[:max_chars].replace("\n", " "),
    }


def calc_basic_metrics(item: dict[str, Any], normalized: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    expected_keywords = item.get("expected_keywords", [])
    answer_text = normalized.get("answer", "").lower()
    keyword_hits = [keyword for keyword in expected_keywords if keyword.lower() in answer_text]

    source_types = [str(source.get("type", "unknown")) for source in sources]
    source_paths = [str(source.get("source", "unknown")) for source in sources]
    source_queries = [str(source.get("query", "")) for source in sources if source.get("query")]

    return {
        "num_sources": len(sources),
        "wiki_hit": any(source_type == "wiki" for source_type in source_types),
        "raw_hit": any(source_type != "wiki" for source_type in source_types),
        "keyword_hit_count": len(keyword_hits),
        "keyword_total": len(expected_keywords),
        "keyword_hits": keyword_hits,
        "keyword_hit_rate": round(len(keyword_hits) / len(expected_keywords), 3) if expected_keywords else None,
        "expected_intent": item.get("expected_intent", ""),
        "intent_match": normalized.get("intent", "") == item.get("expected_intent", ""),
        "fallback_used": normalized.get("fallback_used", False),
        "coverage_status": normalized.get("coverage", {}).get("status"),
        "source_types": source_types,
        "source_paths": source_paths,
        "source_queries": source_queries,
    }


def build_judge_input(item: dict[str, Any], record: dict[str, Any]) -> str:
    sources = record.get("sources", [])
    source_lines: list[str] = []

    for index, source in enumerate(sources, start=1):
        source_lines.append(
            f"[{index}] type={source.get('type')} source={source.get('source')} "
            f"page={source.get('page')} score={source.get('score')} query={source.get('query')}\n"
            f"{source.get('preview', '')}"
        )

    sources_text = "\n\n".join(source_lines) if source_lines else "(no sources)"

    return (
        f"用户问题:\n{item.get('question', '')}\n\n"
        f"期望意图:\n{item.get('expected_intent', '')}\n\n"
        f"系统识别意图:\n{record.get('intent', '')}\n\n"
        f"系统回答:\n{record.get('answer', '')}\n\n"
        f"检索来源:\n{sources_text}\n"
    )


def safe_parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {
        "answer_correctness": None,
        "source_relevance": None,
        "faithfulness": None,
        "language_consistency": None,
        "hallucination_risk": None,
        "comments": f"JSON parse failed: {text[:200]}",
    }


def judge_with_llm(item: dict[str, Any], record: dict[str, Any], judge_prompt_path: Path) -> dict[str, Any]:
    from llm_client import call_llm

    judge_prompt = judge_prompt_path.read_text(encoding="utf-8")
    judge_input = build_judge_input(item, record)
    try:
        response_text = call_llm(f"{judge_prompt}\n\n{judge_input}", temperature=0, max_tokens=700)
        return safe_parse_json(response_text)
    except Exception as exc:
        return {
            "answer_correctness": None,
            "source_relevance": None,
            "faithfulness": None,
            "language_consistency": None,
            "hallucination_risk": None,
            "comments": f"Judge failed: {exc}",
        }


def average_numeric(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    wiki_hits = sum(1 for result in results if result["basic_metrics"].get("wiki_hit"))
    raw_hits = sum(1 for result in results if result["basic_metrics"].get("raw_hit"))
    intent_matches = sum(1 for result in results if result["basic_metrics"].get("intent_match"))
    fallback_count = sum(1 for result in results if result["basic_metrics"].get("fallback_used"))
    latencies = [float(result.get("latency_sec", 0.0)) for result in results]

    summary: dict[str, Any] = {
        "total_cases": total,
        "wiki_hit_rate": round(wiki_hits / total, 3) if total else 0.0,
        "raw_hit_rate": round(raw_hits / total, 3) if total else 0.0,
        "intent_match_rate": round(intent_matches / total, 3) if total else 0.0,
        "fallback_rate": round(fallback_count / total, 3) if total else 0.0,
        "average_latency_sec": round(statistics.mean(latencies), 3) if latencies else 0.0,
    }

    judged = [result.get("judge", {}) for result in results if isinstance(result.get("judge"), dict)]
    if judged:
        for key in JUDGE_SCORE_KEYS:
            numeric_values = [float(item[key]) for item in judged if isinstance(item.get(key), (int, float))]
            summary[f"avg_{key}"] = average_numeric(numeric_values)

    return summary


def build_failure_analysis(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []

    weak_cases: list[dict[str, Any]] = []
    for result in results:
        judge = result.get("judge", {})
        basic = result.get("basic_metrics", {})

        low_judge = any(
            isinstance(judge.get(key), (int, float)) and float(judge[key]) <= 3
            for key in ("answer_correctness", "faithfulness", "source_relevance")
        )
        weak_retrieval = (
            basic.get("num_sources", 0) == 0
            or basic.get("coverage_status") in {"partial", "uncovered"}
            or basic.get("keyword_hit_rate") == 0
            or not basic.get("intent_match", True)
        )
        if low_judge or weak_retrieval:
            weak_cases.append(result)

    if not weak_cases:
        lines.append("No obvious failure cases were detected in this run.")
        return lines

    for result in weak_cases[:5]:
        basic = result["basic_metrics"]
        judge = result.get("judge", {})
        reasons: list[str] = []

        if not basic.get("intent_match", True):
            reasons.append("intent routing mismatch")
        if basic.get("coverage_status") in {"partial", "uncovered"}:
            reasons.append(f"coverage={basic.get('coverage_status')}")
        if basic.get("num_sources", 0) == 0:
            reasons.append("no retrieved sources")
        if basic.get("keyword_hit_rate") == 0 and basic.get("keyword_total", 0) > 0:
            reasons.append("answer missed expected keywords")
        if isinstance(judge.get("source_relevance"), (int, float)) and judge["source_relevance"] <= 3:
            reasons.append("judge flagged weak source relevance")
        if isinstance(judge.get("faithfulness"), (int, float)) and judge["faithfulness"] <= 3:
            reasons.append("judge flagged weak faithfulness")

        next_step = "inspect retrieval queries and source mix"
        if result.get("category") == "hard_negative":
            next_step = "add OOD / hard-negative guardrail"
        elif result.get("category") == "comparison":
            next_step = "improve concept decomposition or reranking"
        elif result.get("category") == "definition":
            next_step = "improve bilingual rewrite or wiki retrieval"

        lines.append(f"### {result['id']}")
        lines.append(f"- Problem: {', '.join(reasons) if reasons else 'quality below expectation'}")
        lines.append(f"- Expected behavior: {result.get('question')}")
        lines.append(f"- Next step: {next_step}")
        if judge.get("comments"):
            lines.append(f"- Judge note: {judge['comments']}")
        lines.append("")

    return lines


def write_report(results: list[dict[str, Any]], output_dir: Path) -> Path:
    report_path = output_dir / "report.md"
    summary = summarize_results(results)

    lines = ["# RAG Evaluation Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total questions: {summary['total_cases']}")
    lines.append(f"- Wiki hit rate: {summary['wiki_hit_rate']:.1%}")
    lines.append(f"- Raw hit rate: {summary['raw_hit_rate']:.1%}")
    lines.append(f"- Intent match rate: {summary['intent_match_rate']:.1%}")
    lines.append(f"- Fallback rate: {summary['fallback_rate']:.1%}")
    lines.append(f"- Average latency: {summary['average_latency_sec']:.2f}s")

    for key in JUDGE_SCORE_KEYS:
        value = summary.get(f"avg_{key}")
        if value is not None:
            lines.append(f"- {key}: {value:.2f} / 5")

    lines.append("")
    lines.append("## Failure Analysis")
    lines.append("")
    lines.extend(build_failure_analysis(results))
    lines.append("")
    lines.append("## Case Details")
    lines.append("")

    for result in results:
        lines.append(f"### {result['id']} | {result.get('category', '')} | {result.get('difficulty', '')}")
        lines.append("")
        lines.append(f"- Question: {result.get('question', '')}")
        lines.append(f"- Intent: expected=`{result.get('expected_intent', '')}` predicted=`{result.get('intent', '')}`")
        lines.append(f"- Rewritten query: `{result.get('rewritten_query', '')}`")
        lines.append(f"- Queries: {' | '.join(result.get('queries', []))}")
        lines.append(f"- Coverage: {result.get('coverage', {}).get('status')}")
        lines.append(f"- Latency: {result.get('latency_sec', 0.0):.2f}s")
        lines.append(f"- Basic metrics: `{json.dumps(result.get('basic_metrics', {}), ensure_ascii=False)}`")
        if result.get("judge"):
            lines.append(f"- Judge: `{json.dumps(result['judge'], ensure_ascii=False)}`")
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append(result.get("answer", "")[:1200] or "(empty)")
        lines.append("")
        lines.append("Sources:")
        for source in result.get("sources", []):
            lines.append(
                f"- {source.get('type')} | {source.get('source')} | "
                f"page={source.get('page')} | score={source.get('score')} | query={source.get('query')}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-prompt", type=Path, default=DEFAULT_JUDGE_PROMPT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_RUNS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = load_vectorstore()
    results: list[dict[str, Any]] = []
    results_path = output_dir / "results.jsonl"

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[{index}/{len(questions)}] Running {item['id']}: {question}")

        start_time = time.time()
        rag_result = ask_rag(question, vectorstore)
        latency = time.time() - start_time

        normalized = normalize_rag_result(rag_result)
        formatted_sources = [format_source(source) for source in normalized.get("sources", [])]
        basic_metrics = calc_basic_metrics(item, normalized, formatted_sources)

        record: dict[str, Any] = {
            "id": item["id"],
            "category": item.get("category", ""),
            "difficulty": item.get("difficulty", ""),
            "question": question,
            "expected_intent": item.get("expected_intent", ""),
            "expected_keywords": item.get("expected_keywords", []),
            "should_use_wiki": item.get("should_use_wiki"),
            "answer": normalized.get("answer", ""),
            "sources": formatted_sources,
            "rewritten_query": normalized.get("rewritten_query", ""),
            "queries": normalized.get("queries", []),
            "intent": normalized.get("intent", ""),
            "intent_reason": normalized.get("intent_reason", ""),
            "coverage": normalized.get("coverage", {}),
            "fallback_used": normalized.get("fallback_used", False),
            "plan": normalized.get("plan", {}),
            "basic_metrics": basic_metrics,
            "latency_sec": round(latency, 3),
        }

        if args.judge:
            print("  judging with llm...")
            record["judge"] = judge_with_llm(item, record, args.judge_prompt)

        results.append(record)
        with results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_path = write_report(results, output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summarize_results(results), ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("Evaluation finished.")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
