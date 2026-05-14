import argparse
import csv
import json
import os
from http import HTTPStatus
from pathlib import Path
from typing import Any

from intent_router import route_intent
from prompts import GENERAL_QA_PROMPT
from rag import _call_generation, ask_rag, build_context, format_sources, load_vectorstore
from retrieval_executor import annotate_doc, deduplicate_docs


BASE_DIR = Path(__file__).parent
CASES_PATH = BASE_DIR / "eval" / "answer_quality_cases.jsonl"
OUTPUT_CSV = BASE_DIR / "eval" / "answer_quality_generated_review.csv"
OUTPUT_MD = BASE_DIR / "eval" / "answer_quality_generated_review.md"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def build_baseline_answer(question: str, retriever) -> dict[str, Any]:
    results = retriever.similarity_search_with_score(question, k=5)
    docs = [annotate_doc(doc, query=question, distance=distance) for doc, distance in results]
    docs = deduplicate_docs(docs)[:5]

    context = build_context(docs)
    prompt = GENERAL_QA_PROMPT.format(question=question, context=context)
    response = _call_generation(prompt, temperature=0, max_tokens=900)

    if response.status_code == HTTPStatus.OK:
        answer = response.output.text
    else:
        answer = f"Request failed: {response.code} - {response.message}"

    return {
        "answer": answer,
        "sources": format_sources(docs),
        "doc_count": len(docs),
        "queries": [question],
    }


def generate_review_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    os.chdir(BASE_DIR)
    retriever = load_vectorstore()
    rows = []

    for case in cases:
        question = case["question"]
        expected_intent = case["expected_intent"]
        review_focus = case["review_focus"]

        baseline = build_baseline_answer(question, retriever)
        planned = ask_rag(question, retriever)
        routed_intent = route_intent(question)

        rows.append(
            {
                "question": question,
                "expected_intent": expected_intent,
                "predicted_intent": routed_intent.intent,
                "review_focus": review_focus,
                "baseline_doc_count": baseline["doc_count"],
                "planned_doc_count": planned.get("doc_count", 0),
                "baseline_queries": " | ".join(baseline["queries"]),
                "planned_queries": " | ".join(planned.get("queries", [])),
                "planned_coverage": planned.get("coverage", {}).get("status", ""),
                "baseline_answer": baseline["answer"].replace("\n", "\\n"),
                "planned_answer": planned["answer"].replace("\n", "\\n"),
                "manual_winner": "",
                "notes": "",
            }
        )

        write_csv(rows, OUTPUT_CSV)
        write_markdown(rows, OUTPUT_MD)

    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Generated Answer Review Draft", ""]

    for index, row in enumerate(rows, start=1):
        lines.append(f"## Case {index}")
        lines.append(f"- Question: {row['question']}")
        lines.append(f"- Expected intent: {row['expected_intent']}")
        lines.append(f"- Predicted intent: {row['predicted_intent']}")
        lines.append(f"- Review focus: {row['review_focus']}")
        lines.append(f"- Baseline doc count: {row['baseline_doc_count']}")
        lines.append(f"- Planned doc count: {row['planned_doc_count']}")
        lines.append(f"- Planned coverage: {row['planned_coverage']}")
        lines.append(f"- Baseline queries: {row['baseline_queries']}")
        lines.append(f"- Planned queries: {row['planned_queries']}")
        lines.append("")
        lines.append("### Baseline Answer")
        lines.append("")
        lines.append(row["baseline_answer"].replace("\\n", "\n"))
        lines.append("")
        lines.append("### Retrieval Planner Answer")
        lines.append("")
        lines.append(row["planned_answer"].replace("\\n", "\n"))
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only generate the first N review cases.")
    args = parser.parse_args()

    cases = load_cases(CASES_PATH)
    if args.limit is not None:
        cases = cases[: args.limit]

    rows = generate_review_rows(cases)
    write_csv(rows, OUTPUT_CSV)
    write_markdown(rows, OUTPUT_MD)
    print(f"Wrote review CSV to: {OUTPUT_CSV}")
    print(f"Wrote review Markdown to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
