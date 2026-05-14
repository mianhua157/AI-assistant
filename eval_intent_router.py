import json
from collections import Counter, defaultdict
from pathlib import Path

from intent_router import route_intent


DATASET_PATH = Path(__file__).parent / "eval" / "intent_eval.jsonl"


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def evaluate(cases: list[dict]) -> dict:
    total = len(cases)
    correct = 0
    per_intent_total = Counter()
    per_intent_correct = Counter()
    mismatches = []

    for case in cases:
        expected = case["expected_intent"]
        predicted = route_intent(case["question"])

        per_intent_total[expected] += 1
        if predicted.intent == expected:
            correct += 1
            per_intent_correct[expected] += 1
        else:
            mismatches.append(
                {
                    "question": case["question"],
                    "expected": expected,
                    "predicted": predicted.intent,
                    "reason": predicted.reason,
                }
            )

    per_intent_accuracy = {}
    for intent, count in per_intent_total.items():
        per_intent_accuracy[intent] = round(per_intent_correct[intent] / count, 3)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "per_intent_accuracy": per_intent_accuracy,
        "mismatches": mismatches,
    }


def main() -> None:
    cases = load_cases(DATASET_PATH)
    result = evaluate(cases)

    print("Intent Router Evaluation")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total cases: {result['total']}")
    print(f"Correct: {result['correct']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print("")
    print("Per-intent accuracy:")
    for intent in sorted(result["per_intent_accuracy"]):
        print(f"- {intent}: {result['per_intent_accuracy'][intent]:.3f}")

    if result["mismatches"]:
        print("")
        print("Mismatches:")
        for mismatch in result["mismatches"]:
            print(
                f"- Q: {mismatch['question']} | "
                f"expected={mismatch['expected']} | "
                f"predicted={mismatch['predicted']} | "
                f"reason={mismatch['reason']}"
            )
    else:
        print("")
        print("No mismatches found.")


if __name__ == "__main__":
    main()
