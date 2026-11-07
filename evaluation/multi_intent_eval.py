from __future__ import annotations

import json
from pathlib import Path

from app.intent.decomposer import decompose_business_message

ROOT = Path(__file__).resolve().parents[1]
MULTI_CASES = ROOT / "data" / "bitext_derived" / "multi_intent_eval_cases.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    cases = load_jsonl(MULTI_CASES)
    exact = 0
    contains_all = 0
    side_effect_ok = 0
    failures = []
    for case in cases:
        tasks = decompose_business_message(case["message"])
        predicted = [task.intent for task in tasks]
        expected = case.get("expected_skills", case.get("expected_intents", []))
        expected_set = set(expected)
        predicted_set = set(predicted)
        exact_ok = predicted == expected
        contains_ok = expected_set.issubset(predicted_set)
        side_effect = any(task.side_effect for task in tasks)
        side_effect_match = side_effect == case["has_side_effect"]
        exact += int(exact_ok)
        contains_all += int(contains_ok)
        side_effect_ok += int(side_effect_match)
        if not (exact_ok and contains_ok and side_effect_match):
            failures.append((case["id"], expected, predicted))

    print(f"Multi-intent eval cases: {len(cases)}")
    print(f"Exact task list accuracy: {exact / len(cases):.2%}")
    print(f"Contains-all accuracy: {contains_all / len(cases):.2%}")
    print(f"Side-effect detection accuracy: {side_effect_ok / len(cases):.2%}")
    if failures:
        print("Sample failures:")
        for case_id, expected, predicted in failures[:5]:
            print(f"  {case_id}: expected={expected}, predicted={predicted}")


if __name__ == "__main__":
    main()
