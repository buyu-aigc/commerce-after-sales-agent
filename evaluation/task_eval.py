import json
from pathlib import Path

from app.olist.service import OlistService

EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "olist_derived" / "eval_cases.jsonl"


def load_cases() -> list[dict]:
    with EVAL_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    service = OlistService()
    cases = load_cases()
    total = 0
    passed = 0
    failures: list[str] = []

    for case in cases:
        expected = case["expected"]
        expected_intent = case.get("expected_intent", case.get("expected_skill"))
        if expected_intent == "order_status":
            total += 1
            order = service.get_order_status(expected["order_id"])
            ok = order is not None and order.status == expected["status"]
        elif expected_intent == "escalation":
            total += 1
            draft = service.escalation_draft(expected["order_id"])
            ok = draft is not None
            if "delay_days" in expected:
                order = service.get_order_status(expected["order_id"])
                ok = ok and order is not None and order.delay_days == expected["delay_days"]
            if "review_score" in expected:
                order = service.get_order_status(expected["order_id"])
                ok = ok and order is not None and order.review_score == expected["review_score"]
        elif expected_intent == "qa":
            total += 1
            insights = service.category_insights(expected["category"])
            ok = any(item["name"] == expected["category"] for item in insights)
        else:
            continue

        if ok:
            passed += 1
        else:
            failures.append(case["id"])

    print(f"Task eval cases: {total}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {passed / total:.2%}" if total else "Accuracy: n/a")
    if failures:
        print("Failures:")
        for failure in failures[:20]:
            print(f"  {failure}")


if __name__ == "__main__":
    main()
