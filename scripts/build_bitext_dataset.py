from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.intent.mapping import INTENT_TO_ROUTE, map_intent  # noqa: E402

RAW = ROOT / "data" / "bitext_raw" / "bitext_customer_support.csv"
OUT = ROOT / "data" / "bitext_derived"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RAW.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    by_intent: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_intent.setdefault(row["intent"], []).append(row)

    eval_cases = []
    for intent, intent_rows in sorted(by_intent.items()):
        mapping = map_intent(intent)
        for row in intent_rows[:40]:
            eval_cases.append(
                {
                    "id": f"bitext-{intent}-{len(eval_cases)}",
                    "source": "Bitext customer support dataset",
                    "message": row["instruction"],
                    "category": row["category"],
                    "intent": intent,
                    "expected_intent": mapping.route_intent,
                    "needs_order_id": mapping.needs_order_id,
                    "side_effect_risk": mapping.side_effect_risk,
                }
            )

    multi_intent_cases = build_multi_intent_cases(eval_cases)
    dataset = {
        "metadata": {
            "source": "Bitext customer support LLM chatbot training dataset",
            "source_url": "https://github.com/bitext/customer-support-llm-chatbot-training-dataset",
            "raw_rows": len(rows),
            "intent_count": len(by_intent),
            "category_counts": Counter(row["category"] for row in rows).most_common(),
            "supported_intents": sorted(INTENT_TO_ROUTE),
        },
        "eval_case_count": len(eval_cases),
        "multi_intent_case_count": len(multi_intent_cases),
    }
    (OUT / "metadata.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(OUT / "intent_eval_cases.jsonl", eval_cases)
    _write_jsonl(OUT / "multi_intent_eval_cases.jsonl", multi_intent_cases)
    print(f"Wrote {OUT / 'intent_eval_cases.jsonl'} ({len(eval_cases)} cases)")
    print(f"Wrote {OUT / 'multi_intent_eval_cases.jsonl'} ({len(multi_intent_cases)} cases)")


def build_multi_intent_cases(cases: list[dict]) -> list[dict]:
    by_intent = {}
    for case in cases:
        by_intent.setdefault(case["expected_intent"], []).append(case)
    pairs = [
        ("order_status", "policy"),
        ("order_status", "escalation"),
        ("policy", "escalation"),
    ]
    combined = []
    for first_intent, second_intent in pairs:
        left = by_intent.get(first_intent, [])[:20]
        right = by_intent.get(second_intent, [])[:20]
        for i, (a, b) in enumerate(zip(left, right, strict=False)):
            combined.append(
                {
                    "id": f"multi-{first_intent}-{second_intent}-{i}",
                    "message": f"{a['message']}，另外{b['message']}",
                    "expected_intents": [first_intent, second_intent],
                    "source_intents": [a["intent"], b["intent"]],
                    "has_side_effect": "escalation" in {first_intent, second_intent},
                }
            )
    return combined


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
