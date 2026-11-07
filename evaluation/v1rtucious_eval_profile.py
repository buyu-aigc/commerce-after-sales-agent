from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "v1rtucious_derived" / "ecom_agent_eval_cases.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"V1rtucious ecom eval cases: {len(rows)}")
    print("response_type distribution:")
    for key, count in Counter(row["expected_response_type"] for row in rows).most_common():
        print(f"  {key}: {count}")
    print("intent_category distribution:")
    for key, count in Counter(row["expected_intent_category"] for row in rows).most_common():
        print(f"  {key}: {count}")
    print("group distribution:")
    for key, count in Counter(row["group"] for row in rows).most_common():
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
