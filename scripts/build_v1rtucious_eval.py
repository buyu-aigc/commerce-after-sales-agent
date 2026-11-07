from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "v1rtucious_raw" / "test.parquet"
OUT = ROOT / "data" / "v1rtucious_derived"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(RAW).to_pylist()
    cases = []
    groups = Counter()
    response_types = Counter()
    intents = Counter()
    for row in rows:
        groups[row.get("group", "")] += 1
        response_types[row.get("response_type", "")] += 1
        intents[row.get("intent", "")] += 1
        cases.append(
            {
                "id": row["id"],
                "group": row.get("group", ""),
                "difficulty": row.get("difficulty"),
                "query": row.get("prompt", ""),
                "context": row.get("context", ""),
                "tools": row.get("tools", ""),
                "expected_response_type": row.get("response_type", ""),
                "expected_intent_category": row.get("intent_category", ""),
                "expected_intent": row.get("intent", ""),
                "expected_sub_intent": row.get("sub_intent", ""),
            }
        )
    _write_jsonl(OUT / "ecom_agent_eval_cases.jsonl", cases)
    metadata = {
        "source": "V1rtucious/Ecom-Chatbot-Test-Set",
        "source_url": "https://huggingface.co/datasets/V1rtucious/Ecom-Chatbot-Test-Set",
        "license": "MIT",
        "case_count": len(cases),
        "group_counts": groups.most_common(),
        "response_type_counts": response_types.most_common(),
        "top_intents": intents.most_common(20),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'ecom_agent_eval_cases.jsonl'} ({len(cases)} cases)")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
