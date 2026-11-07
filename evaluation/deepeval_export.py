from __future__ import annotations

import json
from pathlib import Path

from evaluation.rag_retrieval_eval import build_cases

OUT = Path(__file__).resolve().parents[1] / "data" / "olist_derived" / "deepeval_rag_cases.jsonl"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for case in build_cases():
            f.write(
                json.dumps(
                    {
                        "input": case.query,
                        "expected_output": case.expected_category,
                        "retrieval_context": [case.expected_category],
                        "metadata": {
                            "case_id": case.case_id,
                            "variant": case.variant,
                            "metric_hint": "contextual_recall/contextual_precision/tool_correctness",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
