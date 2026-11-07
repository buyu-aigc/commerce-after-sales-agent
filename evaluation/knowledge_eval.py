from __future__ import annotations

import json
from pathlib import Path

from app.olist.knowledge import MarkdownKnowledgeBase

CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge_base" / "policy_eval_cases.jsonl"


def load_cases() -> list[dict[str, str]]:
    with CASES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    kb = MarkdownKnowledgeBase()
    cases = load_cases()
    top1 = 0
    recall3 = 0
    reciprocal_rank = 0.0
    failures: list[str] = []

    for case in cases:
        hits = kb.search(case["query"], k=3)
        ranked = [hit.section_title for hit in hits]
        expected = case["expected_section"]
        if ranked and ranked[0] == expected:
            top1 += 1
        if expected in ranked:
            recall3 += 1
            reciprocal_rank += 1 / (ranked.index(expected) + 1)
        else:
            failures.append(case["id"])

    total = len(cases)
    print(f"Knowledge eval cases: {total}")
    print(f"Top1: {top1 / total:.2%}" if total else "Top1: n/a")
    print(f"Recall@3: {recall3 / total:.2%}" if total else "Recall@3: n/a")
    print(f"MRR@3: {reciprocal_rank / total:.2%}" if total else "MRR@3: n/a")
    if failures:
        print("Failures:")
        for failure in failures[:20]:
            print(f"  {failure}")


if __name__ == "__main__":
    main()
