from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retrieval.hybrid import HybridSupportRetriever

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "rescommons_derived" / "hybrid_retrieval_eval_cases.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25, vector, and hybrid retrieval.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("bm25", "local_vector_store", "hybrid_rerank"),
        default=("bm25", "local_vector_store", "hybrid_rerank"),
    )
    args = parser.parse_args()
    retriever = HybridSupportRetriever()
    cases = load_jsonl(CASES)[:100]
    print(f"Hybrid retrieval eval cases: {len(cases)}")
    print("strategy,cases,intent@1,intent@5,intent_mrr@5,capability@1,capability@5,capability_mrr@5")
    all_strategies = {
        "bm25": retriever.bm25_search,
        "local_vector_store": retriever.vector_search,
        "hybrid_rerank": retriever.hybrid_search,
    }
    for name in args.strategies:
        method = all_strategies[name]
        intent_at_1 = 0
        intent_at_5 = 0
        intent_mrr = 0.0
        capability_at_1 = 0
        capability_at_5 = 0
        capability_mrr = 0.0
        for case in cases:
            hits = method(case["query"], k=5)
            top_intents = [hit.doc.get("intent") for hit in hits]
            top_capabilities = [hit.doc.get("capability") for hit in hits]
            intent_at_1 += int(bool(top_intents) and top_intents[0] == case["intent"])
            intent_at_5 += int(case["intent"] in top_intents)
            intent_mrr += _reciprocal_rank(top_intents, case["intent"])
            capability_at_1 += int(
                bool(top_capabilities) and top_capabilities[0] == case["capability"]
            )
            capability_at_5 += int(case["capability"] in top_capabilities)
            capability_mrr += _reciprocal_rank(top_capabilities, case["capability"])
        n = len(cases)
        print(
            f"{name},{n},{intent_at_1 / n:.2%},{intent_at_5 / n:.2%},"
            f"{intent_mrr / n:.2%},{capability_at_1 / n:.2%},"
            f"{capability_at_5 / n:.2%},{capability_mrr / n:.2%}"
        )


def _reciprocal_rank(values: list[str | None], expected: str) -> float:
    for rank, value in enumerate(values, start=1):
        if value == expected:
            return 1.0 / rank
    return 0.0


if __name__ == "__main__":
    main()
