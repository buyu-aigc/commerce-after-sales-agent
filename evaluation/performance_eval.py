from __future__ import annotations

import statistics
import time
from pathlib import Path

from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import OlistService

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "olist_derived"
ORDER_ID = "203096f03d82e0dffbc41ebc2e2bcfb7"


def main() -> None:
    service = OlistService()
    kb = MarkdownKnowledgeBase()

    workloads = {
        "order_status_lookup": lambda: service.get_order_status(ORDER_ID),
        "category_risk_retrieval": lambda: service.category_insights("health beauty category risk"),
        "policy_kb_retrieval": lambda: kb.search("退款补偿需要人工确认吗", k=3),
        "escalation_draft": lambda: service.escalation_draft(ORDER_ID),
    }

    print("Performance eval: deterministic tool layer")
    print("workload,runs,p50_ms,p95_ms,avg_ms,qps")
    for name, fn in workloads.items():
        latencies = _measure(fn)
        avg = statistics.mean(latencies)
        print(
            f"{name},{len(latencies)},"
            f"{_percentile(latencies, 50):.3f},"
            f"{_percentile(latencies, 95):.3f},"
            f"{avg:.3f},"
            f"{1000 / avg:.1f}"
        )

    print("\nData footprint")
    for path in sorted(DERIVED.glob("*.json*")):
        print(f"{path.name},{path.stat().st_size / 1024 / 1024:.2f} MiB")

    print("\nApproximate prompt input tokens")
    print(f"route_eval_cases,{_approx_tokens((DERIVED / 'eval_cases.jsonl').read_text(encoding='utf-8'))}")
    policy_text = (ROOT / "data" / "knowledge_base" / "support_policy.md").read_text(encoding="utf-8")
    print(f"support_policy,{_approx_tokens(policy_text)}")


def _measure(fn, warmup: int = 20, runs: int = 500) -> list[float]:
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def _percentile(values: list[float], percentile: int) -> float:
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * percentile / 100)
    return sorted_values[index]


def _approx_tokens(text: str) -> int:
    # Rough multilingual heuristic for budgeting only; provider billing should use returned usage.
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return round(ascii_chars / 4 + non_ascii_chars * 1.5)


if __name__ == "__main__":
    main()
