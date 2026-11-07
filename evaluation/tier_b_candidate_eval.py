#!/usr/bin/env python3
"""Compare current Tier A with the Tier B candidate on all Olist orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.policy.engine import PolicyEngine
from evaluation.full_order_eval import INDEX, facts_for, to_row

ROOT = Path(__file__).resolve().parents[1]


def load_orders(path: Path) -> list[dict]:
    return list(json.loads(path.read_text(encoding="utf-8"))["orders"].values())
DEFAULT_CANDIDATE = ROOT / "policies" / "candidates" / "demo_after_sales_tier_b.json"
DEFAULT_OUTPUT = ROOT / "reports" / "tier_b_candidate_report.json"


def summarize(engine: PolicyEngine, rows: list[dict]) -> dict:
    auto = 0
    low_score_auto = 0
    by_group: dict[str, dict[str, int]] = {}
    for row, order in rows:
        decision = engine.evaluate(facts_for(order))
        group = str(row["fixture_group"])
        bucket = by_group.setdefault(group, {"total": 0, "auto": 0, "review": 0})
        bucket["total"] += 1
        if decision["decision"] == "auto":
            auto += 1
            bucket["auto"] += 1
            if int(order.get("review_score") or 0) <= 2:
                low_score_auto += 1
        else:
            bucket["review"] += 1
    return {
        "version": engine.version,
        "auto": auto,
        "review": len(rows) - auto,
        "auto_rate": round(auto / len(rows), 6),
        "low_score_auto": low_score_auto,
        "by_group": by_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    orders = load_orders(args.index)
    rows = [(to_row(order), order) for order in orders]
    current = summarize(PolicyEngine(), rows)
    candidate = summarize(PolicyEngine(args.candidate), rows)
    report = {
        "total_orders": len(rows),
        "current": current,
        "candidate": candidate,
        "delta_auto": candidate["auto"] - current["auto"],
        "delta_low_score_auto": candidate["low_score_auto"] - current["low_score_auto"],
        "assumption": "order-level refund_request workload; review_score<=2 is the risk proxy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Tier B Candidate Evaluation",
        "",
        f"- total orders: {len(rows):,}",
        f"- current {current['version']}: auto {current['auto']:,} / review {current['review']:,} "
        f"({current['auto_rate']:.2%} auto), low-score auto {current['low_score_auto']}",
        f"- candidate {candidate['version']}: auto {candidate['auto']:,} / review {candidate['review']:,} "
        f"({candidate['auto_rate']:.2%} auto), low-score auto {candidate['low_score_auto']}",
        f"- delta auto: {report['delta_auto']:+,}",
        f"- delta low-score auto: {report['delta_low_score_auto']:+,}",
        "",
        "> Order-level proxy; not a request-level production effect size.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "current": current["auto"],
                "candidate": candidate["auto"],
                "delta": report["delta_auto"],
                "low_score_auto": candidate["low_score_auto"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
