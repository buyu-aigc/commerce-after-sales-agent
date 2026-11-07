#!/usr/bin/env python3
"""Full-Olist policy replay + enriched full-graph sample.

The full Olist index has ~98.7k orders but no real support-policy labels.  This
script therefore does two separate things:

1. a full deterministic PolicyEngine replay over every order, assuming a
   uniform refund_request workload (no support labels exist in Olist);
2. an enriched full LangGraph sample that deliberately oversamples the 200
   auto-eligible rows so the auto path is actually exercised.

Results are proxy-label results, not production effect sizes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from app.policy.engine import PolicyEngine
from evaluation.policy_real_order_bench import run_scenario

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "olist_derived" / "order_facts_index.json"
DEFAULT_OUTPUT = ROOT / "reports" / "full_order_eval_report.json"

GROUP_AUTO = "auto_low_delay"


def classify(order: dict[str, Any]) -> str:
    status = str(order.get("status") or "")
    delay = int(order.get("delay_days") or 0)
    amount = float(order.get("payment_value") or 0)
    score = int(order.get("review_score") or 0)
    if status == "canceled":
        return "canceled"
    if status in {"shipped", "processing", "approved", "invoiced"}:
        return "in_transit"
    if status == "unavailable":
        return "unavailable"
    if status != "delivered":
        return "other"
    if score <= 2:
        return "low_score"
    if amount > 100:
        return "high_value"
    if delay < 10:
        return "short_delay"
    return GROUP_AUTO


def to_row(order: dict[str, Any]) -> dict[str, Any]:
    group = classify(order)
    products = order.get("products") or []
    category = str((products[0] or {}).get("category") or "") if products else ""
    return {
        "order_id": order["order_id"],
        "status": order.get("status"),
        "customer_state": order.get("customer_state") or "",
        "purchase_timestamp": order.get("purchase_timestamp") or "",
        "estimated_delivery_date": order.get("estimated_delivery_date") or "",
        "delivered_customer_date": order.get("delivered_customer_date") or "",
        "delay_days": order.get("delay_days"),
        "payment_value": float(order.get("payment_value") or 0),
        "review_score": order.get("review_score"),
        "category_summary": category,
        "refund_status": None,
        "address_change_status": None,
        "invoice_status": None,
        "fixture_group": group,
        "user_risk_level": "high" if int(order.get("review_score") or 0) <= 2 else "low",
        "repeat_claim": False,
        "refund_failure": False,
    }


def facts_for(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": "refund_request",
        "order_status": order.get("status"),
        "delay_days": int(order.get("delay_days") or 0),
        "amount": float(order.get("payment_value") or 0),
        "prior_compensation": False,
        "repeat_claim": False,
        "risk_level": "high" if int(order.get("review_score") or 0) <= 2 else "low",
        "refund_failure": False,
    }


def full_policy_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    engine = PolicyEngine()
    groups: Counter[str] = Counter()
    auto: Counter[str] = Counter()
    review: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    auto_outside = 0
    for row in rows:
        group = str(row["fixture_group"])
        groups[group] += 1
        decision = engine.evaluate(facts_for(row))
        hits[str(decision["policy_id"])] += 1
        if decision["decision"] == "auto":
            auto[group] += 1
            if group != GROUP_AUTO:
                auto_outside += 1
        else:
            review[group] += 1
    total = len(rows)
    auto_total = sum(auto.values())
    return {
        "total": total,
        "auto": auto_total,
        "review": total - auto_total,
        "auto_rate": round(auto_total / total, 6) if total else 0,
        "review_rate": round((total - auto_total) / total, 6) if total else 0,
        "by_group": {
            group: {"total": groups[group], "auto": auto[group], "review": review[group]}
            for group in sorted(groups)
        },
        "policy_hits": dict(sorted(hits.items())),
        "proxy_false_auto": auto_outside,
        "assumption": "uniform refund_request workload; Olist has no support-action labels",
    }


def enriched_sample(
    rows: list[dict[str, Any]], other_quota: int = 800, seed: int = 42
) -> list[dict[str, Any]]:
    auto_rows = [row for row in rows if row["fixture_group"] == GROUP_AUTO]
    other_rows = [row for row in rows if row["fixture_group"] != GROUP_AUTO]
    rng = random.Random(seed)
    sample = list(auto_rows)
    sample.extend(rng.sample(other_rows, min(other_quota, len(other_rows))))
    rng.shuffle(sample)
    return sample


def graph_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        without = await run_scenario(False, rows)
        with_policy = await run_scenario(True, rows)
        policy_auto = sum(
            1
            for row in with_policy["rows"]
            if row["group"] == GROUP_AUTO and row["status"] == "auto"
        )
        false_auto = sum(
            1
            for row in with_policy["rows"]
            if row["group"] != GROUP_AUTO and row["status"] == "auto"
        )
        return {
            "total": with_policy["total"],
            "without_policy": {
                "auto": without["auto"],
                "review": without["review"],
                "review_rate": without["review_rate"],
            },
            "with_policy": {
                "auto": with_policy["auto"],
                "review": with_policy["review"],
                "review_rate": with_policy["review_rate"],
            },
            "review_reduction_pp": round(
                (without["review_rate"] - with_policy["review_rate"]) * 100, 2
            ),
            "policy_auto": policy_auto,
            "false_auto": false_auto,
            "sample_composition": dict(Counter(row["fixture_group"] for row in rows)),
            "note": "enriched sample: all auto-eligible rows + random others; not natural distribution",
        }

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph-others", type=int, default=800)
    args = parser.parse_args()
    raw = json.loads(args.index.read_text(encoding="utf-8"))["orders"]
    rows = [to_row(order) for order in raw.values()]
    full = full_policy_replay(rows)
    sample = enriched_sample(rows, other_quota=args.graph_others)
    graph = graph_report(sample)
    report = {
        "source": "Olist Brazilian E-Commerce Public Dataset",
        "index": str(args.index),
        "total_orders": len(rows),
        "full_policy_replay": full,
        "graph_sample": graph,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Full Olist Policy Replay",
        "",
        f"- total orders: {len(rows):,}",
        "- full PolicyEngine: "
        f"auto {full['auto']:,} / review {full['review']:,} "
        f"({full['auto_rate']:.2%} auto)",
        f"- auto outside auto_candidate: {full['proxy_false_auto']}",
        f"- graph sample: {graph['total']:,} (enriched; not natural distribution)",
        "- graph without policy: "
        f"review {graph['without_policy']['review']:,} / {graph['total']:,} "
        f"({graph['without_policy']['review_rate']:.2%})",
        "- graph with policy: "
        f"review {graph['with_policy']['review']:,} / {graph['total']:,} "
        f"({graph['with_policy']['review_rate']:.2%})",
        f"- graph review reduction: {graph['review_reduction_pp']:.2f} pp",
        f"- graph policy auto: {graph['policy_auto']:,}",
        f"- graph false auto: {graph['false_auto']}",
        "",
        "## Full PolicyEngine by group",
        "",
        "| group | total | auto | review |",
        "|---|---:|---:|---:|",
    ]
    for group, counts in full["by_group"].items():
        md.append(f"| {group} | {counts['total']:,} | {counts['auto']:,} | {counts['review']:,} |")
    md.extend(
        [
            "",
            "> Olist has no support-action or risk labels. The action type is assumed to be",
            "> `refund_request` for every order; group labels are deterministic proxies.",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"full": full, "graph": graph}, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
