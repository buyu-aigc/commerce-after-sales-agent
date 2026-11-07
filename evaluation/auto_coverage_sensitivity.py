#!/usr/bin/env python3
"""Auto-coverage sensitivity for the demo refund policy.

Olist has no support-request or risk labels.  This scans deterministic policy
thresholds over all orders and reports:

* auto coverage under a review-score risk guard;
* the low-score exposure if that guard is removed;
* incremental auto vs the current policy;
* an explicitly labelled review-time saving estimate.

It is an order-level sensitivity, not a production effect-size estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data" / "olist_derived" / "order_facts_index.json"
DEFAULT_OUTPUT = ROOT / "reports" / "auto_coverage_sensitivity_report.json"
DELAYS = (0, 5, 7, 10, 14)
AMOUNTS = (50, 100, 150, 200, 300)
CURRENT = (10, 100, True)


def load_orders(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["orders"].values())


def eligible(order: dict[str, Any], delay_min: int, amount_max: float, risk_guard: bool) -> bool:
    if str(order.get("status") or "") != "delivered":
        return False
    if int(order.get("delay_days") or 0) < delay_min:
        return False
    if float(order.get("payment_value") or 0) > amount_max:
        return False
    if risk_guard and int(order.get("review_score") or 0) <= 2:
        return False
    # Olist has no repeat/refund-failure/prior-compensation labels; treat them
    # as false.  Real labels would only reduce these counts.
    return True


def scan(orders: list[dict[str, Any]], delay_min: int, amount_max: float, risk_guard: bool) -> dict[str, Any]:
    auto = [order for order in orders if eligible(order, delay_min, amount_max, risk_guard)]
    low_score_auto = sum(1 for order in auto if int(order.get("review_score") or 0) <= 2)
    return {
        "delay_min": delay_min,
        "amount_max": amount_max,
        "risk_guard": risk_guard,
        "auto": len(auto),
        "review": len(orders) - len(auto),
        "auto_rate_of_all_orders": round(len(auto) / len(orders), 6),
        "low_score_auto": low_score_auto,
    }


def build(orders: list[dict[str, Any]], minutes_per_review: float = 3.0) -> dict[str, Any]:
    current = scan(orders, *CURRENT)
    grid = []
    for delay in DELAYS:
        for amount in AMOUNTS:
            for guard in (True, False):
                row = scan(orders, delay, amount, guard)
                row["delta_auto_vs_current"] = row["auto"] - current["auto"]
                row["review_hours_saved_vs_current"] = round(
                    row["delta_auto_vs_current"] * minutes_per_review / 60, 2
                )
                grid.append(row)
    return {
        "source": "Olist Brazilian E-Commerce Public Dataset (order-level proxy)",
        "total_orders": len(orders),
        "current_policy": {
            **current,
            "policy_id": "demo_delay_refund_auto_v1",
            "conditions": "delivered + delay>=10 + amount<=100 + review_score>=3 (proxy risk guard)",
        },
        "assumption": {
            "action_type": "refund_request for every order",
            "risk_guard": "review_score <= 2 treated as high risk",
            "missing_labels": "repeat_claim / refund_failure / prior_compensation assumed false",
            "minutes_per_review": minutes_per_review,
            "warning": "order-level sensitivity; not a request-level production effect size",
        },
        "grid": grid,
    }


def render(report: dict[str, Any]) -> str:
    current = report["current_policy"]
    lines = [
        "# Auto Coverage Sensitivity",
        "",
        f"- total orders: {report['total_orders']:,}",
        f"- current policy: auto {current['auto']:,} / review {current['review']:,} "
        f"({current['auto_rate_of_all_orders']:.2%} of all orders)",
        f"- current low-score auto: {current['low_score_auto']}",
        "",
        "> Proxy labels: Olist has no support-action labels; every order is assumed to be a",
        "> refund request, and review_score<=2 is used as a risk guard proxy.",
        "",
        "## Risk guard ON (review_score >= 3)",
        "",
        "| delay_min | amount_max | auto | auto rate | delta vs current |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["grid"]:
        if not row["risk_guard"]:
            continue
        lines.append(
            f"| {row['delay_min']} | {row['amount_max']} | {row['auto']:,} | "
            f"{row['auto_rate_of_all_orders']:.2%} | {row['delta_auto_vs_current']:+,} |"
        )
    lines.extend(
        [
            "",
            "## Risk guard OFF (no review-score filter)",
            "",
            "| delay_min | amount_max | auto | low-score auto | delta vs current |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["grid"]:
        if row["risk_guard"]:
            continue
        lines.append(
            f"| {row['delay_min']} | {row['amount_max']} | {row['auto']:,} | "
            f"{row['low_score_auto']:,} | {row['delta_auto_vs_current']:+,} |"
        )
    lines.extend(
        [
            "",
            "## Read this correctly",
            "",
            "- `auto rate` is relative to all 98,666 orders, not to actual support requests.",
            "- The low-score column is a proxy false-auto exposure; the current policy keeps it at 0.",
            "- Removing the risk guard increases coverage but may auto-process low-score orders.",
            "- Any expansion must be validated in shadow mode before enabling auto execution.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minutes-per-review", type=float, default=3.0)
    args = parser.parse_args()
    orders = load_orders(args.index)
    report = build(orders, minutes_per_review=args.minutes_per_review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(report) + "\n", encoding="utf-8")
    current = report["current_policy"]
    print(
        json.dumps(
            {
                "total_orders": report["total_orders"],
                "current_auto": current["auto"],
                "current_review": current["review"],
                "current_auto_rate": current["auto_rate_of_all_orders"],
                "grid_rows": len(report["grid"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
