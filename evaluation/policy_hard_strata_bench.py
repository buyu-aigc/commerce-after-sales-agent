#!/usr/bin/env python3
"""Evaluate hard after-sales strata against the demo policy pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.policy.engine import PolicyEngine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "policy_hard_strata_report.json"


def cases() -> list[dict]:
    base = {
        "action_type": "refund_request",
        "order_status": "delivered",
        "delay_days": 11,
        "amount": 43.0,
        "prior_compensation": False,
        "repeat_claim": False,
        "risk_level": "low",
        "refund_failure": False,
    }
    rows = []
    for index in range(100):
        rows.append({"id": f"control-{index}", "stratum": "control", "facts": dict(base)})
    for index in range(100):
        facts = dict(base, repeat_claim=True)
        rows.append({"id": f"repeat-{index}", "stratum": "repeat_claim", "facts": facts})
    for index in range(100):
        facts = dict(base, risk_level="high")
        rows.append({"id": f"high-risk-{index}", "stratum": "high_risk_user", "facts": facts})
    for index in range(100):
        facts = dict(base, refund_failure=True)
        rows.append({"id": f"refund-failure-{index}", "stratum": "refund_failure", "facts": facts})
    return rows


def run() -> dict:
    engine = PolicyEngine()
    rows = []
    for case in cases():
        decision = engine.evaluate(case["facts"])
        rows.append(
            {
                **case,
                "decision": decision["decision"],
                "policy_id": decision["policy_id"],
                "false_auto": case["stratum"] != "control" and decision["decision"] == "auto",
            }
        )
    total = len(rows)
    auto = sum(row["decision"] == "auto" for row in rows)
    false_auto = sum(row["false_auto"] for row in rows)
    by_stratum = {}
    for stratum in ("control", "repeat_claim", "high_risk_user", "refund_failure"):
        subset = [row for row in rows if row["stratum"] == stratum]
        by_stratum[stratum] = {
            "total": len(subset),
            "auto": sum(row["decision"] == "auto" for row in subset),
            "review": sum(row["decision"] == "review" for row in subset),
        }
    return {
        "demo_only": True,
        "total": total,
        "auto": auto,
        "review": total - auto,
        "false_auto": false_auto,
        "by_stratum": by_stratum,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Policy Hard Strata Bench",
        "",
        "> 400 cases: real-order-shaped facts plus synthetic risk/refund-failure labels.",
        f"- total: {report['total']}",
        f"- auto: {report['auto']}",
        f"- review: {report['review']}",
        f"- false auto: {report['false_auto']}",
        "",
        "| stratum | total | auto | review |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["by_stratum"].items():
        md.append(f"| {name} | {row['total']} | {row['auto']} | {row['review']} |")
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("total", "auto", "review", "false_auto")}, ensure_ascii=False
        )
    )
    if report["false_auto"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
