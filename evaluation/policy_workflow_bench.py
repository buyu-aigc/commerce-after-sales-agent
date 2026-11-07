#!/usr/bin/env python3
"""Evaluate workflow routing before/after policy_check on frozen cases.

This is a routing simulation over the real PolicyEngine, not 20 live browser
conversations. It assumes the no-policy baseline routes every write-intent
case to human review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.policy.engine import PolicyEngine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "policy_golden_cases.jsonl"


def run(cases_path: Path) -> dict:
    engine = PolicyEngine()
    rows = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        decision = engine.evaluate(case["facts"])
        routed = "auto" if decision["decision"] == "auto" else "review"
        rows.append(
            {
                "id": case["id"],
                "group": case["group"],
                "expected_decision": case["expected_decision"],
                "policy_id": decision["policy_id"],
                "routed": routed,
                "false_auto": case["expected_decision"] == "review" and routed == "auto",
            }
        )
    total = len(rows)
    auto = sum(row["routed"] == "auto" for row in rows)
    review = total - auto
    false_auto = sum(row["false_auto"] for row in rows)
    return {
        "total": total,
        "demo_only": engine.pack.get("demo_only", False),
        "before_policy_check": {
            "auto": 0,
            "review": total,
            "auto_rate": 0.0,
            "review_rate": 1.0,
        },
        "after_policy_check": {
            "auto": auto,
            "review": review,
            "auto_rate": round(auto / total, 4) if total else 0,
            "review_rate": round(review / total, 4) if total else 0,
        },
        "review_reduction_pp": round((auto / total) * 100, 2) if total else 0,
        "false_auto": false_auto,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "policy_workflow_report.json",
    )
    args = parser.parse_args()
    report = run(args.cases)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Policy Workflow Routing Bench",
        "",
        "> Routing simulation over the real PolicyEngine; not 20 live conversations.",
        "> Baseline assumes every write-intent case would otherwise go to human review.",
        "",
        f"- cases: {report['total']}",
        f"- before: auto {report['before_policy_check']['auto']} / review {report['before_policy_check']['review']}",
        f"- after: auto {report['after_policy_check']['auto']} / review {report['after_policy_check']['review']}",
        f"- auto rate: {report['before_policy_check']['auto_rate']:.2%} → {report['after_policy_check']['auto_rate']:.2%}",
        f"- review rate: {report['before_policy_check']['review_rate']:.2%} → {report['after_policy_check']['review_rate']:.2%}",
        f"- review reduction: {report['review_reduction_pp']:.2f} pp",
        f"- false auto: {report['false_auto']}",
        f"- demo_only: {report['demo_only']}",
        "",
        "| case | expected | routed | policy | false auto |",
        "|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        md.append(
            f"| {row['id']} | {row['expected_decision']} | {row['routed']} | "
            f"{row['policy_id']} | {row['false_auto']} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "total": report["total"],
                "auto": report["after_policy_check"]["auto"],
                "review": report["after_policy_check"]["review"],
                "review_reduction_pp": report["review_reduction_pp"],
                "false_auto": report["false_auto"],
            },
            ensure_ascii=False,
        )
    )
    if report["false_auto"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
