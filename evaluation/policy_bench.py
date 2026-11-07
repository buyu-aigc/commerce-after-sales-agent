#!/usr/bin/env python3
"""Evaluate the demo after-sales policy pack against frozen golden cases."""
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
        actual = engine.evaluate(case["facts"])
        rows.append(
            {
                "id": case["id"],
                "group": case["group"],
                "expected_decision": case["expected_decision"],
                "actual_decision": actual["decision"],
                "expected_policy_id": case["expected_policy_id"],
                "actual_policy_id": actual["policy_id"],
                "passed": (
                    actual["decision"] == case["expected_decision"]
                    and actual["policy_id"] == case["expected_policy_id"]
                ),
                "facts": case["facts"],
            }
        )
    total = len(rows)
    passed = sum(row["passed"] for row in rows)
    expected_auto = sum(row["expected_decision"] == "auto" for row in rows)
    actual_auto = sum(row["actual_decision"] == "auto" for row in rows)
    false_auto = sum(
        row["expected_decision"] == "review" and row["actual_decision"] == "auto"
        for row in rows
    )
    return {
        "policy_pack": "policies/demo_after_sales.json",
        "demo_only": engine.pack.get("demo_only", False),
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total, 4) if total else 0,
        "expected_auto": expected_auto,
        "expected_auto_rate": round(expected_auto / total, 4) if total else 0,
        "actual_auto": actual_auto,
        "actual_auto_rate": round(actual_auto / total, 4) if total else 0,
        "false_auto": false_auto,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "policy_bench_report.json",
    )
    args = parser.parse_args()
    report = run(args.cases)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Demo Policy Golden Bench",
        "",
        f"- cases: {report['total']}",
        f"- accuracy: {report['accuracy']:.2%}",
        f"- expected auto: {report['expected_auto']} / {report['total']}",
        f"- actual auto: {report['actual_auto']} / {report['total']}",
        f"- false auto: {report['false_auto']}",
        f"- demo_only: {report['demo_only']}",
        "",
        "| case | expected | actual | policy | passed |",
        "|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['id']} | {row['expected_decision']} | {row['actual_decision']} | "
            f"{row['actual_policy_id']} | {row['passed']} |"
        )
    md_path = args.output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        key: report[key]
        for key in ("total", "passed", "accuracy", "actual_auto", "false_auto")
    }
    print(json.dumps(summary, ensure_ascii=False))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
