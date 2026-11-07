#!/usr/bin/env python3
"""Validate and summarize the author/expert-rubric label set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "evaluation" / "expert_labels_v1.jsonl"
DEFAULT_CASES = ROOT / "evaluation" / "complex_request_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "expert_labels_report.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(labels: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = {str(case["id"]) for case in cases}
    errors: list[str] = []
    label_case_ids = {str(label.get("case_id")) for label in labels}
    for case_id in sorted(case_ids - label_case_ids):
        errors.append(f"case_without_labels:{case_id}")
    for label in labels:
        if label.get("case_id") not in case_ids:
            errors.append(f"unknown_case:{label.get('case_id')}")
        if label.get("risk_level") == "high" and label.get("route") == "auto_execute":
            errors.append(f"high_risk_auto:{label.get('label_id')}")
        if label.get("label_source") != "author_expert_rubric_v1":
            errors.append(f"bad_source:{label.get('label_id')}")
        if not label.get("rationale"):
            errors.append(f"missing_rationale:{label.get('label_id')}")
    route_counts = Counter(str(label.get("route")) for label in labels)
    risk_counts = Counter(str(label.get("risk_level")) for label in labels)
    category_counts = Counter(str(label.get("case_category")) for label in labels)
    by_risk = {}
    for risk in sorted(risk_counts):
        subset = [label for label in labels if label.get("risk_level") == risk]
        by_risk[risk] = {
            "labels": len(subset),
            "auto_execute": sum(1 for label in subset if label.get("route") == "auto_execute"),
            "human_review": sum(1 for label in subset if label.get("route") == "human_review"),
            "clarify": sum(1 for label in subset if label.get("route") == "clarify"),
            "reject": sum(1 for label in subset if label.get("route") == "reject"),
        }
    return {
        "label_source": "author_expert_rubric_v1",
        "human_reviewed": False,
        "scope": (
            "24 controlled complex requests; 48 per-sub-case labels. Olist-style order facts "
            "are fixed synthetic fixtures; requests and labels are authored by the evaluator "
            "using the active policy rubric. This is not a human reviewer blind-label set."
        ),
        "total_cases": len(cases),
        "total_labels": len(labels),
        "cases_with_labels": len(label_case_ids),
        "route_counts": dict(route_counts),
        "risk_counts": dict(risk_counts),
        "category_counts": dict(category_counts),
        "by_risk": by_risk,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    labels = load_jsonl(args.labels)
    cases = load_jsonl(args.cases)
    report = build(labels, cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Expert-Rubric Label Set",
        "",
        "> **Author/expert-rubric labels, not human customer-service reviewer labels.**",
        "> Olist-style order facts are fixed fixtures; requests and routes are authored.",
        "",
        f"- cases: {report['total_cases']}",
        f"- per-sub-case labels: {report['total_labels']}",
        f"- label source: `{report['label_source']}`",
        f"- human reviewed: {report['human_reviewed']}",
        "",
        "## Route distribution",
        "",
        "| route | labels |",
        "|---|---:|",
    ]
    for route, count in report["route_counts"].items():
        md.append(f"| {route} | {count} |")
    md.extend(
        [
            "",
            "## By risk",
            "",
            "| risk | labels | auto | review | clarify | reject |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for risk, row in report["by_risk"].items():
        md.append(
            f"| {risk} | {row['labels']} | {row['auto_execute']} | {row['human_review']} | "
            f"{row['clarify']} | {row['reject']} |"
        )
    md.extend(["", "## Validation", "", f"- errors: {len(report['errors'])}"])
    for error in report["errors"]:
        md.append(f"- {error}")
    md.extend(
        [
            "",
            "## How to use",
            "",
            "- Use this as a reproducible rubric benchmark, not as evidence of human reviewer agreement.",
            "- To claim human review labels, fill `evaluation/expert_label_sheet.csv` with 3+ reviewers",
            "  and report agreement / adjudication.",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("total_cases", "total_labels", "route_counts", "errors")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
