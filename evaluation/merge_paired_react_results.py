"""Merge disjoint paired-evaluation batches without making model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.after_sales_react_comparison import _summary


def merge(inputs: list[Path], output: Path) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if not payloads:
        raise ValueError("At least one input is required.")
    models = {str(payload.get("model")) for payload in payloads}
    suites = {str(payload.get("suite")) for payload in payloads}
    if len(models) != 1 or len(suites) != 1:
        raise ValueError("All batches must use the same model and suite.")

    react_rows = [row for payload in payloads for row in payload["react_rows"]]
    workflow_rows = [row for payload in payloads for row in payload["full_workflow_rows"]]
    react = _summary(react_rows, "react_function_calling")
    workflow = _summary(workflow_rows, "langgraph_after_sales_workflow")
    fields = (
        "gold_action_path_success",
        "required_task_coverage",
        "task_plan_exact_match",
        "disposition_accuracy",
        "action_type_accuracy",
        "state_projection_accuracy",
        "extra_task_rate",
        "unsafe_write_rate",
    )
    result = {
        "scope": "Curated Olist after-sales paired same-model live tool-loop ablation.",
        "suite": next(iter(suites)),
        "model": next(iter(models)),
        "case_count": len(react_rows),
        "react_function_calling": react,
        "langgraph_after_sales_workflow": workflow,
        "mean_react_tool_calls": round(
            sum(int(row["tool_call_count"]) for row in react_rows) / len(react_rows),
            2,
        ),
        "delta_percentage_points": {
            field: round((float(workflow[field]) - float(react[field])) * 100, 2)
            for field in fields
        },
        "source_batches": [str(path) for path in inputs],
        "react_rows": react_rows,
        "full_workflow_rows": workflow_rows,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = merge(args.inputs, args.output)
    compact = {key: value for key, value in summary.items() if not key.endswith("rows")}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
