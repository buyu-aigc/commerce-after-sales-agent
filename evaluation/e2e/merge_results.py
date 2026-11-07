"""Merge non-overlapping E2E live-result batches into one reviewed artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.e2e.after_sales_e2e_bench import _stratum_summaries, _summary


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge(paths: list[Path]) -> dict[str, Any]:
    payloads = [_load(path) for path in paths]
    if not payloads:
        raise ValueError("At least one result artifact is required.")
    first = payloads[0]
    keys = ("protocol", "model", "split")
    for payload in payloads[1:]:
        for key in keys:
            if payload.get(key) != first.get(key):
                raise ValueError(f"Cannot merge results with different {key} values.")

    rows: dict[str, list[dict[str, Any]]] = {
        "bare_function_calling": [],
        "langgraph_workflow": [],
    }
    seen: set[str] = set()
    for payload in payloads:
        for variant, variant_rows in payload.get("rows", {}).items():
            if variant not in rows:
                raise ValueError(f"Unexpected variant: {variant}")
            if not isinstance(variant_rows, list):
                raise ValueError(f"Invalid rows for {variant}")
            if variant == "bare_function_calling":
                for row in variant_rows:
                    case_id = str(row.get("case_id", ""))
                    if not case_id or case_id in seen:
                        raise ValueError(f"Duplicate or missing case_id: {case_id}")
                    seen.add(case_id)
            rows[variant].extend(variant_rows)

    return {
        "protocol": first["protocol"],
        "scope": first["scope"],
        "model": first["model"],
        "split": first["split"],
        "case_count": len(seen),
        "source_batches": [str(path) for path in paths],
        "bare_function_calling": _summary(rows["bare_function_calling"], "bare_function_calling"),
        "langgraph_workflow": _summary(rows["langgraph_workflow"], "langgraph_workflow"),
        "by_stratum": {
            "bare_function_calling": _stratum_summaries(
                rows["bare_function_calling"], "bare_function_calling"
            ),
            "langgraph_workflow": _stratum_summaries(
                rows["langgraph_workflow"], "langgraph_workflow"
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
