from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_RESULTS = ROOT / "evaluation" / "live_agent_eval_results.jsonl"
OUT = ROOT / "evaluation" / "route_drift_report.md"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    rows = load_jsonl(LIVE_RESULTS)
    if not rows:
        report = "# Route Drift Report\n\nNo live eval result found."
        OUT.write_text(report, encoding="utf-8")
        print(report)
        return

    expected_first = Counter(_first(row.get("expected_tasks", [])) for row in rows)
    actual_first = Counter(_first(row.get("actual_tasks", [])) for row in rows)
    expected_sequence = Counter(",".join(row.get("expected_tasks", [])) for row in rows)
    actual_sequence = Counter(",".join(row.get("actual_tasks", [])) for row in rows)
    mismatches = [
        row
        for row in rows
        if row.get("expected_tasks") != row.get("actual_tasks")
        or not row.get("checks", {}).get("task_exact", False)
    ]

    report = _render_report(
        rows,
        expected_first,
        actual_first,
        expected_sequence,
        actual_sequence,
        mismatches,
    )
    OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {OUT}")


def _first(values: list[str]) -> str:
    return values[0] if values else "none"


def _render_report(
    rows: list[dict],
    expected_first: Counter[str],
    actual_first: Counter[str],
    expected_sequence: Counter[str],
    actual_sequence: Counter[str],
    mismatches: list[dict],
) -> str:
    model = rows[0].get("model", "unknown")
    prompt_version = rows[0].get("prompt_version", "unknown")
    lines = [
        "# Route Drift Report",
        "",
        "This report compares the live LLM planner output against the pinned eval expectation.",
        f"model: `{model}`; prompt_version: `{prompt_version}`; cases: `{len(rows)}`.",
        "",
        "## First-Intent Distribution",
        "",
        "| intent | expected | actual | delta |",
        "|---|---:|---:|---:|",
    ]
    for intent in sorted(set(expected_first) | set(actual_first)):
        expected = expected_first[intent]
        actual = actual_first[intent]
        lines.append(f"| {intent} | {expected} | {actual} | {actual - expected:+d} |")

    lines.extend(
        [
            "",
            "## Task-Sequence Distribution",
            "",
            "| task_sequence | expected | actual | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for sequence in sorted(set(expected_sequence) | set(actual_sequence)):
        expected = expected_sequence[sequence]
        actual = actual_sequence[sequence]
        lines.append(f"| {sequence or 'none'} | {expected} | {actual} | {actual - expected:+d} |")

    lines.extend(
        [
            "",
            "## Mismatches",
            "",
            f"task_exact_mismatch_rate: `{len(mismatches) / len(rows):.2%}`.",
        ]
    )
    if mismatches:
        lines.extend(["", "| case | expected | actual | failed_checks |", "|---|---|---|---|"])
        for row in mismatches[:20]:
            failed = [name for name, ok in row.get("checks", {}).items() if not ok]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("case", "")),
                        ",".join(row.get("expected_tasks", [])),
                        ",".join(row.get("actual_tasks", [])),
                        ",".join(failed) or "-",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
