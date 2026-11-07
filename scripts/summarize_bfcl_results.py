"""Summarize BFCL score files.

BFCL writes category score JSON files and leaderboard CSV files under score/.
This script intentionally stays schema-tolerant because BFCL category files vary
across versions. It extracts common accuracy fields and writes a compact report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmark_runs" / "bfcl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize BFCL score outputs.")
    parser.add_argument("--score-root", required=True, help="BFCL score/ directory or one score JSON/CSV.")
    parser.add_argument("--model", default="", help="Optional model name filter.")
    parser.add_argument("--out", default=str(OUT_DIR / "last_summary.md"))
    args = parser.parse_args()

    score_root = Path(args.score_root).expanduser().resolve()
    summary = summarize_score_root(score_root, args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(score_root, summary), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(render_markdown(score_root, summary))
    print(f"\nWrote {out}")
    return 0


def summarize_score_root(score_root: Path, model_filter: str = "") -> dict[str, Any]:
    json_rows = []
    csv_rows = []
    paths = _score_paths(score_root)
    for path in paths:
        if path.suffix.lower() == ".json":
            json_rows.extend(_read_json_score(path))
        elif path.suffix.lower() == ".csv":
            csv_rows.extend(_read_csv_score(path, model_filter))
    return {
        "json_score_count": len(json_rows),
        "csv_row_count": len(csv_rows),
        "category_scores": json_rows[:100],
        "leaderboard_rows": csv_rows[:20],
    }


def _score_paths(score_root: Path) -> list[Path]:
    if score_root.is_file():
        return [score_root]
    if not score_root.exists():
        return []
    return sorted(
        {
            *score_root.rglob("*_score.json"),
            *score_root.glob("data_*.csv"),
            *score_root.rglob("data_*.csv"),
        }
    )


def _read_json_score(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    else:
        items = [data]
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        accuracy = _first_number(
            item,
            [
                "accuracy",
                "acc",
                "overall_acc",
                "overall_accuracy",
                "score",
                "pass_rate",
            ],
        )
        rows.append(
            {
                "file": str(path),
                "category": item.get("test_category") or item.get("category") or path.stem,
                "accuracy": accuracy,
                "total": item.get("total") or item.get("num_total") or item.get("count"),
                "correct": item.get("correct") or item.get("num_correct"),
            }
        )
    return rows


def _read_csv_score(path: Path, model_filter: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("Model") or row.get("model") or row.get("Model Name") or ""
            if model_filter and model_filter not in model:
                continue
            rows.append(
                {
                    "file": str(path),
                    "model": model,
                    "overall_acc": _row_value(row, ["Overall Acc", "overall_acc", "Accuracy"]),
                    "cost": _row_value(row, ["Cost ($)", "cost"]),
                    "latency_p95": _row_value(row, ["P95", "Latency P95", "p95"]),
                }
            )
    return rows


def _first_number(item: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip("%")) / (100 if "%" in value else 1)
            except ValueError:
                continue
    return None


def _row_value(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    return ""


def render_markdown(score_root: Path, summary: dict[str, Any]) -> str:
    lines = [
        "# BFCL Benchmark Summary",
        "",
        f"score_root: `{score_root}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| json_score_count | {summary['json_score_count']} |",
        f"| csv_row_count | {summary['csv_row_count']} |",
    ]
    if summary["category_scores"]:
        lines.extend(
            [
                "",
                "## Category Scores",
                "",
                "| category | accuracy | total | correct |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in summary["category_scores"]:
            accuracy = row["accuracy"]
            accuracy_text = "n/a" if accuracy is None else f"{accuracy:.2%}"
            lines.append(
                f"| {row['category']} | {accuracy_text} | "
                f"{row['total'] or ''} | {row['correct'] or ''} |"
            )
    if summary["leaderboard_rows"]:
        lines.extend(
            [
                "",
                "## Leaderboard Rows",
                "",
                "| model | overall_acc | cost | p95 |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in summary["leaderboard_rows"]:
            lines.append(
                f"| {row['model']} | {row['overall_acc']} | {row['cost']} | {row['latency_p95']} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
