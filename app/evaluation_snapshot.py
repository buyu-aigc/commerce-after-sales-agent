"""Read persisted evaluation artifacts for the authenticated review console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_evaluation_snapshot(
    *, runtime_summary: dict[str, Any], runtime_cases: dict[str, Any]
) -> dict[str, Any]:
    published = _load_json(ROOT / "evaluation" / "published_results.json")
    return {
        "runtime": {
            "scope": "本地开发 trace，含手工演示与调试；用于排障，不等同于线上业务 KPI。",
            "trace_count": runtime_summary.get("total", 0),
            "p95_latency_ms": runtime_summary.get("p95_latency_ms", 0.0),
            "case_metrics": runtime_cases,
        },
        "product_evaluation": published,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

