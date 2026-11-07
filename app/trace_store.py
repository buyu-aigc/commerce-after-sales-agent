from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

TRACE_DB = Path("data") / "agent_traces.db"


def record_trace(
    *,
    session_id: str,
    user_message: str,
    result: dict[str, Any],
    latency_ms: float,
    status: str,
) -> str:
    TRACE_DB.parent.mkdir(parents=True, exist_ok=True)
    trace_id = str(uuid.uuid4())
    with sqlite3.connect(TRACE_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_trace (
                trace_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                session_id TEXT NOT NULL,
                route_intent TEXT,
                user_message TEXT NOT NULL,
                final_answer TEXT,
                sources_json TEXT,
                trajectory_json TEXT,
                after_sales_json TEXT,
                latency_ms REAL NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        _ensure_trace_schema(conn)
        sources = _redact_value(result.get("retrieved_insights", []) or result.get("retrieved_policy", []))
        trajectory = _redact_value(result.get("trajectory_events", []))
        after_sales_cases = _redact_value(result.get("after_sales_cases", []))
        conn.execute(
            """
            INSERT INTO agent_trace (
                trace_id, created_at, session_id, route_intent, user_message, final_answer,
                sources_json, trajectory_json, after_sales_json, latency_ms, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                time.time(),
                session_id,
                result.get("route_intent", ""),
                _redact_text(user_message)[:1000],
                _redact_text(str(result.get("final_answer", "")))[:2000],
                json.dumps(sources, ensure_ascii=False),
                json.dumps(trajectory, ensure_ascii=False),
                json.dumps(after_sales_cases, ensure_ascii=False),
                latency_ms,
                status,
            ),
        )
    return trace_id


def _ensure_trace_schema(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_trace)").fetchall()}
    if "route_intent" not in columns:
        conn.execute("ALTER TABLE agent_trace ADD COLUMN route_intent TEXT")
    if "trajectory_json" not in columns:
        conn.execute("ALTER TABLE agent_trace ADD COLUMN trajectory_json TEXT")
    if "after_sales_json" not in columns:
        conn.execute("ALTER TABLE agent_trace ADD COLUMN after_sales_json TEXT")


def list_session_traces(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if not TRACE_DB.exists():
        return []
    with sqlite3.connect(TRACE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT trace_id, created_at, session_id, route_intent, user_message,
                   final_answer, sources_json, trajectory_json, after_sales_json, latency_ms, status
            FROM agent_trace
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def trace_summary() -> dict[str, Any]:
    if not TRACE_DB.exists():
        return {
            "total": 0,
            "status_counts": {},
            "route_intent_counts": {},
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    with sqlite3.connect(TRACE_DB) as conn:
        rows = conn.execute(
            "SELECT status, route_intent, latency_ms FROM agent_trace ORDER BY created_at"
        ).fetchall()
    latencies = sorted(float(row[2]) for row in rows)
    status_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for status, route_intent, _latency in rows:
        status_counts[str(status)] = status_counts.get(str(status), 0) + 1
        route = str(route_intent or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
    p50_index = int(0.50 * (len(latencies) - 1)) if latencies else 0
    p95_index = int(0.95 * (len(latencies) - 1)) if latencies else 0
    return {
        "total": len(rows),
        "status_counts": status_counts,
        "route_intent_counts": route_counts,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(latencies[p50_index], 2) if latencies else 0.0,
        "p95_latency_ms": round(latencies[p95_index], 2) if latencies else 0.0,
    }


def case_metrics() -> dict[str, Any]:
    """Aggregate business-facing after-sales metrics from recorded traces."""
    if not TRACE_DB.exists():
        return _empty_case_metrics()
    with sqlite3.connect(TRACE_DB) as conn:
        _ensure_trace_schema(conn)
        rows = conn.execute(
            """
            SELECT status, route_intent, trajectory_json, after_sales_json, latency_ms
            FROM agent_trace
            ORDER BY created_at
            """
        ).fetchall()

    cases: list[dict[str, Any]] = []
    tool_events = 0
    failed_tool_events = 0
    write_actions = 0
    blocked_writes = 0
    case_latencies = []
    total_cost = 0.0
    cost_samples = 0
    for status, _route, trajectory_json, after_sales_json, latency_ms in rows:
        trajectory = _loads_list(trajectory_json)
        trace_cases = _loads_list(after_sales_json)
        if trace_cases:
            case_latencies.append(float(latency_ms))
        for event in trajectory:
            details = event.get("details", {}) if isinstance(event, dict) else {}
            if isinstance(details, dict) and details.get("tool"):
                tool_events += 1
                if event.get("status") == "failed" or str(status) not in {"ok", "pending_confirmation"}:
                    failed_tool_events += 1
        for case in trace_cases:
            cases.append(case)
            decision = case.get("decision", {}) if isinstance(case, dict) else {}
            action = str(case.get("action_type", ""))
            if action in {"refund_request", "cancel_order", "change_address", "invoice_request"}:
                write_actions += 1
                outcome = str(decision.get("outcome", ""))
                if outcome in {"reject", "ask_clarification"}:
                    blocked_writes += 1
            cost = case.get("cost_usd") if isinstance(case, dict) else None
            if isinstance(cost, (int, float)):
                total_cost += float(cost)
                cost_samples += 1

    total_cases = len(cases)
    if total_cases == 0:
        metrics = _empty_case_metrics()
        metrics["tool_error_rate"] = _rate(failed_tool_events, tool_events)
        metrics["p50_latency_ms"] = _p50([float(row[4]) for row in rows])
        metrics["p95_latency_ms"] = _p95([float(row[4]) for row in rows])
        return metrics

    hitl_cases = 0
    auto_cases = 0
    policy_hit_cases = 0
    for case in cases:
        decision = case.get("decision", {}) if isinstance(case, dict) else {}
        verification = case.get("verification", {}) if isinstance(case, dict) else {}
        next_step = str(verification.get("required_next_step", ""))
        outcome = str(decision.get("outcome", ""))
        if next_step == "hitl" or decision.get("requires_human") is True:
            hitl_cases += 1
        if next_step == "execute" or (outcome == "reject" and decision.get("requires_human") is False):
            auto_cases += 1
        refs = case.get("policy_refs") if isinstance(case, dict) else None
        if isinstance(refs, list) and refs:
            policy_hit_cases += 1

    return {
        "total_cases": total_cases,
        "auto_resolution_rate": _rate(auto_cases, total_cases),
        "hitl_rate": _rate(hitl_cases, total_cases),
        "wrong_write_blocked": blocked_writes,
        "wrong_write_block_rate": _rate(blocked_writes, write_actions),
        "policy_hit_rate": _rate(policy_hit_cases, total_cases),
        "tool_error_rate": _rate(failed_tool_events, tool_events),
        "p50_latency_ms": _p50(case_latencies),
        "p95_latency_ms": _p95(case_latencies),
        "cost_per_case": round(total_cost / cost_samples, 6) if cost_samples else 0.0,
        "cost_sample_count": cost_samples,
    }


def _empty_case_metrics() -> dict[str, Any]:
    return {
        "total_cases": 0,
        "auto_resolution_rate": 0.0,
        "hitl_rate": 0.0,
        "wrong_write_blocked": 0,
        "wrong_write_block_rate": 0.0,
        "policy_hit_rate": 0.0,
        "tool_error_rate": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "cost_per_case": 0.0,
        "cost_sample_count": 0,
    }


def _loads_list(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _p50(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return round(ordered[int(0.50 * (len(ordered) - 1))], 2)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return round(ordered[int(0.95 * (len(ordered) - 1))], 2)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in value.items()}
    return value


def _redact_text(text: str) -> str:
    redacted = re.sub(
        r"\b[0-9a-fA-F]{32}\b",
        lambda match: f"{match.group(0)[:8]}...{match.group(0)[-4:]}",
        text,
    )
    redacted = re.sub(
        r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        r"\1***\2",
        redacted,
    )
    redacted = re.sub(
        r"\b1[3-9]\d{9}\b",
        lambda match: f"{match.group(0)[:3]}****{match.group(0)[-4:]}",
        redacted,
    )
    # Brazilian CPF, international and local BR phone formats.
    redacted = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "***.***.***-**", redacted)
    redacted = re.sub(
        r"\+\d{1,3}[\s-]?\(?\d{2,3}\)?[\s-]?\d{4,5}[\s-]?\d{4}\b",
        lambda match: f"{match.group(0)[:3]}****{match.group(0)[-4:]}",
        redacted,
    )
    redacted = re.sub(
        r"\(\d{2}\)\s?9?\d{4}-\d{4}\b",
        lambda match: f"({match.group(0)[1:3]}) ****-{match.group(0)[-4:]}",
        redacted,
    )
    # Labelled delivery address / recipient fields.  Unlabelled prose is left
    # untouched because aggressive NER-style redaction causes false positives.
    return re.sub(
        r"((?:收货|收件|配送|联系)?地址|收件人|收货人)\s*[:：]\s*[^\n，。;；]{2,}",
        r"\1：[已脱敏]",
        redacted,
    )
