"""Structured runtime observability for tools, policy, and recovery events.

The trace store answers "what happened in this conversation".  This module
answers "how is the runtime behaving right now": tool error rates/latency,
policy auto/review mix, guardrail downgrades, auto execution failures, and
crash-recovery counts.

Counters and latency samples are process-local and intentionally bounded; a
multi-worker deployment should export the same events to a shared metrics
backend.  Every event is also emitted as one JSON log line so a collector can
scrape it without the in-process endpoint.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("aipm.observability")

_MAX_LATENCY_SAMPLES = 5000
_LOCK = threading.Lock()

_TOOL_CALLS: Counter[str] = Counter()
_TOOL_ERRORS: Counter[str] = Counter()
_TOOL_LATENCIES: dict[str, list[float]] = defaultdict(list)
_POLICY_DECISIONS: Counter[str] = Counter()
_POLICY_BY_ID: dict[str, Counter[str]] = defaultdict(Counter)
_POLICY_GUARDRAILS: Counter[str] = Counter()
_POLICY_VERSIONS: Counter[str] = Counter()
_AUTO_OUTCOMES: Counter[str] = Counter()
_RECOVERY: Counter[str] = Counter()
_RECOVERY_BY_TYPE: dict[str, Counter[str]] = defaultdict(Counter)
_LAST_RECOVERY_AT: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def reset() -> None:
    """Clear process-local metrics (tests and explicit operator resets)."""
    with _LOCK:
        _TOOL_CALLS.clear()
        _TOOL_ERRORS.clear()
        _TOOL_LATENCIES.clear()
        _POLICY_DECISIONS.clear()
        _POLICY_BY_ID.clear()
        _POLICY_GUARDRAILS.clear()
        _POLICY_VERSIONS.clear()
        _AUTO_OUTCOMES.clear()
        _RECOVERY.clear()
        _RECOVERY_BY_TYPE.clear()
        global _LAST_RECOVERY_AT
        _LAST_RECOVERY_AT = None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(percentile * (len(ordered) - 1))
    return round(ordered[index], 2)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def record_tool_call(
    *,
    tool: str,
    latency_ms: float,
    ok: bool,
    error_code: str | None = None,
    result_count: int = 0,
    redacted_args: dict[str, Any] | None = None,
    tenant_id: str = "",
    user_id: str = "",
    session_id: str = "",
) -> None:
    latency = max(float(latency_ms or 0.0), 0.0)
    with _LOCK:
        _TOOL_CALLS[tool] += 1
        if not ok:
            _TOOL_ERRORS[tool] += 1
        samples = _TOOL_LATENCIES[tool]
        samples.append(latency)
        if len(samples) > _MAX_LATENCY_SAMPLES:
            del samples[: len(samples) - _MAX_LATENCY_SAMPLES]
    logger.info(
        json.dumps(
            {
                "event": "tool_call",
                "tool": tool,
                "ok": bool(ok),
                "error_code": error_code,
                "result_count": int(result_count or 0),
                "latency_ms": round(latency, 3),
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "args": redacted_args or {},
            },
            ensure_ascii=False,
        )
    )


def record_policy_decision(
    decision: dict[str, Any],
    *,
    action_type: str,
    session_id: str = "",
    user_id: str = "",
    source: str = "workflow",
) -> None:
    verdict = str(decision.get("decision") or "review")
    policy_id = str(decision.get("policy_id") or "unknown")
    version = str(decision.get("policy_version") or "")
    guardrails = [str(item) for item in (decision.get("guardrail") or [])]
    with _LOCK:
        _POLICY_DECISIONS[verdict] += 1
        _POLICY_BY_ID[policy_id][verdict] += 1
        if version:
            _POLICY_VERSIONS[version] += 1
        for guardrail in guardrails:
            _POLICY_GUARDRAILS[guardrail] += 1
    logger.info(
        json.dumps(
            {
                "event": "policy_decision",
                "policy_id": policy_id,
                "decision": verdict,
                "reason": decision.get("reason", ""),
                "policy_version": version,
                "action_type": action_type,
                "source": source,
                "guardrail": guardrails,
                "session_id": session_id,
                "user_id": user_id,
            },
            ensure_ascii=False,
        )
    )


def record_auto_outcome(decision: dict[str, Any], *, executed: bool) -> None:
    """Count an auto-approved case that did or did not execute successfully."""
    outcome = "executed" if executed else "failed"
    with _LOCK:
        _AUTO_OUTCOMES[outcome] += 1
    logger.info(
        json.dumps(
            {
                "event": "auto_outcome",
                "policy_id": decision.get("policy_id", ""),
                "decision": decision.get("decision", "auto"),
                "outcome": outcome,
            },
            ensure_ascii=False,
        )
    )


def record_recovery_run() -> None:
    global _LAST_RECOVERY_AT
    with _LOCK:
        _RECOVERY["runs"] += 1
        _LAST_RECOVERY_AT = _now_iso()
    logger.info(json.dumps({"event": "recovery_run"}, ensure_ascii=False))


def record_recovery(job_type: str, case_id: str, *, success: bool) -> None:
    outcome = "succeeded" if success else "failed"
    with _LOCK:
        _RECOVERY["attempted"] += 1
        _RECOVERY[outcome] += 1
        _RECOVERY_BY_TYPE[job_type][outcome] += 1
    logger.info(
        json.dumps(
            {
                "event": "recovery",
                "job_type": job_type,
                "case_id": case_id,
                "outcome": outcome,
            },
            ensure_ascii=False,
        )
    )


def tool_snapshot() -> dict[str, Any]:
    with _LOCK:
        total = sum(_TOOL_CALLS.values())
        errors = sum(_TOOL_ERRORS.values())
        all_latencies = [value for samples in _TOOL_LATENCIES.values() for value in samples]
        by_tool = {}
        for tool, calls in sorted(_TOOL_CALLS.items()):
            samples = list(_TOOL_LATENCIES.get(tool, []))
            by_tool[tool] = {
                "calls": calls,
                "errors": _TOOL_ERRORS.get(tool, 0),
                "error_rate": _rate(_TOOL_ERRORS.get(tool, 0), calls),
                "p50_latency_ms": _percentile(samples, 0.50),
                "p95_latency_ms": _percentile(samples, 0.95),
            }
    return {
        "total": total,
        "errors": errors,
        "error_rate": _rate(errors, total),
        "p50_latency_ms": _percentile(all_latencies, 0.50),
        "p95_latency_ms": _percentile(all_latencies, 0.95),
        "by_tool": by_tool,
    }


def policy_snapshot() -> dict[str, Any]:
    with _LOCK:
        auto = _POLICY_DECISIONS.get("auto", 0)
        review = _POLICY_DECISIONS.get("review", 0)
        total = auto + review
        auto_failed = _AUTO_OUTCOMES.get("failed", 0)
        auto_executed = _AUTO_OUTCOMES.get("executed", 0)
        by_policy = {
            policy_id: {
                "auto": counts.get("auto", 0),
                "review": counts.get("review", 0),
            }
            for policy_id, counts in sorted(_POLICY_BY_ID.items())
        }
        guardrails = dict(sorted(_POLICY_GUARDRAILS.items()))
        versions = dict(sorted(_POLICY_VERSIONS.items()))
    return {
        "total": total,
        "auto": auto,
        "review": review,
        "auto_rate": _rate(auto, total),
        "auto_executed": auto_executed,
        "auto_failed": auto_failed,
        # Runtime-observable proxy: an auto decision whose execution failed.
        # Policy-quality false-auto still comes from the 600-row replay gate.
        "false_auto": auto_failed,
        "false_auto_rate": _rate(auto_failed, max(auto_executed + auto_failed, 1)),
        "by_policy": by_policy,
        "guardrail_downgrades": guardrails,
        "policy_versions": versions,
    }


def recovery_snapshot() -> dict[str, Any]:
    with _LOCK:
        by_type = {
            job_type: {
                "succeeded": counts.get("succeeded", 0),
                "failed": counts.get("failed", 0),
            }
            for job_type, counts in sorted(_RECOVERY_BY_TYPE.items())
        }
        snapshot = {
            "runs": _RECOVERY.get("runs", 0),
            "attempted": _RECOVERY.get("attempted", 0),
            "succeeded": _RECOVERY.get("succeeded", 0),
            "failed": _RECOVERY.get("failed", 0),
            "by_job_type": by_type,
            "last_run_at": _LAST_RECOVERY_AT,
        }
    return snapshot


def runtime_snapshot() -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "tools": tool_snapshot(),
        "policy": policy_snapshot(),
        "recovery": recovery_snapshot(),
    }
