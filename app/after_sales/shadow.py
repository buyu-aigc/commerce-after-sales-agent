"""Shadow-mode suggestions and human-decision comparison.

Shadow mode runs the same decomposition and policy checks as the live path, but
never writes an order, never executes an action and never creates a real Case.
When shadow sampling is on, the live path deliberately routes every write
sub-request to human review so that the comparison has a real human label.

The store is intentionally separate from the business Case store: shadow data
must be deletable independently and must never become an execution source.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.after_sales.complexity import ComplexityAssessment, SubRequest

SUGGEST_AUTO = "auto_execute"
SUGGEST_REVIEW = "human_review"
SUGGEST_CLARIFY = "clarify"

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
AMOUNT_BANDS = (
    (0.0, 50.0, "0-50"),
    (50.0, 100.0, "50-100"),
    (100.0, 300.0, "100-300"),
    (300.0, 1000.0, "300-1000"),
    (1000.0, float("inf"), "1000+"),
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(max(value, 0.0), 1.0)


def _amount_band(amount: float) -> str:
    for low, high, label in AMOUNT_BANDS:
        if low <= amount < high:
            return label
    return "1000+"


@dataclass(frozen=True)
class ShadowSuggestion:
    observation_id: str
    request_id: str
    session_id: str
    sub_task_index: int
    sub_task_total: int
    action_type: str
    order_id: str
    risk_level: str
    complexity_tier: str
    policy_decision: str
    policy_id: str
    suggested_outcome: str
    suggested_reason: str
    evidence_complete: bool
    missing_evidence: list[str]
    risk_reasons: list[str]
    amount: float
    amount_band: str

    def to_record(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "sub_task_index": self.sub_task_index,
            "sub_task_total": self.sub_task_total,
            "action_type": self.action_type,
            "order_id": self.order_id,
            "risk_level": self.risk_level,
            "complexity_tier": self.complexity_tier,
            "policy_decision": self.policy_decision,
            "policy_id": self.policy_id,
            "suggested_outcome": self.suggested_outcome,
            "suggested_reason": self.suggested_reason,
            "evidence_complete": self.evidence_complete,
            "missing_evidence": json.dumps(self.missing_evidence, ensure_ascii=False),
            "risk_reasons": json.dumps(self.risk_reasons, ensure_ascii=False),
            "amount": self.amount,
            "amount_band": self.amount_band,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "sub_task_index": self.sub_task_index,
            "sub_task_total": self.sub_task_total,
            "action_type": self.action_type,
            "risk_level": self.risk_level,
            "complexity_tier": self.complexity_tier,
            "policy_decision": self.policy_decision,
            "policy_id": self.policy_id,
            "suggested_outcome": self.suggested_outcome,
            "suggested_reason": self.suggested_reason,
            "evidence_complete": self.evidence_complete,
            "missing_evidence": list(self.missing_evidence),
            "risk_reasons": list(self.risk_reasons),
            "amount_band": self.amount_band,
        }


class SQLiteShadowStore:
    """Separate, append-only shadow observation store."""

    def __init__(self, path: str | Path = "data/shadow.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_observations (
                    observation_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    sub_task_index INTEGER NOT NULL,
                    sub_task_total INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    order_id TEXT NOT NULL DEFAULT '',
                    risk_level TEXT NOT NULL,
                    complexity_tier TEXT NOT NULL DEFAULT '',
                    policy_decision TEXT NOT NULL DEFAULT '',
                    policy_id TEXT NOT NULL DEFAULT '',
                    suggested_outcome TEXT NOT NULL,
                    suggested_reason TEXT NOT NULL DEFAULT '',
                    evidence_complete INTEGER NOT NULL DEFAULT 1,
                    missing_evidence TEXT NOT NULL DEFAULT '[]',
                    risk_reasons TEXT NOT NULL DEFAULT '[]',
                    amount REAL NOT NULL DEFAULT 0,
                    amount_band TEXT NOT NULL DEFAULT '',
                    case_id TEXT,
                    human_decision TEXT,
                    human_reviewer TEXT,
                    human_reason TEXT,
                    decided_at TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(shadow_observations)")}
            for name, ddl in {
                "automation_eligible": "INTEGER",
                "label_source": "TEXT",
                "eligibility_reviewer": "TEXT",
            }.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE shadow_observations ADD COLUMN {name} {ddl}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_request ON shadow_observations(request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_case ON shadow_observations(case_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_created ON shadow_observations(created_at)")

    def record_suggestions(self, suggestions: list[ShadowSuggestion]) -> int:
        if not suggestions:
            return 0
        written = 0
        with self._connect() as conn:
            for suggestion in suggestions:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO shadow_observations (
                        observation_id, request_id, session_id, sub_task_index,
                        sub_task_total, action_type, order_id, risk_level,
                        complexity_tier, policy_decision, policy_id,
                        suggested_outcome, suggested_reason, evidence_complete,
                        missing_evidence, risk_reasons, amount, amount_band,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        suggestion.observation_id,
                        suggestion.request_id,
                        suggestion.session_id,
                        suggestion.sub_task_index,
                        suggestion.sub_task_total,
                        suggestion.action_type,
                        suggestion.order_id,
                        suggestion.risk_level,
                        suggestion.complexity_tier,
                        suggestion.policy_decision,
                        suggestion.policy_id,
                        suggestion.suggested_outcome,
                        suggestion.suggested_reason,
                        1 if suggestion.evidence_complete else 0,
                        json.dumps(suggestion.missing_evidence, ensure_ascii=False),
                        json.dumps(suggestion.risk_reasons, ensure_ascii=False),
                        suggestion.amount,
                        suggestion.amount_band,
                        _now(),
                    ),
                )
                written += max(cursor.rowcount, 0)
        return written

    def link_case(
        self,
        *,
        case_id: str,
        request_id: str,
        sub_task_index: int,
        action_type: str,
    ) -> int:
        if not case_id or not request_id:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE shadow_observations
                SET case_id = ?
                WHERE request_id = ?
                  AND sub_task_index = ?
                  AND action_type = ?
                  AND (case_id IS NULL OR case_id = '')
                """,
                (case_id, request_id, int(sub_task_index), action_type),
            )
            if cursor.rowcount:
                return cursor.rowcount
            cursor = conn.execute(
                """
                UPDATE shadow_observations
                SET case_id = ?
                WHERE request_id = ?
                  AND action_type = ?
                  AND (case_id IS NULL OR case_id = '')
                """,
                (case_id, request_id, action_type),
            )
            return max(cursor.rowcount, 0)

    def record_human_decision(
        self,
        *,
        case_id: str,
        decision: str,
        reviewer_id: str = "",
        reason: str = "",
        decided_at: str | None = None,
    ) -> int:
        if decision not in {"approve", "reject"} or not case_id:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE shadow_observations
                SET human_decision = ?, human_reviewer = ?, human_reason = ?,
                    decided_at = ?
                WHERE case_id = ?
                  AND (human_decision IS NULL OR human_decision = '')
                """,
                (decision, reviewer_id, reason[:1000], decided_at or _now(), case_id),
            )
            return max(cursor.rowcount, 0)

    def record_automation_label(
        self, *, observation_id: str, eligible: bool, source: str, reviewer_id: str
    ) -> int:
        """Independent automation eligibility label, separate from business approval.

        A caller must actually arrange blind review before using real_blind_review;
        a source string by itself is not proof of label quality.
        """
        if type(eligible) is not bool or not source.strip() or not reviewer_id.strip():
            raise ValueError("explicit_eligibility_source_and_reviewer_required")
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE shadow_observations SET automation_eligible = ?,
                label_source = ?, eligibility_reviewer = ?
                WHERE observation_id = ? AND automation_eligible IS NULL""",
                (int(eligible), source, reviewer_id, observation_id),
            )
            return max(cursor.rowcount, 0)

    def record_decision_for_observation(
        self,
        *,
        observation_id: str,
        decision: str,
        reviewer_id: str = "",
        reason: str = "",
    ) -> int:
        if decision not in {"approve", "reject"} or not observation_id:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE shadow_observations
                SET human_decision = ?, human_reviewer = ?, human_reason = ?,
                    decided_at = ?
                WHERE observation_id = ?
                  AND (human_decision IS NULL OR human_decision = '')
                """,
                (decision, reviewer_id, reason[:1000], _now(), observation_id),
            )
            return max(cursor.rowcount, 0)

    def records(
        self,
        *,
        limit: int = 100,
        risk_level: str = "",
        action_type: str = "",
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if risk_level:
            clauses.append("risk_level = ?")
            params.append(risk_level)
        if action_type:
            clauses.append("action_type = ?")
            params.append(action_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM shadow_observations {where} ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def metrics(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM shadow_observations ORDER BY created_at ASC"
                ).fetchall()
            ]
        return _metrics_from_rows(rows)

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM shadow_observations")


class ShadowService:
    """Build non-executing suggestions and compare them with human decisions."""

    def __init__(
        self,
        store: SQLiteShadowStore,
        *,
        enabled: bool = False,
        sample_rate: float = 1.0,
    ) -> None:
        self.store = store
        self.enabled = bool(enabled)
        self.sample_rate = min(max(float(sample_rate), 0.0), 1.0)

    def should_shadow(self, session_id: str, tenant_id: str = "") -> bool:
        if not self.enabled or self.sample_rate <= 0:
            return False
        if self.sample_rate >= 1:
            return True
        seed = f"{tenant_id}:{session_id}"
        bucket = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16) % 10000
        return bucket < int(self.sample_rate * 10000)

    def record_analysis(
        self,
        *,
        request_id: str,
        session_id: str,
        analysis: ComplexityAssessment,
        order_lookup: Callable[[str], Any] | None,
        policy_guard: Any | None,
        repeat_claim: bool = False,
        refund_failure: bool = False,
    ) -> list[dict[str, Any]]:
        suggestions = self.build_suggestions(
            request_id=request_id,
            session_id=session_id,
            analysis=analysis,
            order_lookup=order_lookup,
            policy_guard=policy_guard,
            repeat_claim=repeat_claim,
            refund_failure=refund_failure,
        )
        self.store.record_suggestions(suggestions)
        return [suggestion.to_dict() for suggestion in suggestions]

    def build_suggestions(
        self,
        *,
        request_id: str,
        session_id: str,
        analysis: ComplexityAssessment,
        order_lookup: Callable[[str], Any] | None,
        policy_guard: Any | None,
        repeat_claim: bool = False,
        refund_failure: bool = False,
    ) -> list[ShadowSuggestion]:
        write_units = [
            (index, unit) for index, unit in enumerate(analysis.units, start=1) if unit.side_effect
        ]
        total = len(analysis.units)
        suggestions: list[ShadowSuggestion] = []
        for index, unit in write_units:
            order = _lookup_order(order_lookup, unit.order_id)
            policy, suggested, reason = _suggest(
                unit,
                order,
                policy_guard,
                repeat_claim=repeat_claim,
                refund_failure=refund_failure,
            )
            amount = _order_float(order, "payment_value")
            suggestions.append(
                ShadowSuggestion(
                    observation_id=_observation_id(request_id, index, unit),
                    request_id=request_id,
                    session_id=session_id,
                    sub_task_index=index,
                    sub_task_total=total,
                    action_type=unit.action_type,
                    order_id=unit.order_id,
                    risk_level=unit.risk_level,
                    complexity_tier=analysis.tier,
                    policy_decision=str((policy or {}).get("decision") or "review"),
                    policy_id=str((policy or {}).get("policy_id") or ""),
                    suggested_outcome=suggested,
                    suggested_reason=reason,
                    evidence_complete=unit.evidence_complete,
                    missing_evidence=list(unit.missing_evidence),
                    risk_reasons=list(unit.risk_reasons),
                    amount=amount,
                    amount_band=_amount_band(amount),
                )
            )
        return suggestions

    def link_case(
        self,
        *,
        case_id: str,
        request_id: str,
        sub_task_index: int,
        action_type: str,
    ) -> int:
        return self.store.link_case(
            case_id=case_id,
            request_id=request_id,
            sub_task_index=sub_task_index,
            action_type=action_type,
        )

    def record_human_decision(
        self,
        *,
        case_id: str,
        decision: str,
        reviewer_id: str = "",
        reason: str = "",
    ) -> int:
        return self.store.record_human_decision(
            case_id=case_id,
            decision=decision,
            reviewer_id=reviewer_id,
            reason=reason,
        )

    def metrics(self) -> dict[str, Any]:
        payload = self.store.metrics()
        payload["enabled"] = self.enabled
        payload["sample_rate"] = self.sample_rate
        return payload

    def recent(self, *, limit: int = 50, risk_level: str = "", action_type: str = "") -> list[dict]:
        return [
            _public_record(row)
            for row in self.store.records(limit=limit, risk_level=risk_level, action_type=action_type)
        ]


def build_shadow_service_from_env() -> ShadowService:
    enabled = _env_bool("SHADOW_MODE", False)
    default_rate = 1.0 if enabled else 0.0
    return ShadowService(
        SQLiteShadowStore(os.environ.get("SHADOW_DB_PATH") or "data/shadow.db"),
        enabled=enabled,
        sample_rate=_env_float("SHADOW_SAMPLE_RATE", default_rate),
    )


def _suggest(
    unit: SubRequest,
    order: Any,
    policy_guard: Any | None,
    *,
    repeat_claim: bool = False,
    refund_failure: bool = False,
) -> tuple[dict[str, Any] | None, str, str]:
    if not unit.order_id or order is None:
        return None, SUGGEST_CLARIFY, "missing_order_evidence"
    policy: dict[str, Any] | None = None
    if policy_guard is not None:
        try:
            policy = policy_guard.check(
                unit.action_type,
                unit.order_id,
                risk_level=unit.risk_level,
                repeat_claim=repeat_claim,
                refund_failure=refund_failure,
            )
        except Exception:
            policy = None
    if unit.risk_level == "high":
        return policy, SUGGEST_REVIEW, "high_risk_blocked"
    if not unit.evidence_complete:
        return policy, SUGGEST_REVIEW, "missing_customer_evidence"
    if policy is None:
        return None, SUGGEST_REVIEW, "no_policy_decision"
    if policy.get("decision") == "auto":
        return policy, SUGGEST_AUTO, "policy_auto"
    return policy, SUGGEST_REVIEW, "policy_review"


def _lookup_order(order_lookup: Callable[[str], Any] | None, order_id: str) -> Any:
    if order_lookup is None or not order_id:
        return None
    try:
        return order_lookup(order_id)
    except Exception:
        return None


def _order_float(order: Any, name: str) -> float:
    value = order.get(name) if isinstance(order, dict) else getattr(order, name, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _observation_id(request_id: str, index: int, unit: SubRequest) -> str:
    seed = f"{request_id}:{index}:{unit.action_type}:{unit.order_id}"
    return f"shadow-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    decided_rows = [row for row in rows if row.get("human_decision") in {"approve", "reject"}]
    auto_rows = [row for row in rows if row.get("suggested_outcome") == SUGGEST_AUTO]
    review_rows = [row for row in rows if row.get("suggested_outcome") == SUGGEST_REVIEW]
    clarify_rows = [row for row in rows if row.get("suggested_outcome") == SUGGEST_CLARIFY]
    auto_decided = [row for row in auto_rows if row.get("human_decision") in {"approve", "reject"}]
    review_decided = [row for row in review_rows if row.get("human_decision") in {"approve", "reject"}]
    auto_approved = [row for row in auto_decided if row["human_decision"] == "approve"]
    auto_rejected = [row for row in auto_decided if row["human_decision"] == "reject"]
    review_approved = [row for row in review_decided if row["human_decision"] == "approve"]
    review_rejected = [row for row in review_decided if row["human_decision"] == "reject"]
    routing_rows = auto_rows + review_rows
    routing_fully_labeled = bool(routing_rows) and all(
        row.get("automation_eligible") in {0, 1} for row in routing_rows
    )
    eligible_total = sum(row.get("automation_eligible") == 1 for row in routing_rows)
    eligible_reviews = [row for row in review_rows if row.get("automation_eligible") == 1]
    labeled_auto = [row for row in auto_rows if row.get("automation_eligible") in {0, 1}]
    unsafe_auto = [row for row in labeled_auto if row.get("automation_eligible") == 0]
    trusted_auto = [
        row
        for row in labeled_auto
        if row.get("label_source") == "real_blind_review"
        and row.get("eligibility_reviewer")
        and row.get("human_decision") in {"approve", "reject"}
    ]
    independent_auto = len({row.get("request_id") for row in trusted_auto if row.get("request_id")})
    # One-sided 95% zero-event bound on independent requests; correlated subcases
    # are deliberately not counted as extra samples.
    zero_errors = not unsafe_auto and not auto_rejected
    error_upper = 1 - 0.05 ** (1 / independent_auto) if independent_auto and zero_errors else None
    gate_ready = (
        independent_auto >= 30
        and len(trusted_auto) == len(auto_rows)
        and zero_errors
        and error_upper is not None
        and error_upper <= 0.01
    )
    high_risk = [row for row in rows if row.get("risk_level") == "high"]
    high_risk_auto = [row for row in high_risk if row.get("suggested_outcome") == SUGGEST_AUTO]
    return {
        "summary": {
            "suggestions": total,
            "requests": len({row.get("request_id") for row in rows if row.get("request_id")}),
            "linked_cases": sum(1 for row in rows if row.get("case_id")),
            "decided": len(decided_rows),
            "pending_decision": total - len(decided_rows),
            "auto_suggestions": len(auto_rows),
            "auto_decided": len(auto_decided),
            "auto_approved": len(auto_approved),
            "false_auto": len(auto_rejected),
            "false_auto_rate": round(len(auto_rejected) / len(auto_decided), 4) if auto_decided else None,
            "business_approval_rate": round(len(auto_approved) / len(auto_decided), 4)
            if auto_decided
            else None,
            "auto_precision": round(1 - len(unsafe_auto) / len(labeled_auto), 4) if labeled_auto else None,
            "automation_labeled": sum(row.get("automation_eligible") in {0, 1} for row in rows),
            "auto_eligibility_labeled": len(labeled_auto),
            "unsafe_auto_labels": len(unsafe_auto),
            "independent_auto_requests": independent_auto,
            "zero_error_upper_95": error_upper,
            "review_suggestions": len(review_rows),
            "review_decided": len(review_decided),
            "review_approved": len(review_approved),
            "missed_auto": len(eligible_reviews),
            "missed_auto_rate": (
                round(len(eligible_reviews) / eligible_total, 4)
                if routing_fully_labeled and eligible_total
                else None
            ),
            "routing_eligibility_complete": routing_fully_labeled,
            "review_rejected": len(review_rejected),
            "clarify_suggestions": len(clarify_rows),
            "high_risk_suggestions": len(high_risk),
            "high_risk_auto_suggestions": len(high_risk_auto),
            "routing_agreement": round((len(auto_approved) + len(review_rejected)) / len(decided_rows), 4)
            if decided_rows
            else 0.0,
        },
        "gates": {
            "high_risk_auto_zero": len(high_risk_auto) == 0,
            "false_auto_zero": len(auto_rejected) == 0,
            "minimum_decided_samples": len(decided_rows) >= 30,
            "minimum_independent_auto_samples": independent_auto >= 30,
            "independent_labels_complete": bool(auto_rows) and len(trusted_auto) == len(auto_rows),
            "auto_error_bound_below_one_percent": error_upper is not None and error_upper <= 0.01,
            "eligibility_safety": not unsafe_auto,
            "coverage_requires_manual_review": True,
            "metric_version": "2-eligibility-separated",
            "recommendation": (
                "review_for_rollout" if len(high_risk_auto) == 0 and gate_ready else "keep_shadow"
            ),
        },
        "by_risk": _group_metrics(rows, "risk_level"),
        "by_action": _group_metrics(rows, "action_type"),
        "by_complexity": _group_metrics(rows, "complexity_tier"),
        "by_amount": _group_metrics(rows, "amount_band"),
    }


def _group_metrics(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    result = []
    for name in sorted(groups):
        bucket = groups[name]
        decided = [row for row in bucket if row.get("human_decision") in {"approve", "reject"}]
        auto = [row for row in bucket if row.get("suggested_outcome") == SUGGEST_AUTO]
        auto_decided = [row for row in auto if row.get("human_decision") in {"approve", "reject"}]
        false_auto = [row for row in auto_decided if row["human_decision"] == "reject"]
        approved_review = [
            row
            for row in bucket
            if row.get("suggested_outcome") == SUGGEST_REVIEW and row.get("automation_eligible") == 1
        ]
        result.append(
            {
                "key": name,
                "suggestions": len(bucket),
                "decided": len(decided),
                "auto_suggestions": len(auto),
                "false_auto": len(false_auto),
                "missed_auto": len(approved_review),
                "high_risk_auto": sum(
                    1
                    for row in bucket
                    if row.get("risk_level") == "high" and row.get("suggested_outcome") == SUGGEST_AUTO
                ),
            }
        )
    return result


def _public_record(row: dict[str, Any]) -> dict[str, Any]:
    order_id = str(row.get("order_id") or "")
    decision = str(row.get("human_decision") or "")
    suggested = str(row.get("suggested_outcome") or "")
    agreement = None
    if decision in {"approve", "reject"}:
        agreement = (suggested == SUGGEST_AUTO and decision == "approve") or (
            suggested == SUGGEST_REVIEW and decision == "reject"
        )
    return {
        "observation_id": row.get("observation_id"),
        "request_id": row.get("request_id"),
        "session_id": row.get("session_id"),
        "sub_task_index": row.get("sub_task_index"),
        "sub_task_total": row.get("sub_task_total"),
        "action_type": row.get("action_type"),
        "order_id": f"{order_id[:8]}...{order_id[-4:]}" if len(order_id) >= 12 else order_id,
        "risk_level": row.get("risk_level"),
        "complexity_tier": row.get("complexity_tier"),
        "policy_decision": row.get("policy_decision"),
        "policy_id": row.get("policy_id"),
        "suggested_outcome": suggested,
        "suggested_reason": row.get("suggested_reason"),
        "amount_band": row.get("amount_band"),
        "case_id": row.get("case_id"),
        "human_decision": decision or None,
        "automation_eligible": (
            bool(row["automation_eligible"]) if row.get("automation_eligible") is not None else None
        ),
        "label_source": row.get("label_source"),
        "human_reviewer": row.get("human_reviewer"),
        "human_reason": row.get("human_reason"),
        "decided_at": row.get("decided_at"),
        "created_at": row.get("created_at"),
        "agreement": agreement,
    }


__all__ = [
    "SQLiteShadowStore",
    "ShadowService",
    "ShadowSuggestion",
    "SUGGEST_AUTO",
    "SUGGEST_REVIEW",
    "SUGGEST_CLARIFY",
    "build_shadow_service_from_env",
]
