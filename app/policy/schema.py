"""Validation, versioning, and conflict checks for after-sales policy packs.

The demo pack is data, not code.  Treating it as an untrusted artifact means the
process must fail closed when the data is malformed, ambiguous, or internally
contradictory.  This module is deliberately dependency-light: ``jsonschema`` is
already part of the runtime environment and everything else is pure Python.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

KNOWN_ACTIONS = (
    "refund_request",
    "cancel_order",
    "change_address",
    "invoice_request",
    "complaint_escalation",
    "open_support_case",
)
KNOWN_ORDER_STATUSES = (
    "created",
    "approved",
    "invoiced",
    "processing",
    "shipped",
    "delivered",
    "canceled",
    "unavailable",
)
KNOWN_RISK_LEVELS = ("low", "medium", "high")
KNOWN_FACT_KEYS = (
    "action_type",
    "order_status",
    "delay_days",
    "amount",
    "prior_compensation",
    "repeat_claim",
    "risk_level",
    "refund_failure",
)
KNOWN_CONDITION_KEYS = (
    "order_status",
    "delay_days_min",
    "delay_days_max",
    "amount_min",
    "amount_max",
    "prior_compensation",
    "repeat_claim",
    "refund_failure",
    "risk_level",
    "risk_level_not",
)


class PolicyPackError(ValueError):
    """Raised when a policy pack is invalid or internally contradictory."""


_ISO_FORMAT = FormatChecker()


def _iso_datetime(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    try:
        parse_timestamp(value)
    except (TypeError, ValueError):
        return False
    return True


_ISO_FORMAT.checks("date-time")(_iso_datetime)


CONDITIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "order_status": {
            "anyOf": [
                {"type": "string", "enum": list(KNOWN_ORDER_STATUSES)},
                {
                    "type": "array",
                    "items": {"type": "string", "enum": list(KNOWN_ORDER_STATUSES)},
                    "minItems": 1,
                    "uniqueItems": True,
                },
            ]
        },
        "delay_days_min": {"type": "number"},
        "delay_days_max": {"type": "number"},
        "amount_min": {"type": "number"},
        "amount_max": {"type": "number"},
        "prior_compensation": {"type": "boolean"},
        "repeat_claim": {"type": "boolean"},
        "refund_failure": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": list(KNOWN_RISK_LEVELS)},
        "risk_level_not": {
            "type": "array",
            "items": {"type": "string", "enum": list(KNOWN_RISK_LEVELS)},
            "uniqueItems": True,
        },
    },
}

POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "policy_id",
        "priority",
        "action_types",
        "conditions",
        "decision",
        "max_amount",
        "reason",
    ],
    "properties": {
        "policy_id": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9_]*_v[0-9]+$"},
        "priority": {"type": "integer", "minimum": 0},
        "action_types": {
            "type": "array",
            "items": {"type": "string", "enum": list(KNOWN_ACTIONS)},
            "minItems": 1,
            "uniqueItems": True,
        },
        "conditions": CONDITIONS_SCHEMA,
        "decision": {"type": "string", "enum": ["auto", "review"]},
        "max_amount": {"type": "number", "minimum": 0},
        "reason": {"type": "string", "minLength": 1},
        "effective_from": {"type": ["string", "null"], "format": "date-time"},
        "effective_to": {"type": ["string", "null"], "format": "date-time"},
    },
}

POLICY_PACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["demo_only", "version", "policies"],
    "properties": {
        "demo_only": {"type": "boolean"},
        "version": {"type": "string", "pattern": r"^[0-9]+\.[0-9]+(\.[0-9]+)?$"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "effective_from": {"type": ["string", "null"], "format": "date-time"},
        "effective_to": {"type": ["string", "null"], "format": "date-time"},
        "policies": {
            "type": "array",
            "minItems": 1,
            "items": POLICY_SCHEMA,
        },
    },
}


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an effective window boundary into an aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=UTC)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def canonical_pack_hash(pack: dict[str, Any]) -> str:
    payload = json.dumps(pack, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _issue(code: str, message: str, *, policy_ids: list[str] | None = None, path: str = "") -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "message": message}
    if policy_ids:
        issue["policy_ids"] = policy_ids
    if path:
        issue["path"] = path
    return issue


def _fmt_path(parts: list[Any]) -> str:
    return ".".join(str(part) for part in parts)


def validate_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Return a structured validation report; never raise for bad data."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    validator = Draft202012Validator(POLICY_PACK_SCHEMA, format_checker=_ISO_FORMAT)
    for error in sorted(validator.iter_errors(pack), key=lambda item: list(item.path)):
        errors.append(
            _issue(
                "schema_error",
                error.message,
                path=_fmt_path(list(error.path)),
            )
        )
    if errors:
        return _report(pack, errors, warnings)

    policies: list[dict[str, Any]] = list(pack.get("policies") or [])
    by_id: dict[str, dict[str, Any]] = {}
    for policy in policies:
        policy_id = str(policy["policy_id"])
        if policy_id in by_id:
            errors.append(
                _issue("duplicate_policy_id", f"policy_id 重复：{policy_id}", policy_ids=[policy_id])
            )
        by_id[policy_id] = policy

    default_policies = [policy for policy in policies if not policy.get("conditions")]
    if not default_policies:
        errors.append(
            _issue(
                "missing_default_policy",
                "缺少兜底 review 政策（conditions 为空且 decision=review）",
            )
        )
    for policy in default_policies:
        if policy.get("decision") != "review":
            errors.append(
                _issue(
                    "unsafe_default_policy",
                    "conditions 为空的兜底政策必须是 review，不能自动执行",
                    policy_ids=[str(policy["policy_id"])],
                )
            )

    max_priority = max(int(policy["priority"]) for policy in policies)
    for policy in policies:
        policy_id = str(policy["policy_id"])
        conditions = dict(policy.get("conditions") or {})
        if not conditions and int(policy["priority"]) != max_priority:
            errors.append(
                _issue(
                    "default_policy_not_last",
                    "兜底政策必须使用最大 priority，确保最后才匹配",
                    policy_ids=[policy_id],
                )
            )
        _check_window(pack, policy, errors)
        if policy.get("decision") == "auto":
            _check_auto_policy(policy, errors)

    if default_policies:
        default_actions = set(default_policies[0].get("action_types") or [])
        used_actions = {action for policy in policies for action in policy.get("action_types") or []}
        missing = sorted(used_actions - default_actions)
        if missing:
            errors.append(
                _issue(
                    "default_action_coverage",
                    f"兜底政策未覆盖动作：{', '.join(missing)}",
                    policy_ids=[str(default_policies[0]["policy_id"])],
                )
            )

    conflicts = detect_conflicts(pack)
    errors.extend(conflicts["errors"])
    warnings.extend(conflicts["warnings"])
    return _report(pack, errors, warnings)


def _check_window(pack: dict[str, Any], policy: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    policy_id = str(policy["policy_id"])
    try:
        start = parse_timestamp(policy.get("effective_from") or pack.get("effective_from"))
        end = parse_timestamp(policy.get("effective_to") or pack.get("effective_to"))
    except (TypeError, ValueError) as exc:
        errors.append(
            _issue("invalid_effective_window", f"生效时间无法解析：{exc}", policy_ids=[policy_id])
        )
        return
    if start and end and start >= end:
        errors.append(
            _issue(
                "invalid_effective_window",
                "effective_from 必须早于 effective_to（右开区间）",
                policy_ids=[policy_id],
            )
        )


def _check_auto_policy(policy: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    policy_id = str(policy["policy_id"])
    conditions = dict(policy.get("conditions") or {})
    ceiling = float(policy.get("max_amount") or 0)
    if ceiling <= 0:
        errors.append(
            _issue(
                "auto_policy_missing_amount_cap",
                "自动政策必须设置大于 0 的 max_amount 上限",
                policy_ids=[policy_id],
            )
        )
    amount_max = conditions.get("amount_max")
    if amount_max is None:
        errors.append(
            _issue(
                "auto_policy_missing_amount_condition",
                "自动政策必须在 conditions 中声明 amount_max，避免无金额边界",
                policy_ids=[policy_id],
            )
        )
    elif ceiling > float(amount_max):
        errors.append(
            _issue(
                "auto_policy_cap_exceeds_condition",
                f"max_amount({ceiling}) 不能大于 amount_max({amount_max})",
                policy_ids=[policy_id],
            )
        )


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Domain:
    """Allowed values for one fact: either a discrete set or a numeric interval."""

    values: frozenset[Any] | None = None
    lo: float | None = None
    hi: float | None = None


_ALL_STATUS = frozenset(KNOWN_ORDER_STATUSES)
_ALL_RISK = frozenset(KNOWN_RISK_LEVELS)
_ALL_BOOL = frozenset({False, True})


def _numeric_condition(conditions: dict[str, Any], name: str) -> _Domain:
    lo = conditions.get(f"{name}_min")
    hi = conditions.get(f"{name}_max")
    return _Domain(
        lo=float(lo) if lo is not None else None,
        hi=float(hi) if hi is not None else None,
    )


def _policy_domains(policy: dict[str, Any]) -> dict[str, _Domain]:
    conditions = dict(policy.get("conditions") or {})
    statuses = conditions.get("order_status", KNOWN_ORDER_STATUSES)
    if isinstance(statuses, str):
        status_values = frozenset({statuses})
    else:
        status_values = frozenset(statuses)
    risk_excluded = frozenset(conditions.get("risk_level_not") or [])
    if "risk_level" in conditions:
        risk_values = frozenset({str(conditions["risk_level"])})
    elif risk_excluded:
        risk_values = _ALL_RISK - risk_excluded
    else:
        risk_values = _ALL_RISK
    return {
        "action_type": _Domain(values=frozenset(policy.get("action_types") or [])),
        "order_status": _Domain(values=status_values),
        "risk_level": _Domain(values=risk_values),
        "delay_days": _numeric_condition(conditions, "delay_days"),
        "amount": _numeric_condition(conditions, "amount"),
        "prior_compensation": _Domain(values=frozenset({bool(conditions["prior_compensation"])}))
        if "prior_compensation" in conditions
        else _Domain(values=_ALL_BOOL),
        "repeat_claim": _Domain(values=frozenset({bool(conditions["repeat_claim"])}))
        if "repeat_claim" in conditions
        else _Domain(values=_ALL_BOOL),
        "refund_failure": _Domain(values=frozenset({bool(conditions["refund_failure"])}))
        if "refund_failure" in conditions
        else _Domain(values=_ALL_BOOL),
    }


def _overlap(left: _Domain, right: _Domain) -> bool:
    if left.values is not None and right.values is not None:
        return bool(left.values & right.values)
    if left.values is None and right.values is None:
        lo = max((value for value in (left.lo, right.lo) if value is not None), default=float("-inf"))
        hi = min((value for value in (left.hi, right.hi) if value is not None), default=float("inf"))
        return lo <= hi
    return True


def _subset(left: _Domain, right: _Domain) -> bool:
    if left.values is not None and right.values is not None:
        return left.values <= right.values
    if left.values is None and right.values is None:
        lo_ok = right.lo is None or (left.lo is not None and left.lo >= right.lo)
        hi_ok = right.hi is None or (left.hi is not None and left.hi <= right.hi)
        return lo_ok and hi_ok
    return False


def _domains_overlap(left: dict[str, _Domain], right: dict[str, _Domain]) -> bool:
    return all(_overlap(left[key], right[key]) for key in left)


def _domains_subset(left: dict[str, _Domain], right: dict[str, _Domain]) -> bool:
    return all(_subset(left[key], right[key]) for key in left)


def _same_rule(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        set(left.get("action_types") or []) == set(right.get("action_types") or [])
        and dict(left.get("conditions") or {}) == dict(right.get("conditions") or {})
        and left.get("decision") == right.get("decision")
    )


def detect_conflicts(pack: dict[str, Any]) -> dict[str, Any]:
    """Detect ambiguous, duplicate, shadowed, or unsafe overlapping rules."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    policies = sorted(
        (dict(policy) for policy in pack.get("policies") or []),
        key=lambda item: (int(item.get("priority", 999)), str(item.get("policy_id", ""))),
    )
    domains = {str(policy["policy_id"]): _policy_domains(policy) for policy in policies}

    for index, first in enumerate(policies):
        first_id = str(first["policy_id"])
        for second in policies[index + 1 :]:
            second_id = str(second["policy_id"])
            if not _domains_overlap(domains[first_id], domains[second_id]):
                continue
            if _same_rule(first, second):
                errors.append(
                    _issue(
                        "duplicate_rule",
                        "存在动作、条件、决策完全相同的重复规则",
                        policy_ids=[first_id, second_id],
                    )
                )
                continue
            if int(first["priority"]) == int(second["priority"]) and first["decision"] != second["decision"]:
                errors.append(
                    _issue(
                        "same_priority_conflict",
                        "同一 priority 下存在可重叠但决策不同的政策，结果不确定",
                        policy_ids=[first_id, second_id],
                    )
                )
                continue
            if first["decision"] == "auto" and second["decision"] == "review":
                if second.get("conditions") and _domains_subset(domains[second_id], domains[first_id]):
                    warnings.append(
                        _issue(
                            "auto_overrides_review",
                            "高优先级自动政策完全覆盖了一条更窄的人工审核政策",
                            policy_ids=[first_id, second_id],
                        )
                    )
            elif first["decision"] == second["decision"] and second.get("conditions"):
                if _domains_subset(domains[second_id], domains[first_id]):
                    warnings.append(
                        _issue(
                            "shadowed_policy",
                            "较低优先级政策完全被前面的同类政策遮蔽，永远不会命中",
                            policy_ids=[first_id, second_id],
                        )
                    )
    return {"errors": errors, "warnings": warnings}


def _report(
    pack: dict[str, Any], errors: list[dict[str, Any]], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "ok": not errors,
        "version": pack.get("version"),
        "pack_hash": canonical_pack_hash(pack),
        "policy_count": len(pack.get("policies") or []),
        "errors": errors,
        "warnings": warnings,
    }


def load_pack(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
