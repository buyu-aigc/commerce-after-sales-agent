"""Deterministic policy engine for the demo after-sales pack.

Hardening added in V2 P0-2:

* the pack is schema-validated and conflict-checked before use;
* policies carry effective windows (inherited from the pack when omitted);
* every decision records the pack version/hash for replay and audit;
* an automatic decision is downgraded to review if it would cross the amount
  ceiling or touch a hard-risk fact, even when a malformed pack says otherwise.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.policy.schema import (
    PolicyPackError,
    canonical_pack_hash,
    detect_conflicts,
    parse_timestamp,
    validate_pack,
)

DEFAULT_PACK = Path(__file__).resolve().parents[2] / "policies" / "demo_after_sales.json"
DEFAULT_REVIEW_POLICY_ID = "demo_default_review_v1"
HARD_REVIEW_FACTS = ("repeat_claim", "refund_failure", "prior_compensation")


class PolicyEngine:
    def __init__(
        self,
        pack_path: Path | str | None = None,
        *,
        as_of: datetime | str | float | None = None,
        validate: bool = True,
    ) -> None:
        self.path = Path(pack_path) if pack_path else DEFAULT_PACK
        self.pack = json.loads(self.path.read_text(encoding="utf-8"))
        self.pack_hash = canonical_pack_hash(self.pack)
        self.version = str(self.pack.get("version") or "")
        self.as_of = parse_timestamp(as_of)
        self.validation = validate_pack(self.pack)
        if validate and self.validation["errors"]:
            first = self.validation["errors"][0]
            raise PolicyPackError(
                f"invalid policy pack {self.path}: {first['code']}: {first['message']}"
            )
        self.conflicts = detect_conflicts(self.pack)
        self.policies = sorted(
            self.pack.get("policies", []),
            key=lambda item: (int(item.get("priority", 999)), str(item.get("policy_id", ""))),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        facts: dict[str, Any],
        *,
        as_of: datetime | str | float | None = None,
    ) -> dict[str, Any]:
        at = parse_timestamp(as_of) if as_of is not None else (self.as_of or _now())
        action_type = str(facts.get("action_type") or "")
        for policy in self.policies:
            if not self._active(policy, at):
                continue
            if action_type not in policy.get("action_types", []):
                continue
            if not self._matches(policy.get("conditions", {}), facts):
                continue
            decision = str(policy["decision"])
            if decision == "auto":
                blockers = self._auto_blockers(policy, facts)
                if blockers:
                    return self._decision(
                        policy,
                        facts,
                        at,
                        decision="review",
                        reason=(
                            f"自动政策 {policy['policy_id']} 被安全护栏拦截："
                            f"{', '.join(blockers)}；转人工审核"
                        ),
                        guardrail=blockers,
                    )
            return self._decision(policy, facts, at, decision=decision)
        return {
            "policy_id": DEFAULT_REVIEW_POLICY_ID,
            "decision": "review",
            "max_amount": 0.0,
            "reason": "没有命中的自动政策，默认转人工审核",
            "demo_only": bool(self.pack.get("demo_only")),
            "facts": facts,
            "policy_version": self.version,
            "pack_hash": self.pack_hash,
            "evaluated_at": at.isoformat(),
        }

    def _decision(
        self,
        policy: dict[str, Any],
        facts: dict[str, Any],
        at: datetime,
        *,
        decision: str,
        reason: str | None = None,
        guardrail: list[str] | None = None,
    ) -> dict[str, Any]:
        result = {
            "policy_id": policy["policy_id"],
            "decision": decision,
            "max_amount": float(policy.get("max_amount") or 0),
            "reason": reason if reason is not None else policy.get("reason", ""),
            "demo_only": bool(self.pack.get("demo_only")),
            "facts": facts,
            "policy_version": self.version,
            "pack_hash": self.pack_hash,
            "evaluated_at": at.isoformat(),
            "effective_from": policy.get("effective_from") or self.pack.get("effective_from"),
            "effective_to": policy.get("effective_to") or self.pack.get("effective_to"),
        }
        if guardrail:
            result["guardrail"] = guardrail
        return result

    def _active(self, policy: dict[str, Any], at: datetime) -> bool:
        start = parse_timestamp(policy.get("effective_from") or self.pack.get("effective_from"))
        end = parse_timestamp(policy.get("effective_to") or self.pack.get("effective_to"))
        if start is not None and at < start:
            return False
        if end is not None and at >= end:
            return False
        return True

    @staticmethod
    def _auto_blockers(policy: dict[str, Any], facts: dict[str, Any]) -> list[str]:
        blockers: list[str] = []
        try:
            amount = float(facts.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0.0
        ceiling = float(policy.get("max_amount") or 0)
        if ceiling <= 0:
            blockers.append("amount_cap_missing")
        elif amount > ceiling:
            blockers.append("amount_cap_exceeded")
        for key in HARD_REVIEW_FACTS:
            if bool(facts.get(key)):
                blockers.append(f"{key}_blocked")
        if str(facts.get("risk_level") or "") == "high":
            blockers.append("high_risk_blocked")
        return blockers

    @staticmethod
    def _matches(conditions: dict[str, Any], facts: dict[str, Any]) -> bool:
        for key, expected in conditions.items():
            if key.endswith("_min"):
                if float(facts.get(key[:-4]) or 0) < float(expected):
                    return False
            elif key.endswith("_max"):
                if float(facts.get(key[:-4]) or 0) > float(expected):
                    return False
            elif key.endswith("_not"):
                if facts.get(key[:-4]) in expected:
                    return False
            elif isinstance(expected, list):
                if facts.get(key) not in expected:
                    return False
            elif facts.get(key) != expected:
                return False
        return True

    # ------------------------------------------------------------------
    # Introspection helpers for operators / replay tooling
    # ------------------------------------------------------------------

    def active_policies(
        self, as_of: datetime | str | float | None = None
    ) -> list[dict[str, Any]]:
        at = parse_timestamp(as_of) if as_of is not None else (self.as_of or _now())
        return [policy for policy in self.policies if self._active(policy, at)]

    def describe(self, as_of: datetime | str | float | None = None) -> dict[str, Any]:
        at = parse_timestamp(as_of) if as_of is not None else (self.as_of or _now())
        return {
            "path": str(self.path),
            "version": self.version,
            "pack_hash": self.pack_hash,
            "as_of": at.isoformat(),
            "demo_only": bool(self.pack.get("demo_only")),
            "policy_count": len(self.policies),
            "active_policy_count": len(self.active_policies(at)),
            "validation": self.validation,
            "conflicts": self.conflicts,
        }


def _now() -> datetime:
    return datetime.now(UTC)
