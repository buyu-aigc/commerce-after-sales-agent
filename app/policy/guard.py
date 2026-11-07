"""Shared policy guard for both the agent tool and the execution path."""

from __future__ import annotations

from typing import Any


class PolicyGuard:
    def __init__(self, engine: Any, olist_service: Any, case_service: Any) -> None:
        self._engine = engine
        self._olist = olist_service
        self._case_service = case_service

    def check(
        self,
        action_type: str,
        order_id: str,
        *,
        risk_level: str = "",
        repeat_claim: bool = False,
        refund_failure: bool = False,
    ) -> dict[str, Any]:
        order = self._olist.get_order_status(str(order_id or ""))
        if order is None:
            return {
                "policy_id": "demo_default_review_v1",
                "decision": "review",
                "max_amount": 0.0,
                "reason": "order_not_found",
                "demo_only": True,
                "facts": {"action_type": action_type, "order_id": order_id},
            }
        try:
            prior = bool(self._case_service.count_order_events(order_id))
        except Exception:
            prior = False
        facts = {
            "action_type": action_type,
            "order_status": order.status,
            "delay_days": int(order.delay_days or 0),
            "amount": float(order.payment_value or 0),
            "prior_compensation": prior,
            "repeat_claim": bool(repeat_claim),
            "risk_level": str(risk_level or ""),
            "refund_failure": bool(refund_failure),
        }
        return self._engine.evaluate(facts)

    @staticmethod
    def customer_safe(decision: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": decision.get("decision"),
            "max_amount": decision.get("max_amount", 0),
            "policy_id": decision.get("policy_id"),
            "reason": decision.get("reason", ""),
        }
