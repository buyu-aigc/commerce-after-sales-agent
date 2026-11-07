from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AfterSalesAction = Literal[
    "open_support_case",
    "refund_request",
    "cancel_order",
    "change_address",
    "invoice_request",
    "complaint_escalation",
]

DecisionOutcome = Literal[
    "approve",
    "reject",
    "needs_human_review",
    "ask_clarification",
]

RiskLevel = Literal["low", "medium", "high"]


class AfterSalesDecision(BaseModel):
    """Structured business decision before any support side effect is executed."""

    outcome: DecisionOutcome
    action_type: AfterSalesAction
    reason_code: str
    confidence: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    requires_human: bool
    handoff_reasons: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    blocked_actions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    policy_refs: list[str] = Field(default_factory=list)
    customer_message_points: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Verifier output used to prevent unsafe promises and unsupported writes."""

    passed: bool
    flags: list[str] = Field(default_factory=list)
    required_next_step: Literal["execute", "hitl", "clarify", "stop"]


class AfterSalesCase(BaseModel):
    """Durable case-resolution object that the graph can trace and replay."""

    case_type: str = "after_sales_resolution"
    order_id: str
    user_request: str
    action_type: AfterSalesAction
    order_status: str
    payment_value: float
    delay_days: int | None = None
    review_score: int | None = None
    category_summary: str = ""
    risk_signals: dict[str, object] = Field(default_factory=dict)
    policy_refs: list[str] = Field(default_factory=list)
    decision: AfterSalesDecision
    verification: VerificationResult
    customer_reply: str
