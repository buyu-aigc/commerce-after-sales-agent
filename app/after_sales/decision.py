from __future__ import annotations

from app.after_sales.models import AfterSalesCase, AfterSalesDecision, VerificationResult
from app.olist.service import OrderStatusView
from app.olist.status import PRE_SHIPMENT_STATUSES


class AfterSalesDecisionEngine:
    """Rule-backed decision layer for e-commerce after-sales handling.

    LLMs can understand the customer's wording, but refund/cancellation/address
    writes need deterministic policy gates. This engine turns order facts and
    policy hits into an auditable approve/reject/review decision.
    """

    def assess(
        self,
        *,
        action_type: str,
        order: OrderStatusView,
        user_request: str,
        policy_sections: list[dict[str, object]],
    ) -> AfterSalesCase:
        safe_action = _normalize_action(action_type)
        reason_code = _reason_code(user_request, order)
        policy_refs = _policy_refs(policy_sections)
        risk_signals = _risk_signals(order, user_request)
        evidence = _evidence(order)

        decision = self._decide(
            action_type=safe_action,
            reason_code=reason_code,
            order=order,
            policy_refs=policy_refs,
            risk_signals=risk_signals,
            evidence=evidence,
        )
        verification = verify_decision(decision)
        reply = draft_customer_reply(order, decision, verification)
        return AfterSalesCase(
            order_id=order.order_id,
            user_request=user_request,
            action_type=safe_action,
            order_status=order.status,
            payment_value=order.payment_value,
            delay_days=order.delay_days,
            review_score=order.review_score,
            category_summary=order.category_summary,
            risk_signals=risk_signals,
            policy_refs=policy_refs,
            decision=decision,
            verification=verification,
            customer_reply=reply,
        )

    def _decide(
        self,
        *,
        action_type: str,
        reason_code: str,
        order: OrderStatusView,
        policy_refs: list[str],
        risk_signals: dict[str, object],
        evidence: list[str],
    ) -> AfterSalesDecision:
        allowed: list[str] = []
        blocked: list[str] = []
        points: list[str] = []
        risk = "medium"
        confidence = 0.72
        outcome = "needs_human_review"
        requires_human = True

        if action_type == "cancel_order":
            if order.status in PRE_SHIPMENT_STATUSES:
                allowed.append("cancel_order")
                points.append("订单尚未发货，可以提交取消申请。")
                risk = "medium" if order.payment_value >= 300 else "low"
                confidence = 0.82
                if _low_risk_auto_gate(action_type, order, policy_refs, risk_signals):
                    outcome = "approve"
                    requires_human = False
            elif order.status == "canceled":
                outcome = "reject"
                blocked.append("cancel_order")
                points.append("订单已经取消，不能重复取消。")
                risk = "low"
                confidence = 0.9
                requires_human = False
            else:
                outcome = "reject"
                blocked.append("cancel_order")
                points.append("订单已发货或已送达，不能直接取消。")
                risk = "medium"
                confidence = 0.86
                requires_human = False
        elif action_type == "change_address":
            if order.status in PRE_SHIPMENT_STATUSES:
                allowed.append("change_address")
                points.append("订单尚未锁定物流，可以准备改址申请。")
                risk = "medium"
                confidence = 0.78
                if _low_risk_auto_gate(action_type, order, policy_refs, risk_signals):
                    outcome = "approve"
                    requires_human = False
            else:
                outcome = "reject"
                blocked.append("change_address")
                points.append("订单已发货、已送达或已取消，不能直接修改地址。")
                risk = "medium"
                confidence = 0.84
                requires_human = False
        elif action_type == "invoice_request":
            allowed.append("invoice_request")
            points.append("可以准备发票申请，但需要补齐抬头、税号和邮箱等资料。")
            risk = "medium"
            confidence = 0.75
        elif action_type == "complaint_escalation":
            allowed.append("complaint_escalation")
            points.append("投诉升级会创建可追踪工单，但不能在确认前承诺退款、补偿或处罚结果。")
            risk = "high" if order.payment_value >= 300 else "medium"
            confidence = 0.76
        elif action_type == "refund_request":
            if order.status == "canceled":
                allowed.append("refund_request")
                points.append("订单已取消且存在支付记录，可以创建退款核查申请。")
                confidence = 0.8
            elif isinstance(order.delay_days, int) and order.delay_days >= 7:
                allowed.append("refund_request")
                points.append(f"订单延迟 {order.delay_days} 天，命中延迟售后核查条件。")
                confidence = 0.82
            elif isinstance(order.review_score, int) and order.review_score <= 2:
                allowed.append("open_support_case")
                points.append("低评分需要先核查商品、物流或服务问题，不直接承诺退款。")
                confidence = 0.7
            else:
                outcome = "ask_clarification"
                blocked.append("refund_request")
                points.append("当前事实不足以判断退款资格，需要补充退款原因或异常证据。")
                risk = "low"
                confidence = 0.58
                requires_human = False
        else:
            allowed.append("open_support_case")
            points.append("可以创建售后跟进工单，后续由人工或下游系统核查。")
            confidence = 0.68
            if _low_risk_auto_gate("open_support_case", order, policy_refs, risk_signals):
                outcome = "approve"
                risk = "low"
                confidence = 0.84
                requires_human = False

        if order.payment_value >= 300 and outcome != "reject":
            risk = "high"
            confidence = min(confidence, 0.74)
            points.append("订单金额较高，需要人工复核后再执行。")
            if outcome == "approve":
                outcome = "needs_human_review"
                requires_human = True

        if not policy_refs:
            confidence = min(confidence, 0.62)
            points.append("未命中明确政策段落，不能自动执行高风险动作。")

        if outcome == "needs_human_review":
            requires_human = True

        handoff_reasons = _handoff_reasons(
            action_type=action_type,
            outcome=outcome,
            risk=risk,
            confidence=confidence,
            requires_human=requires_human,
            policy_refs=policy_refs,
            risk_signals=risk_signals,
            reason_code=reason_code,
        )

        return AfterSalesDecision(
            outcome=outcome,
            action_type=action_type,  # type: ignore[arg-type]
            reason_code=reason_code,
            confidence=round(confidence, 2),
            risk_level=risk,  # type: ignore[arg-type]
            requires_human=requires_human,
            handoff_reasons=handoff_reasons,
            allowed_actions=allowed,
            blocked_actions=blocked,
            evidence=evidence,
            policy_refs=policy_refs,
            customer_message_points=points,
        )


def verify_decision(decision: AfterSalesDecision) -> VerificationResult:
    flags: list[str] = []
    if decision.action_type in {
        "refund_request",
        "cancel_order",
        "change_address",
        "invoice_request",
        "complaint_escalation",
    }:
        if not decision.requires_human and decision.outcome == "needs_human_review":
            flags.append("high_risk_action_without_hitl")
        if decision.action_type in decision.allowed_actions and not decision.policy_refs:
            flags.append("write_action_without_policy_reference")
    if decision.confidence < 0.65 and decision.outcome not in {"ask_clarification", "needs_human_review"}:
        flags.append("low_confidence_without_safe_exit")
    if decision.outcome == "reject" and decision.allowed_actions:
        flags.append("reject_with_allowed_write_action")

    if flags:
        next_step = "hitl" if decision.requires_human else "stop"
    elif decision.outcome == "ask_clarification":
        next_step = "clarify"
    elif decision.requires_human:
        next_step = "hitl"
    elif decision.outcome == "reject":
        next_step = "stop"
    else:
        next_step = "execute"
    return VerificationResult(passed=not flags, flags=flags, required_next_step=next_step)


def _handoff_reasons(
    *,
    action_type: str,
    outcome: str,
    risk: str,
    confidence: float,
    requires_human: bool,
    policy_refs: list[str],
    risk_signals: dict[str, object],
    reason_code: str,
) -> list[str]:
    reasons: list[str] = []
    if requires_human:
        reasons.append("human_required_by_policy")
    if requires_human and action_type in {
        "refund_request",
        "cancel_order",
        "change_address",
        "invoice_request",
    }:
        reasons.append("write_action_requires_review")
    if requires_human and (action_type == "complaint_escalation" or risk_signals.get("mentions_complaint")):
        reasons.append("complaint_or_emotional_escalation")
    if requires_human and (risk == "high" or risk_signals.get("high_value")):
        reasons.append("high_value_or_high_risk_order")
    if requires_human and risk_signals.get("has_delivery_delay"):
        reasons.append("delivery_sla_breach")
    if requires_human and risk_signals.get("has_low_review"):
        reasons.append("low_customer_rating")
    if requires_human and confidence < 0.75:
        reasons.append("low_decision_confidence")
    if requires_human and not policy_refs:
        reasons.append("missing_policy_evidence")
    if outcome == "ask_clarification":
        reasons.append("missing_required_customer_information")
    if requires_human and "refund_requested" in reason_code and risk_signals.get("mentions_compensation"):
        reasons.append("refund_or_compensation_request")
    return list(dict.fromkeys(reasons))


def draft_customer_reply(
    order: OrderStatusView,
    decision: AfterSalesDecision,
    verification: VerificationResult,
) -> str:
    facts = (
        f"订单 {order.order_id[:8]} 当前状态为 {order.status}，"
        f"支付金额 {order.payment_value:.2f}，类目 {order.category_summary}。"
    )
    if order.delay_days is not None:
        facts += f"系统记录显示配送延迟 {order.delay_days} 天。"
    if order.review_score is not None:
        facts += f"评价分为 {order.review_score}。"

    if decision.outcome == "reject":
        action = "根据当前订单状态，不能直接执行该售后动作。"
    elif decision.outcome == "ask_clarification":
        action = "目前还缺少判断资格所需的信息，请补充具体售后原因或异常证据。"
    elif verification.required_next_step == "hitl":
        action = "我可以先为您提交人工审核/售后处理申请，在确认前不会承诺退款或补偿结果。"
    else:
        action = "该请求符合低风险处理条件，可以继续执行。"

    points = "；".join(decision.customer_message_points[:3])
    return f"您好，{facts}{action}处理依据：{points}。"


def _normalize_action(action_type: str) -> str:
    allowed = {
        "open_support_case",
        "refund_request",
        "cancel_order",
        "change_address",
        "invoice_request",
        "complaint_escalation",
    }
    return action_type if action_type in allowed else "open_support_case"


def _policy_refs(policy_sections: list[dict[str, object]]) -> list[str]:
    refs = []
    for item in policy_sections:
        title = str(item.get("section_title", "")).strip()
        if title and title not in refs:
            refs.append(title)
    return refs[:5]


def _evidence(order: OrderStatusView) -> list[str]:
    evidence = [
        f"order_status={order.status}",
        f"payment_value={order.payment_value:.2f}",
        f"category={order.category_summary}",
    ]
    if order.delay_days is not None:
        evidence.append(f"delay_days={order.delay_days}")
    if order.review_score is not None:
        evidence.append(f"review_score={order.review_score}")
    return evidence


def _risk_signals(order: OrderStatusView, user_request: str) -> dict[str, object]:
    text = user_request.lower()
    return {
        "is_delivered": order.status == "delivered",
        "is_shipped_or_delivered": order.status in {"shipped", "delivered"},
        "is_canceled": order.status == "canceled",
        "pre_shipment": order.status in PRE_SHIPMENT_STATUSES,
        "high_value": order.payment_value >= 300,
        "has_delivery_delay": isinstance(order.delay_days, int) and order.delay_days >= 7,
        "has_low_review": isinstance(order.review_score, int) and order.review_score <= 2,
        "mentions_compensation": any(token in text for token in ("refund", "退款", "赔付", "补偿", "退钱")),
        "mentions_address": any(token in text for token in ("address", "地址")),
        "mentions_complaint": any(token in text for token in ("complaint", "投诉", "升级")),
        "already_refunded": False,
        "partial_refund": False,
        "coupon_or_points_payment": False,
        "suspected_abuse": False,
    }


def _low_risk_auto_gate(
    action_type: str,
    order: OrderStatusView,
    policy_refs: list[str],
    risk_signals: dict[str, object],
) -> bool:
    if not policy_refs:
        return False
    blocking_signals = {
        "high_value",
        "suspected_abuse",
        "already_refunded",
        "partial_refund",
        "has_delivery_delay",
        "has_low_review",
        "mentions_compensation",
        "mentions_complaint",
    }
    if any(risk_signals.get(name) for name in blocking_signals):
        return False
    if action_type == "open_support_case":
        return order.payment_value < 300
    if action_type == "cancel_order":
        return bool(risk_signals.get("pre_shipment")) and order.payment_value < 50
    if action_type == "change_address":
        return bool(risk_signals.get("pre_shipment")) and order.payment_value < 50
    return False


def _reason_code(user_request: str, order: OrderStatusView) -> str:
    text = user_request.lower()
    reasons = []
    if "refund" in text or "退款" in text or "退钱" in text or "赔付" in text:
        reasons.append("refund_requested")
    if "cancel" in text or "取消" in text:
        reasons.append("cancel_requested")
    if "address" in text or "地址" in text:
        reasons.append("address_change_requested")
    if "invoice" in text or "发票" in text:
        reasons.append("invoice_requested")
    if "complaint" in text or "投诉" in text or "升级" in text:
        reasons.append("complaint_escalation_requested")
    if isinstance(order.delay_days, int) and order.delay_days > 0:
        reasons.append("delivery_delay")
    if isinstance(order.review_score, int) and order.review_score <= 2:
        reasons.append("low_review")
    return "+".join(reasons) if reasons else "generic_after_sales"
