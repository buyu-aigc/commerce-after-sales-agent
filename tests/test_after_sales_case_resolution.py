import pytest

from app.after_sales import AfterSalesDecisionEngine
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import InMemoryCaseService, OlistService, SQLiteCaseService, format_order_status
from app.retrieval.hybrid import HybridSupportRetriever
from app.tool_call import ToolCallContext, build_business_tool_manager

ORDER_ID = "203096f03d82e0dffbc41ebc2e2bcfb7"


def test_after_sales_engine_requires_hitl_for_refund_case() -> None:
    service = OlistService()
    order = service.get_order_status(ORDER_ID)
    assert order is not None
    policies = MarkdownKnowledgeBase().search("退款 延迟 补偿 人工确认", k=3)
    case = AfterSalesDecisionEngine().assess(
        action_type="refund_request",
        order=order,
        user_request=f"我的订单 {ORDER_ID} 晚到很多天，我要退款",
        policy_sections=[
            {
                "section_title": hit.section_title,
                "source": hit.source,
                "text": hit.text,
                "score": hit.score,
            }
            for hit in policies
        ],
    )

    assert case.decision.action_type == "refund_request"
    assert case.decision.outcome == "needs_human_review"
    assert case.decision.requires_human is True
    assert "write_action_requires_review" in case.decision.handoff_reasons
    assert "delivery_sla_breach" in case.decision.handoff_reasons
    assert "delay_days=11" in case.decision.evidence
    assert case.verification.required_next_step == "hitl"
    assert "不会承诺退款" in case.customer_reply


def test_after_sales_engine_rejects_delivered_order_cancellation() -> None:
    service = OlistService()
    order = service.get_order_status(ORDER_ID)
    assert order is not None

    case = AfterSalesDecisionEngine().assess(
        action_type="cancel_order",
        order=order,
        user_request=f"取消订单 {ORDER_ID}",
        policy_sections=[{"section_title": "Address Change Policy"}],
    )

    assert case.decision.outcome == "reject"
    assert case.decision.requires_human is False
    assert case.decision.handoff_reasons == []
    assert "cancel_order" in case.decision.blocked_actions
    assert case.verification.required_next_step == "stop"
    assert "不能直接执行" in case.customer_reply


def test_after_sales_engine_routes_complaint_escalation_through_hitl() -> None:
    service = OlistService()
    order = service.get_order_status(ORDER_ID)
    assert order is not None

    case = AfterSalesDecisionEngine().assess(
        action_type="complaint_escalation",
        order=order,
        user_request=f"我要投诉升级订单 {ORDER_ID} 的延迟和低评分问题",
        policy_sections=[{"section_title": "Complaint Escalation FAQ"}],
    )

    assert case.decision.action_type == "complaint_escalation"
    assert case.decision.outcome == "needs_human_review"
    assert case.decision.requires_human is True
    assert "complaint_or_emotional_escalation" in case.decision.handoff_reasons
    assert case.verification.required_next_step == "hitl"
    assert "投诉升级" in case.customer_reply


@pytest.mark.anyio
async def test_tool_manager_exposes_after_sales_case_assessment() -> None:
    manager = build_business_tool_manager(
        olist_service=OlistService(),
        knowledge_base=MarkdownKnowledgeBase(),
        support_retriever=HybridSupportRetriever(),
        case_service=InMemoryCaseService(),
    )

    result = await manager.call(
        "assess_after_sales_case",
        {
            "action_type": "refund_request",
            "order_id": ORDER_ID,
            "user_request": f"给订单 {ORDER_ID} 申请退款",
            "policy_sections": [{"section_title": "Compensation Boundary Policy"}],
        },
        ToolCallContext(role="support_agent", auth_scopes=["after_sales:assess"]),
    )

    assert result.ok
    assert result.data["found"] is True
    assert result.data["decision"]["outcome"] == "needs_human_review"
    assert result.data["verification"]["required_next_step"] == "hitl"


def test_after_sales_engine_auto_executes_low_risk_support_case() -> None:
    from app.olist.service import OrderStatusView

    order = OrderStatusView(
        order_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        status="delivered",
        customer_state="SP",
        purchase_timestamp="2018-01-01 10:00:00",
        estimated_delivery_date="2018-01-05 00:00:00",
        delivered_customer_date="2018-01-04 12:00:00",
        delay_days=0,
        payment_value=42.0,
        review_score=5,
        category_summary="books",
    )

    case = AfterSalesDecisionEngine().assess(
        action_type="open_support_case",
        order=order,
        user_request="请帮我创建一个普通售后跟进记录",
        policy_sections=[{"section_title": "General Support Case Policy"}],
    )

    assert case.decision.outcome == "approve"
    assert case.decision.requires_human is False
    assert case.decision.handoff_reasons == []
    assert case.decision.risk_level == "low"
    assert case.verification.required_next_step == "execute"
    assert case.risk_signals["high_value"] is False


def test_after_sales_engine_keeps_low_score_followup_in_hitl() -> None:
    service = OlistService()
    order = service.get_order_status(ORDER_ID)
    assert order is not None

    case = AfterSalesDecisionEngine().assess(
        action_type="open_support_case",
        order=order,
        user_request=f"订单 {ORDER_ID} 延迟且低分，生成客服跟进话术",
        policy_sections=[{"section_title": "Low Review Recovery Policy"}],
    )

    assert case.decision.outcome == "needs_human_review"
    assert case.decision.requires_human is True
    assert "delivery_sla_breach" in case.decision.handoff_reasons
    assert "low_customer_rating" in case.decision.handoff_reasons
    assert case.risk_signals["has_delivery_delay"] is True
    assert case.risk_signals["has_low_review"] is True


def test_approved_action_updates_order_projection_once(tmp_path) -> None:
    """A write case changes the sandbox read model and stays idempotent."""
    case_service = SQLiteCaseService(tmp_path / "cases.db")
    service = OlistService(projection_store=case_service)

    first = case_service.execute_action("refund_request", ORDER_ID, "订单延迟，申请退款")
    second = case_service.execute_action("refund_request", ORDER_ID, "订单延迟，申请退款")
    order = service.get_order_status(ORDER_ID)

    assert first["duplicate"] is False
    assert first["order_event"] is not None
    assert second["duplicate"] is True
    assert second["order_event"] is None
    assert order is not None
    assert order.refund_status == "requested"
    assert "退款申请：requested" in format_order_status(order)
