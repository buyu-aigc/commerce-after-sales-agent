import asyncio
import os

# Must be set BEFORE importing app — LlmClient reads OPENAI_API_KEY at import time.
os.environ["OPENAI_API_KEY"] = "test-key"

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import (
    _customer_order_clarification,
    _resolve_auth_scopes,
    _should_use_customer_order_agent,
    app,
)
from app.models import ChatRequest, CustomerPageContext

ORDER_ID = "203096f03d82e0dffbc41ebc2e2bcfb7"
REVIEW_HEADERS = {"X-Review-Token": "local-review-demo"}


@pytest.fixture(autouse=True)
def setup_graph_and_state() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    import app.main as main_module
    from app.config.di import agent_graph_builder, case_service

    main_module.agent = agent_graph_builder.build(InMemorySaver())
    case_service.reset()


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_frontend_entrypoints_available(client: AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert "订单助手" in response.text
    assert "/static/product-customer.js" in response.text

    customer = await client.get("/customer")
    assert customer.status_code == 200
    assert "/static/product-customer.js" in customer.text

    review = await client.get("/review")
    assert review.status_code == 200
    assert "/static/product-review.js" in review.text

    customer_js = await client.get("/static/product-customer.js")
    assert customer_js.status_code == 200
    assert "/customer/chat" in customer_js.text

    review_js = await client.get("/static/product-review.js")
    assert review_js.status_code == 200
    assert "/review/sessions" in review_js.text

    technical = await client.get("/technical")
    assert technical.status_code == 200
    assert "/static/demo.js" in technical.text

    guided_demo = await client.get("/demo")
    assert guided_demo.status_code == 200
    assert "售后服务体验" in guided_demo.text

    scenarios = await client.get("/demo/scenarios")
    assert scenarios.status_code == 200
    assert len(scenarios.json()["scenarios"]) == 3


@pytest.mark.anyio
async def test_guided_demo_uses_customer_workflow_and_resumes_review(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2:
        started = await client.post("/demo/run", json={"scenario_id": "delayed_refund"})
    assert started.status_code == 200
    payload = started.json()
    assert payload["review_required"] is True
    assert payload["status"] == "pending_review"
    assert payload["review_packet"]["action_type"] == "refund_request"

    with g1, g2:
        approved = await client.post(
            f"/demo/sessions/{payload['session_id']}/review",
            json={"decision": "approve"},
        )
    assert approved.status_code == 200
    assert approved.json()["review_required"] is False
    assert approved.json()["status"] == "executed"
    assert "退款核查申请" in approved.json()["customer"]["answer"]
    assert "Agent" not in approved.json()["customer"]["answer"]

    for scenario_id in ("track_delivery", "delivered_address"):
        with g1, g2:
            response = await client.post("/demo/run", json={"scenario_id": scenario_id})
        assert response.status_code == 200
        assert response.json()["review_required"] is False


@pytest.mark.anyio
async def test_customer_small_talk_and_ambiguous_support_use_no_tool_terminal(client: AsyncClient) -> None:
    greeting = await client.post(
        "/customer/chat",
        json={"message": "你好", "user_id": "demo-customer", "session_id": "small-talk"},
    )
    ambiguous = await client.post(
        "/customer/chat",
        json={"message": "订单有问题", "user_id": "demo-customer", "session_id": "clarify"},
    )
    assert greeting.status_code == 200
    assert "查询订单" in greeting.json()["answer"]
    assert ambiguous.status_code == 200
    assert "订单" in ambiguous.json()["answer"]
    assert "退款" in ambiguous.json()["answer"]


def _mock_guard(input_on_topic: bool, output_valid: bool = True):
    from app.llm.types import InputGuardResult, OutputGuardResult

    async def check_input(self, message: str, history=None) -> InputGuardResult:
        reason = "ok" if input_on_topic else "off-topic"
        return InputGuardResult(on_topic=input_on_topic, reason=reason)

    async def check_output(self, answer: str) -> OutputGuardResult:
        return OutputGuardResult(valid=output_valid, reason="ok" if output_valid else "invalid")

    return (
        patch("app.llm.guardrail.Guardrail.check_input", check_input),
        patch("app.llm.guardrail.Guardrail.check_output", check_output),
    )


def _mock_plan(*intents: str):
    from app.llm.types import PlannedTask, TaskPlanResult

    tasks = [
        PlannedTask(
            intent=intent,
            text="",
            side_effect=(intent == "escalation"),
            action_type="open_support_case" if intent == "escalation" else "none",
        )
        for intent in intents
    ]
    return patch(
        "app.llm.intent_planner.IntentPlanner.plan",
        AsyncMock(return_value=TaskPlanResult(tasks=tasks)),
    )


def _mock_plan_action(action_type: str):
    from app.llm.types import PlannedTask, TaskPlanResult

    task = PlannedTask(
        intent="escalation",
        text="",
        side_effect=True,
        action_type=action_type,
    )
    return patch(
        "app.llm.intent_planner.IntentPlanner.plan",
        AsyncMock(return_value=TaskPlanResult(tasks=[task])),
    )


def _mock_qa_answer(response: str):
    return patch("app.llm.response_generator.QaResponseGenerator.generate", AsyncMock(return_value=response))


def _mock_policy_answer(response: str):
    return patch(
        "app.llm.response_generator.PolicyResponseGenerator.generate",
        AsyncMock(return_value=response),
    )


def _mock_task(order_id: str = ORDER_ID, category: str = "health_beauty"):
    from app.llm.types import OlistTaskResult

    result = OlistTaskResult(order_id=order_id, category=category, user_goal="support")
    return patch("app.llm.response_generator.OlistTaskExtractor.extract", AsyncMock(return_value=result))


def _operator_payload(message: str, session_id: str) -> dict[str, object]:
    return {
        "message": message,
        "session_id": session_id,
        "role": "after_sales_operator",
        "auth_scopes": ["*"],
        "channel": "review_console",
    }


def _support_payload(message: str, session_id: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "message": message,
        "role": "support_agent",
        "channel": "support_console",
    }
    if session_id:
        payload["session_id"] = session_id
    return payload


def _ops_payload(message: str, session_id: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "message": message,
        "role": "ops_manager",
        "channel": "ops_console",
    }
    if session_id:
        payload["session_id"] = session_id
    return payload


def test_chat_default_scopes_are_role_aware() -> None:
    default_request = ChatRequest(message="退款政策是什么？")
    assert default_request.role == "support_agent"
    assert default_request.channel == "internal_debug"

    customer_scopes = _resolve_auth_scopes("customer", None)
    support_scopes = _resolve_auth_scopes("support_agent", None)
    ops_scopes = _resolve_auth_scopes("ops_manager", None)
    operator_scopes = _resolve_auth_scopes("after_sales_operator", None)

    assert "orders:read" in customer_scopes
    assert "after_sales:draft" in customer_scopes
    assert "analytics:read" not in customer_scopes
    assert "after_sales:write" not in customer_scopes
    assert "after_sales:draft" in support_scopes
    assert "after_sales:write" not in support_scopes
    assert "after_sales:ops_report" in ops_scopes
    assert "after_sales:write" not in ops_scopes
    assert "after_sales:write" in operator_scopes
    assert _resolve_auth_scopes("after_sales_operator", []) == []


@pytest.mark.anyio
async def test_customer_order_access_guard_blocks_other_customer_order(client: AsyncClient) -> None:
    response = await client.post(
        "/customer/chat",
        json={
            "message": "帮我查一下订单 8aec3a066f732dd927ec8fef1752415b 的状态",
            "session_id": "unauthorized-order",
        },
        headers={"X-Demo-Customer": "customer-smoke"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "只能处理当前账号名下的订单" in body["answer"]
    assert body["session_id"] == "unauthorized-order"


@pytest.mark.anyio
async def test_customer_context_is_resolved_server_side_and_minimized(client: AsyncClient) -> None:
    response = await client.get("/customer/context", headers={"X-Demo-Customer": "customer-smoke"})

    assert response.status_code == 200
    body = response.json()
    assert body["customer"]["id"] == "customer-smoke"
    assert body["summary"]["order_count"] == 1
    assert [item["order_id"] for item in body["orders"]] == [ORDER_ID]
    assert "customer_state" not in body["orders"][0]
    assert body["orders"][0]["actions"][0]["id"] == "track_order"


@pytest.mark.anyio
async def test_customer_order_controls_follow_order_state(client: AsyncClient) -> None:
    response = await client.get("/customer/context")
    assert response.status_code == 200
    orders = {item["order_id"]: item for item in response.json()["orders"]}

    delivered_actions = {action["id"] for action in orders[ORDER_ID]["actions"]}
    pre_shipment_actions = {
        action["id"] for action in orders["8aec3a066f732dd927ec8fef1752415b"]["actions"]
    }
    canceled_actions = {
        action["id"] for action in orders["1b9ecfe83cdc259250e1a8aca174f0ad"]["actions"]
    }

    assert {"refund_request", "complaint_escalation"} <= delivered_actions
    assert "cancel_order" not in delivered_actions
    assert {"cancel_order", "change_address"} <= pre_shipment_actions
    assert "cancel_order" not in canceled_actions
    assert "refund_request" in canceled_actions


@pytest.mark.anyio
async def test_customer_structured_order_action_skips_planner_and_uses_page_context(
    client: AsyncClient,
) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, patch(
        "app.llm.intent_planner.IntentPlanner.plan",
        AsyncMock(side_effect=AssertionError("native action must not call planner")),
    ):
        response = await client.post(
            "/customer/chat",
            json={
                "message": "帮我查看这笔订单的物流进展。",
                "session_id": "native-track-order",
                "requested_action": "track_order",
                "page_context": {"surface": "order_detail", "selected_order_id": ORDER_ID},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert "已送达" in body["answer"]
    assert body["ui_actions"]


@pytest.mark.anyio
async def test_customer_structured_overview_lists_owned_active_orders(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, patch(
        "app.llm.intent_planner.IntentPlanner.plan",
        AsyncMock(side_effect=AssertionError("overview control must not call planner")),
    ):
        response = await client.post(
            "/customer/chat",
            json={
                "message": "我有哪些订单还在运输中？",
                "session_id": "native-active-orders",
                "requested_action": "list_active_shipments",
            },
        )

    assert response.status_code == 200
    assert "还在处理中" in response.json()["answer"]


@pytest.mark.anyio
async def test_customer_order_summary_uses_owned_orders_without_planner(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2:
        response = await client.post(
            "/customer/chat",
            json={"message": "我喜欢什么样的产品？", "session_id": "customer-order-summary"},
        )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "从您最近的" in answer
    assert "健康与美容用品" in answer
    assert "查看在途订单" in answer
    assert "[订单小结]" not in answer


@pytest.mark.anyio
async def test_customer_new_turn_clears_prior_answer_buffers(client: AsyncClient) -> None:
    """A follow-up must keep conversational context without reusing old output."""

    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2:
        profile = await client.post(
            "/customer/chat",
            json={"message": "我的购物习惯是什么？", "session_id": "customer-clean-turn"},
        )
        shipments = await client.post(
            "/customer/chat",
            json={
                "message": "我有哪些订单还在运输中？",
                "session_id": "customer-clean-turn",
                "requested_action": "list_active_shipments",
            },
        )

    assert profile.status_code == 200
    assert "买过" in profile.json()["answer"]
    assert shipments.status_code == 200
    answer = shipments.json()["answer"]
    assert "还在处理中" in answer
    assert "从您最近的" not in answer
    assert "[政策问答]" not in answer
    assert "[我的订单]" not in answer


@pytest.mark.anyio
async def test_customer_rejects_unowned_selected_order_context(client: AsyncClient) -> None:
    response = await client.post(
        "/customer/chat",
        json={
            "message": "查看物流",
            "requested_action": "track_order",
            "page_context": {
                "surface": "order_detail",
                "selected_order_id": "8aec3a066f732dd927ec8fef1752415b",
            },
        },
        headers={"X-Demo-Customer": "customer-smoke"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "selected_order_owner_required"


@pytest.mark.anyio
async def test_customer_session_and_case_endpoints_enforce_server_resolved_owner(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        created = await client.post(
            "/customer/chat",
            json={
                "message": f"给订单 {ORDER_ID} 申请退款",
                "session_id": "owner-bound-customer-session",
            },
        )
    assert created.status_code == 200

    progress = await client.get("/customer/sessions/brand-new-browser-session/progress")
    assert progress.status_code == 200
    assert progress.json() == {"case": None}

    messages = await client.get(
        "/customer/sessions/owner-bound-customer-session/messages",
        headers={"X-Demo-Customer": "customer-smoke"},
    )
    assert messages.status_code == 403
    assert messages.json()["detail"] == "conversation_owner_required"

    review = await client.get("/review/cases", headers=REVIEW_HEADERS)
    case_id = next(
        item["case_id"]
        for item in review.json()["cases"]
        if item["session_id"] == "owner-bound-customer-session"
    )
    customer_case = await client.get(
        f"/customer/cases/{case_id}",
        headers={"X-Demo-Customer": "customer-smoke"},
    )
    assert customer_case.status_code == 403


@pytest.mark.anyio
async def test_qa_category_insights(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("qa"), _mock_qa_answer("LLM: health_beauty has delay and review risk"):
        response = await client.post("/chat", json=_support_payload("health_beauty 类目有什么运营风险？"))
    assert response.status_code == 200
    assert "health_beauty" in response.json()["answer"]
    assert response.json()["sources"] == ["health_beauty"]


@pytest.mark.anyio
async def test_after_sales_ops_decision_report(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("ops_decision"):
        response = await client.post(
            "/chat",
            json=_ops_payload("生成审核台优先处理队列，列出最需要人工跟进的类目和订单"),
        )
    assert response.status_code == 200
    body = response.json()
    assert "[审核台优先处理]" in body["answer"]
    assert "高风险类目" in body["answer"]
    assert "优先跟进订单" in body["answer"]
    assert "HITL" in body["answer"]
    assert body["sources"]


@pytest.mark.anyio
async def test_offline_after_sales_ops_decision_report(client: AsyncClient) -> None:
    response = await client.post(
        "/chat",
        json=_ops_payload("生成审核台优先处理队列，列出最需要人工跟进的类目和订单"),
    )
    assert response.status_code == 200
    assert "[审核台优先处理]" in response.json()["answer"]


@pytest.mark.anyio
async def test_order_status_lookup(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("order_status"), _mock_task():
        response = await client.post("/chat", json={"message": f"帮我查订单 {ORDER_ID} 状态"})
    assert response.status_code == 200
    body = response.json()["answer"]
    assert ORDER_ID in body
    assert "delivered" in body
    assert "延迟 11 天" in body
    assert "评价分：2" in body


@pytest.mark.anyio
async def test_policy_question_uses_policy_knowledge(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("policy"), _mock_policy_answer("LLM: compensation needs human approval"):
        response = await client.post("/chat", json={"message": "退款补偿能不能直接承诺？"})
    assert response.status_code == 200
    # /chat is the internal/debug surface. Customer-facing policy language is
    # applied by /customer/chat after the workflow result is grounded.
    assert "[政策问答]" in response.json()["answer"]
    assert "LLM: compensation needs human approval" in response.json()["answer"]
    assert "Compensation Boundary Policy" in response.json()["sources"]


@pytest.mark.anyio
async def test_escalation_first_turn_requires_confirmation(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        response = await client.post(
            "/chat",
            json=_support_payload(f"订单 {ORDER_ID} 延迟且低分，生成客服跟进话术", "s1"),
        )
    assert response.status_code == 200
    assert "创建售后工单" in response.json()["answer"]
    assert "是否确认执行" in response.json()["answer"]


@pytest.mark.anyio
async def test_default_support_agent_cannot_execute_confirmed_side_effect(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/chat",
            json=_support_payload(f"订单 {ORDER_ID} 延迟且低分，生成客服跟进话术", "s-deny"),
        )
    with g1, g2:
        response = await client.post("/chat", json=_support_payload("yes", "s-deny"))

    assert response.status_code == 200
    assert "未执行创建售后工单" in response.json()["answer"]
    assert "permission_denied" in response.json()["answer"] or "not allowed" in response.json()["answer"]


@pytest.mark.anyio
async def test_customer_escalation_creates_staff_review_without_internal_confirmation(
    client: AsyncClient,
) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        response = await client.post(
            "/customer/chat",
            json={
                "message": f"给订单 {ORDER_ID} 申请退款",
                "session_id": "customer-review-s1",
            },
        )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "审核" in answer
    assert "是否确认执行" not in answer
    assert "Verifier" not in answer

    review = await client.get("/review/sessions/customer-review-s1", headers=REVIEW_HEADERS)
    assert review.status_code == 200
    body = review.json()
    assert body["has_pending"] is True
    assert body["pending_side_effect"]["requires_confirmation"] is True


@pytest.mark.anyio
async def test_review_endpoints_require_token_and_after_sales_role(client: AsyncClient) -> None:
    missing_token = await client.get("/review/sessions/any-session")
    assert missing_token.status_code == 403

    customer_role = await client.get(
        "/review/sessions/any-session",
        params={"role": "customer", "auth_scopes": "after_sales:write"},
        headers=REVIEW_HEADERS,
    )
    assert customer_role.status_code == 403

    no_write_scope = await client.post(
        "/review/sessions/any-session/approve",
        headers=REVIEW_HEADERS,
        json={
            "reviewer_id": "reviewer-1",
            "role": "after_sales_operator",
            "auth_scopes": ["orders:read"],
        },
    )
    assert no_write_scope.status_code == 403


@pytest.mark.anyio
async def test_customer_yes_cannot_approve_pending_write_action(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/customer/chat",
            json={
                "message": f"给订单 {ORDER_ID} 申请退款",
                "session_id": "customer-no-approve",
            },
        )

    with g1, g2:
        response = await client.post(
            "/customer/chat",
            json={"message": "yes", "session_id": "customer-no-approve"},
        )

    assert response.status_code == 200
    assert "工作人员正在核查" in response.json()["answer"]
    assert "已执行" not in response.json()["answer"]


@pytest.mark.anyio
async def test_review_queue_reject_and_customer_appeal_flow(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        created = await client.post(
            "/customer/chat",
            json={
                "message": f"给订单 {ORDER_ID} 申请退款",
                "session_id": "customer-appeal-s1",
            },
        )
    assert created.status_code == 200

    queue = await client.get("/review/cases", headers=REVIEW_HEADERS)
    assert queue.status_code == 200
    cases = queue.json()["cases"]
    assert cases
    case_id = cases[0]["case_id"]
    assert cases[0]["status"] == "pending_review"

    rejected = await client.post(
        "/review/sessions/customer-appeal-s1/reject",
        headers=REVIEW_HEADERS,
        json={
            "reviewer_id": "reviewer-appeal",
            "role": "after_sales_operator",
            "auth_scopes": ["after_sales:write"],
        },
    )
    assert rejected.status_code == 200

    status = await client.get(f"/customer/cases/{case_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "rejected"

    appeal = await client.post(
        f"/customer/cases/{case_id}/appeal",
        json={"reason": "我补充了物流延迟证明，请重新审核。"},
    )
    assert appeal.status_code == 200
    assert appeal.json()["status"] == "appealed_pending_review"
    assert appeal.json()["appeal_count"] == 1

    queue_after_appeal = await client.get("/review/cases", headers=REVIEW_HEADERS)
    assert any(item["case_id"] == case_id for item in queue_after_appeal.json()["cases"])


@pytest.mark.anyio
async def test_staff_review_endpoint_approves_customer_case(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/customer/chat",
            json={
                "message": f"给订单 {ORDER_ID} 申请退款",
                "session_id": "staff-approve-s1",
            },
        )

    approved = await client.post(
        "/review/sessions/staff-approve-s1/approve",
        headers=REVIEW_HEADERS,
        json={
            "reviewer_id": "reviewer-1",
            "role": "after_sales_operator",
            "auth_scopes": ["after_sales:write"],
        },
    )
    assert approved.status_code == 200
    assert "已执行创建售后工单" in approved.json()["answer"]
    assert "CASE-" in approved.json()["answer"]

    review = await client.get("/review/sessions/staff-approve-s1", headers=REVIEW_HEADERS)
    assert review.status_code == 200
    assert review.json()["has_pending"] is False


@pytest.mark.anyio
async def test_escalation_second_turn_confirm(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/chat",
            json=_operator_payload(f"订单 {ORDER_ID} 延迟且低分，生成客服跟进话术", "s2"),
        )
    with g1, g2:
        response = await client.post("/chat", json=_operator_payload("yes", "s2"))
    assert response.status_code == 200
    assert "已执行创建售后工单" in response.json()["answer"]
    assert "CASE-" in response.json()["answer"]


@pytest.mark.anyio
async def test_confirmed_hitl_session_can_accept_new_policy_task(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/chat",
            json=_operator_payload(f"订单 {ORDER_ID} 延迟且低分，生成客服跟进话术", "s2-next"),
        )
    with g1, g2:
        confirm_response = await client.post("/chat", json=_operator_payload("yes", "s2-next"))
    assert confirm_response.status_code == 200

    with g1, g2, _mock_plan("policy"), _mock_policy_answer("LLM: compensation requires approval"):
        response = await client.post(
            "/chat",
            json=_support_payload("退款补偿能不能直接承诺？", "s2-next"),
        )
    assert response.status_code == 200
    assert "[政策问答]" in response.json()["answer"]
    assert "order_id" not in response.json()["answer"]


@pytest.mark.anyio
async def test_escalation_second_turn_cancel(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/chat",
            json=_support_payload(f"订单 {ORDER_ID} 延迟且低分，生成客服跟进话术", "s3"),
        )
    with g1, g2:
        response = await client.post("/chat", json=_support_payload("no", "s3"))
    assert response.status_code == 200
    assert "售后申请已被审核拒绝" in response.json()["answer"]


@pytest.mark.anyio
async def test_escalation_confirmation_timeout_cancels_pending_action(
    client: AsyncClient,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HITL_TIMEOUT_SECONDS", "1")
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/chat",
            json=_support_payload(f"订单 {ORDER_ID} 延迟且低分，生成客服跟进话术", "timeout-s1"),
        )

    await asyncio.sleep(1.1)
    with g1, g2:
        response = await client.post("/chat", json=_support_payload("yes", "timeout-s1"))
    assert response.status_code == 200
    assert "已超时自动取消" in response.json()["answer"]
    assert "CASE-" not in response.json()["answer"]


@pytest.mark.anyio
async def test_pending_hilt_does_not_consume_unrelated_message(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/chat",
            json=_support_payload(f"订单 {ORDER_ID} 延迟且低分，生成客服跟进话术", "pending-s1"),
        )
    with g1, g2:
        response = await client.post(
            "/chat",
            json=_support_payload("health beauty 类目有什么运营风险？", "pending-s1"),
        )
    assert response.status_code == 200
    assert "待确认" in response.json()["answer"]
    assert "换一个 Session ID" in response.json()["answer"]


@pytest.mark.anyio
async def test_off_topic_rejected_by_guardrail(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=False)
    with g1, g2:
        response = await client.post("/chat", json={"message": "帮我写一个操作系统内核"})
    assert response.status_code == 200
    assert "only help" in response.json()["answer"]


@pytest.mark.anyio
async def test_chat_rate_limit_uses_runtime_store(
    client: AsyncClient,
    monkeypatch,
) -> None:
    import app.main as main_module
    from app.tool_call import InMemoryRuntimeStore

    monkeypatch.setattr(main_module, "runtime_store", InMemoryRuntimeStore())
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MINUTE", "1")
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("policy"), _mock_policy_answer("LLM: policy answer"):
        first = await client.post("/chat", json={"message": "退款政策是什么？", "session_id": "rate-s1"})
        second = await client.post("/chat", json={"message": "退款政策是什么？", "session_id": "rate-s2"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert "请求过于频繁" in second.json()["answer"]


@pytest.mark.anyio
async def test_empty_message_rejected_by_pydantic(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_session_id_preserved(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("qa"), _mock_qa_answer("LLM: hello"):
        response = await client.post("/chat", json={"message": "Hi", "session_id": "my-session-123"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "my-session-123"


@pytest.mark.anyio
async def test_multi_task_plan_executes_read_only_tasks_before_escalation(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with (
        g1,
        g2,
        _mock_plan("order_status", "policy", "escalation"),
        _mock_task(),
        _mock_policy_answer("LLM: refund policy says confirm first"),
    ):
        response = await client.post(
            "/chat",
            json=_support_payload(
                f"查订单 {ORDER_ID} 状态，并且说明退款政策，然后申请退款",
                "multi-task",
            ),
        )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "[订单查询]" in answer
    assert "[政策问答]" in answer
    assert "[售后升级]" in answer
    assert "是否确认执行" in answer


@pytest.mark.anyio
async def test_offline_multi_intent_escalation_inherits_order_context(client: AsyncClient) -> None:
    response = await client.post(
        "/chat",
        json=_support_payload(
            f"查订单 {ORDER_ID} 状态，并且说明退款政策，然后申请退款",
            "offline-context-carry",
        ),
    )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "[订单查询]" in answer
    assert "[政策问答]" in answer
    assert "[售后升级]" in answer
    assert "是否确认执行" in answer
    assert "没有检测到有效 order_id" not in answer


@pytest.mark.anyio
async def test_executor_defers_side_effect_until_read_only_tasks_finish(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with (
        g1,
        g2,
        _mock_plan("escalation", "policy"),
        _mock_task(),
        _mock_policy_answer("LLM: refund policy says confirm first"),
    ):
        response = await client.post(
            "/chat",
            json=_support_payload(
                f"给订单 {ORDER_ID} 申请退款，并且说明退款政策",
                "side-effect-ordering",
            ),
        )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer.index("[政策问答]") < answer.index("[售后升级]")
    assert "是否确认执行" in answer


@pytest.mark.anyio
async def test_refund_side_effect_uses_refund_tool(client: AsyncClient) -> None:
    from app.llm.types import PlannedTask, TaskPlanResult

    g1, g2 = _mock_guard(input_on_topic=True)
    plan = TaskPlanResult(
        tasks=[
            PlannedTask(
                intent="escalation",
                text=f"给订单 {ORDER_ID} 申请退款",
                side_effect=True,
                action_type="refund_request",
            )
        ]
    )
    with (
        g1,
        g2,
        patch("app.llm.intent_planner.IntentPlanner.plan", AsyncMock(return_value=plan)),
        _mock_task(),
    ):
        await client.post(
            "/chat",
            json=_operator_payload(f"给订单 {ORDER_ID} 申请退款", "refund-task"),
        )
    with g1, g2:
        response = await client.post("/chat", json=_operator_payload("确认", "refund-task"))
    assert response.status_code == 200
    assert "REFUND-" in response.json()["answer"]


@pytest.mark.anyio
async def test_delivered_order_cancel_is_rejected_before_hitl(client: AsyncClient) -> None:
    from app.llm.types import PlannedTask, TaskPlanResult

    g1, g2 = _mock_guard(input_on_topic=True)
    plan = TaskPlanResult(
        tasks=[
            PlannedTask(
                intent="escalation",
                text=f"取消订单 {ORDER_ID}",
                side_effect=True,
                action_type="cancel_order",
            )
        ]
    )
    with (
        g1,
        g2,
        patch("app.llm.intent_planner.IntentPlanner.plan", AsyncMock(return_value=plan)),
        _mock_task(),
    ):
        response = await client.post(
            "/chat",
            json=_support_payload(f"取消订单 {ORDER_ID}", "delivered-cancel"),
        )

    answer = response.json()["answer"]
    assert "售后 case 决策" in answer
    assert "结论：reject" in answer
    assert "本轮不会执行" in answer
    assert "是否确认执行" not in answer


@pytest.mark.anyio
async def test_multiple_escalations_continue_after_first_confirmation(client: AsyncClient) -> None:
    from app.llm.types import PlannedTask, TaskPlanResult

    g1, g2 = _mock_guard(input_on_topic=True)
    plan = TaskPlanResult(
        tasks=[
            PlannedTask(
                intent="escalation",
                text=f"给订单 {ORDER_ID} 申请退款",
                side_effect=True,
                action_type="refund_request",
            ),
            PlannedTask(
                intent="escalation",
                text=f"给订单 {ORDER_ID} 申请发票",
                side_effect=True,
                action_type="invoice_request",
            ),
        ]
    )
    with (
        g1,
        g2,
        patch("app.llm.intent_planner.IntentPlanner.plan", AsyncMock(return_value=plan)),
        _mock_task(),
    ):
        first = await client.post(
            "/chat",
            json=_operator_payload(f"给订单 {ORDER_ID} 申请退款，再申请发票", "two-effects"),
        )
    assert first.status_code == 200
    assert "提交退款/补偿申请" in first.json()["answer"]

    with g1, g2:
        second = await client.post("/chat", json=_operator_payload("yes", "two-effects"))
    assert second.status_code == 200
    assert "已执行提交退款/补偿申请" in second.json()["answer"]
    assert "提交发票申请" in second.json()["answer"]
    assert "是否确认执行" in second.json()["answer"]

    with g1, g2:
        third = await client.post("/chat", json=_operator_payload("yes", "two-effects"))
    assert third.status_code == 200
    assert "已执行提交发票申请" in third.json()["answer"]
    assert "INV-" in third.json()["answer"]


@pytest.mark.anyio
async def test_chat_trace_contains_trajectory_events(client: AsyncClient, tmp_path, monkeypatch) -> None:
    import app.trace_store as trace_store
    from app.trace_store import list_session_traces

    monkeypatch.setattr(trace_store, "TRACE_DB", tmp_path / "agent_traces.db")
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("policy"), _mock_policy_answer("LLM: policy answer"):
        response = await client.post("/chat", json={"message": "退款政策是什么？", "session_id": "trace-s1"})

    assert response.status_code == 200
    traces = list_session_traces("trace-s1")
    assert traces
    assert "plan_tasks" in traces[0]["trajectory_json"]
    assert "search_policy_knowledge" in traces[0]["trajectory_json"]

    denied = await client.get("/observability/traces/trace-s1")
    assert denied.status_code == 403
    replay = await client.get("/observability/traces/trace-s1", headers=REVIEW_HEADERS)
    assert replay.status_code == 200
    assert replay.json()["traces"]


@pytest.mark.anyio
async def test_observability_case_metrics_endpoint(client: AsyncClient, tmp_path, monkeypatch) -> None:
    import app.trace_store as trace_store

    monkeypatch.setattr(trace_store, "TRACE_DB", tmp_path / "traces.db")
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan("escalation"), _mock_task():
        await client.post(
            "/chat",
            json={"message": f"给订单 {ORDER_ID} 申请退款", "session_id": "metrics-s1"},
        )

    denied = await client.get("/observability/case-metrics")
    assert denied.status_code == 403

    response = await client.get("/observability/case-metrics", headers=REVIEW_HEADERS)
    body = response.json()
    assert response.status_code == 200
    assert body["total_cases"] == 1
    assert body["hitl_rate"] == 1.0
    assert body["policy_hit_rate"] == 1.0


@pytest.mark.anyio
async def test_review_metrics_requires_staff_token(client: AsyncClient) -> None:
    denied = await client.get("/review/metrics")
    assert denied.status_code == 403

    response = await client.get("/review/metrics", headers=REVIEW_HEADERS)
    assert response.status_code == 200
    body = response.json()
    baselines = body["product_evaluation"]["baselines"]
    current = baselines["heldout_current"]
    assert (current["requests_total"], current["cases_total"], current["policy_floor"]) == (99, 52, 26)
    assert current["wrong_executions"] == current["high_risk_auto"] == current["leaks"] == 0


@pytest.mark.anyio
async def test_chat_complaint_escalation_enters_after_sales_case_lifecycle(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_plan_action("complaint_escalation"), _mock_task():
        response = await client.post(
            "/chat",
            json=_operator_payload(f"我要投诉升级订单 {ORDER_ID} 的延迟问题", "complaint-s1"),
        )

    body = response.json()
    assert response.status_code == 200
    assert "售后 case 决策" in body["answer"]
    assert "提交投诉升级工单" in body["answer"]
    assert "是否确认执行" in body["answer"]

    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2:
        confirmed = await client.post("/chat", json=_operator_payload("yes", "complaint-s1"))
    assert confirmed.status_code == 200
    assert "COMP-" in confirmed.json()["answer"]


def test_customer_policy_questions_do_not_use_order_tool_agent() -> None:
    context = CustomerPageContext()
    for message in (
        "退款一般需要满足什么条件？",
        "退款政策是什么？",
        "退款需要多久到账？",
        "改地址有什么规则？",
    ):
        assert _should_use_customer_order_agent(message, context) is False
    assert _should_use_customer_order_agent("我有哪些订单还在运输中？", context) is True
    assert _should_use_customer_order_agent("帮我查物流进展", context) is True

    assert _customer_order_clarification(
        "demo-customer", "退款一般需要满足什么条件？", CustomerPageContext()
    ) is None
