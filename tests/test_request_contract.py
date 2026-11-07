from unittest.mock import AsyncMock, patch

import pytest

from app.after_sales.address import address_fields, missing_address_fields
from app.after_sales.complexity import assess_request
from app.after_sales.decision import AfterSalesDecisionEngine
from app.agent import semantic_order_grounding as semantic
from app.llm.customer_presenter import present
from app.llm.types import PlannedTask, TaskPlanResult
from tests import test_v2_conversation_case_decoupling as acceptance
from tests.test_order_discovery import A, B, order
from tests.test_semantic_order_grounding import SemanticAgent, decision

v2_runtime = acceptance.v2_runtime
client = acceptance.client


@pytest.mark.parametrize("negative", ["退款先不办了", "退款我暂时不申请", "退款不用处理", "退款不要"])
def test_postposed_denial_keeps_other_action(negative):
    result = assess_request(f"订单 {A} {negative}，先开发票")
    assert [(u.order_id, u.action_type) for u in result.write_units] == [(A, "invoice_request")]
    assert ("refund_request", A) in result.restricted_actions


def test_negative_reason_does_not_revoke_refund():
    result = assess_request(f"订单 {A} 我要退款，退款不影响我开发票")
    assert "refund_request" in result.action_types


@pytest.mark.parametrize("text", ["只查发货", "先查询配送", "Just check delivery, do not cancel it."])
def test_shipping_read_is_preserved_with_prohibition(text):
    result = assess_request(text, selected_order_id=A)
    assert not result.write_units
    assert any(u.intent == "order_status" for u in result.read_units)


@pytest.mark.parametrize("status", ["created", "approved", "invoiced", "processing"])
@pytest.mark.parametrize("action", ["cancel_order", "change_address"])
def test_pre_shipment_state_has_consistent_review_gate(status, action):
    view = acceptance._fake_view(A, status=status, amount=498.92, delay=0)
    case = AfterSalesDecisionEngine().assess(
        action_type=action, order=view, user_request="申请处理", policy_sections=[]
    )
    assert case.risk_signals["pre_shipment"]
    assert case.decision.outcome == "needs_human_review"
    assert case.decision.requires_human


@pytest.mark.parametrize("action", ["cancel_order", "change_address"])
def test_shipped_order_remains_blocked(action):
    view = acceptance._fake_view(A, status="shipped", amount=498.92, delay=0)
    case = AfterSalesDecisionEngine().assess(
        action_type=action, order=view, user_request="申请处理", policy_sections=[]
    )
    assert case.decision.outcome == "reject"


@pytest.mark.anyio
async def test_wrong_planner_and_ui_cannot_revive_postposed_refund(client, v2_runtime):
    bad = TaskPlanResult(
        tasks=[PlannedTask(intent="escalation", text="退款", side_effect=True, action_type="refund_request")]
    )
    with (
        acceptance._guard()[0],
        acceptance._guard()[1],
        patch("app.llm.intent_planner.IntentPlanner.plan", AsyncMock(return_value=bad)) as planner,
    ):
        response = await client.post(
            "/customer/chat",
            json={
                "session_id": "postposed",
                "message": f"订单 {A} 退款先不办了，我这次只要发票，抬头测试采购中心",
                "requested_action": "refund_request",
                "page_context": {"selected_order_id": A},
            },
        )
    assert response.status_code == 200
    planner.assert_not_awaited()
    assert [c["action_type"] for c in v2_runtime["case_service"].list_by_session("postposed")] == [
        "invoice_request"
    ]
    assert v2_runtime["case_service"].count_order_events(A) == 0


@pytest.mark.anyio
async def test_address_evidence_continues_only_confirmed_order():
    agent = SemanticAgent(
        [order(A, "电脑配件"), order(B, "家居用品")], decision(goal="address", request="request")
    )
    first = await semantic.ground(agent, "电脑配件这笔想换个收货地方", {}, {A, B}, [])
    assert first["context"]["address_pending"]["order_id"] == A
    second = await semantic.ground(
        agent, "收件人林禾，电话13800000001，送到上海市浦东新区世纪大道100号", first["context"], {A, B}, []
    )
    assert second["order_id"] == A and second["action"] == "change_address"
    assert not missing_address_fields(second["message"])
    assert not second["context"].get("address_pending")


def test_destination_does_not_invent_other_address_fields():
    fields = address_fields("电脑配件别取消，改到上海市浦东新区世纪大道100号")
    assert fields == {"address": "上海市浦东新区世纪大道100号"}
    assert missing_address_fields("改到上海市浦东新区世纪大道100号") == ["收件人", "联系电话"]


@pytest.mark.anyio
async def test_rejection_reply_uses_state_without_inventing_handoff():
    llm = type("Llm", (), {"ainvoke": AsyncMock()})()
    answer = await present(
        llm,
        "改地址",
        {"after_sales_cases": [{"order_status": "shipped", "decision": {"outcome": "reject"}}]},
        None,
    )
    assert "运输中" in answer and "没有提交" in answer
    llm.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("text", ["退款和取消都别办", "退款与开发票暂时不申请"])
def test_coordinated_postposed_prohibition_covers_all_objects(text):
    assert not assess_request(f"订单 {A} {text}").write_units


@pytest.mark.anyio
async def test_withdrawn_address_request_cannot_resume_from_evidence_alone():
    agent = SemanticAgent(
        [order(A, "电脑配件"), order(B, "家居用品")], decision(goal="address", request="request")
    )
    first = await semantic.ground(agent, "电脑配件想换个收货地方", {}, {A, B}, [])
    stopped = await semantic.ground(agent, "算了，不用了", first["context"], {A, B}, [])
    assert not stopped["context"].get("address_pending")
    resumed = await semantic.ground(
        agent, "收件人林禾，电话13800000001，送到上海市世纪大道100号", stopped["context"], {A, B}, []
    )
    assert not resumed.get("action")


@pytest.mark.anyio
async def test_compound_hypothetical_cannot_be_promoted_to_write_by_model():
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")], decision(goal="refund", request="request")
    )
    result = await semantic.ground(agent, "家居用品如果明天还没到就退款，今天先查物流", {}, {A, B}, [])
    assert not result.get("action") or result["action"] == "track_order"
    assert not result.get("order_id") or result.get("read_only")


@pytest.mark.parametrize(
    "text", ["配送政策一般怎么规定", "发货一般要哪些资料", "What is your delivery policy?"]
)
def test_general_shipping_policy_does_not_become_order_lookup(text):
    result = assess_request(text, selected_order_id=A)
    assert result.read_units and all(u.intent == "policy" for u in result.read_units)
    assert not semantic.applicable(text, {})


@pytest.mark.parametrize("text", ["我买的东西怎么一直没消息", "之前买的东西现在怎么样了", "快递怎么回事"])
def test_vague_order_question_still_discovers_orders(text):
    assert semantic.applicable(text, {})


@pytest.mark.parametrize("indices", [[1], [1, 2]])
@pytest.mark.anyio
async def test_policy_question_does_not_discard_explicit_complaint(client, v2_runtime, monkeypatch, indices):
    main = v2_runtime["main"]
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")],
        decision(indices=indices, goal="policy", request="question"),
    )
    monkeypatch.setattr(main, "customer_order_agent", agent)
    monkeypatch.setattr(main, "_allowed_customer_order_ids", lambda user: {A, B})
    with acceptance._guard()[0], acceptance._guard()[1]:
        response = await client.post(
            "/customer/chat",
            json={
                "session_id": "complaint-policy",
                "message": "家居用品配送晚到，我要投诉；另外，没发货的东西一般能不能取消",
            },
        )
    assert response.status_code == 200
    assert [c["action_type"] for c in v2_runtime["case_service"].list_by_session("complaint-policy")] == [
        "complaint_escalation"
    ]
    state = await main.conversation_agent.aget_state({"configurable": {"thread_id": "complaint-policy"}})
    assert any(task["intent"] == "policy" for task in state.values["task_plan"])
    assert v2_runtime["case_service"].count_order_events(A) == 0


@pytest.mark.parametrize(
    "question", ["我想知道能不能退款", "请帮我了解退款", "I want to know whether I can get a refund"]
)
def test_information_request_is_not_explicit_submission(question):
    assert not assess_request(f"订单 {A} {question}，再查配送状态").has_explicit_write_request


@pytest.mark.parametrize("text", ["第一个不退了", "第一个先别退", "第一个退款不要", "第一个退和取消都别办"])
@pytest.mark.anyio
async def test_colloquial_denial_cannot_authorize_refund_even_with_bad_model(text):
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")], decision(goal="refund", request="request")
    )
    result = await semantic.ground(agent, text, {}, {A, B}, [])
    assert result.get("action") != "refund_request"


def test_bare_refund_alias_never_adds_positive_write_units():
    assert not assess_request(f"订单 {A} 的退税问题").write_units
