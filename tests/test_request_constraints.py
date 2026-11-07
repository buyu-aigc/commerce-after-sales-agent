"""Customer intent constraints must hold before any Case or business write."""

from unittest.mock import AsyncMock, patch

import pytest

from app.after_sales.complexity import assess_request
from app.llm.customer_presenter import _customer_order_status
from app.llm.types import PlannedTask, TaskPlanResult
from tests import test_v2_conversation_case_decoupling as acceptance

v2_runtime = acceptance.v2_runtime
client = acceptance.client
A = acceptance.A
B = acceptance.B


@pytest.mark.parametrize("text", [
    f"订单 {A} 我不申请退款，只查询物流状态",
    f"先查询订单 {A} 的物流，别替订单 {A} 提交退款申请",
    f"订单 {A} 不要退款和取消订单",
    f"订单 {A} 不要退款以及取消订单",
    f"订单 {A} 不用退款、取消订单",
    f"订单 {A} 不要退款或者订单 {B} 退款",
    f"订单 {A} 不想取消退款申请",
    f"订单 {A} 不是不退款",  # Double negation is ambiguous: do not write.
    f"订单 {A} 不能取消，只查物流",
    f"order {A} do not request a refund and cancel the order",
    f"order {A} I don't want to cancel the order",
])
def test_prohibitions_never_become_write_units(text):
    result = assess_request(text)
    assert result.write_units == []
    assert result.restricted_actions


@pytest.mark.parametrize("text,expected", [
    (f"订单 {A} 不退款但要开发票", [(A, "invoice_request")]),
    (f"订单 {A} 不要退款改为开发票", [(A, "invoice_request")]),
    (f"订单 {A} 开具发票，不是退款，也不是取消订单", [(A, "invoice_request")]),
    (f"订单 {A} 不要退款，订单 {B} 晚到我要退款", [(B, "refund_request")]),
    (f"订单 {A} 不要退款 订单 {B} 申请退款", [(B, "refund_request")]),
    (f"订单 {A} 不要退款，取消订单", [(A, "cancel_order")]),
    (f"订单 {A} 没收到货我要退款", [(A, "refund_request")]),
    (f"订单 {A} 商品不好用想退款", [(A, "refund_request")]),
    (f"订单 {A} 地址改为上海市浦东新区世纪大道100号", [(A, "change_address")]),
])
def test_negation_is_scoped_and_preserves_affirmative_requests(text, expected):
    result = assess_request(text)
    assert [(unit.order_id, unit.action_type) for unit in result.write_units] == expected


@pytest.mark.parametrize("text", [
    f"订单 {A} 查退款状态",
    f"请帮我查订单 {A} 的退款进度",
    f"我要查询订单 {A} 取消了吗",
    f"订单 {A} 不退款，只查退款进度",
])
def test_status_questions_are_reads_even_when_they_name_a_write_action(text):
    result = assess_request(text)
    assert result.write_units == []
    assert any(unit.intent == "order_status" for unit in result.read_units)


def test_prohibited_action_is_not_counted_as_refund_evidence():
    result = assess_request(f"订单 {A} 退款，不是投诉")
    assert len(result.write_units) == 1
    refund = result.write_units[0]
    assert refund.action_type == "refund_request"
    assert "refund_reason" in refund.missing_evidence
    assert "投诉" not in refund.text


def test_request_constraints_do_not_rewrite_address_evidence():
    result = assess_request(
        f"订单 {A} 改地址，收件人但明，电话13800138000，新地址上海市世纪大道100号"
    )
    assert len(result.write_units) == 1
    address = result.write_units[0]
    assert address.action_type == "change_address"
    assert "收件人但明" in address.text
    assert "13800138000" in address.text


@pytest.mark.parametrize("selected,ui_action", [(False, None), (True, None), (True, "refund_request")])
@pytest.mark.anyio
async def test_customer_query_prohibition_overrides_planner_and_ui_shortcuts(
    client, v2_runtime, selected, ui_action
):
    task = PlannedTask(
        intent="escalation", side_effect=True, action_type="refund_request", text=f"订单 {A} 退款"
    )
    with acceptance._guard()[0], acceptance._guard()[1], patch(
        "app.llm.intent_planner.IntentPlanner.plan", AsyncMock(return_value=TaskPlanResult(tasks=[task]))
    ) as planner:
        body = {"session_id": "negative-query", "message": f"订单 {A} 我不申请退款，只查询物流状态"}
        if selected:
            body["page_context"] = {"selected_order_id": A}
        if ui_action:
            body["requested_action"] = ui_action
        response = await client.post("/customer/chat", json=body)
    assert response.status_code == 200, response.text
    planner.assert_not_awaited()
    assert not v2_runtime["case_service"].list_by_session("negative-query")
    assert v2_runtime["case_service"].count_order_events(A) == 0
    snapshot = await v2_runtime["main"].conversation_agent.aget_state(
        {"configurable": {"thread_id": "negative-query"}}
    )
    assert [task["intent"] for task in snapshot.values["task_plan"]] == ["order_status"]
    assert snapshot.values["active_order_id"] == A


@pytest.mark.anyio
async def test_negating_other_actions_keeps_invoice_case_only(client, v2_runtime):
    with acceptance._guard()[0], acceptance._guard()[1]:
        response = await client.post("/customer/chat", json={
            "session_id": "invoice-only",
            "message": f"订单 {A} 要开具发票，不是退款，也不是取消订单",
        })
    assert response.status_code == 200, response.text
    records = v2_runtime["case_service"].list_by_session("invoice-only")
    assert [(r["order_id"], r["action_type"], r["status"]) for r in records] == [
        (A, "invoice_request", "pending_review")
    ]
    assert v2_runtime["case_service"].count_order_events(A) == 0


@pytest.mark.anyio
async def test_negative_order_does_not_block_positive_order(client, v2_runtime, monkeypatch):
    views = {
        order: acceptance._fake_view(order, status="delivered", amount=500, delay=11) for order in (A, B)
    }
    monkeypatch.setattr(v2_runtime["olist_service"], "get_order_status", views.get)
    with acceptance._guard()[0], acceptance._guard()[1]:
        response = await client.post("/customer/chat", json={
            "session_id": "negative-a-positive-b",
            "message": f"订单 {A} 不要退款，订单 {B} 因迟到申请退款",
        })
    assert response.status_code == 200, response.text
    records = v2_runtime["case_service"].list_by_session("negative-a-positive-b")
    assert [(r["order_id"], r["action_type"], r["status"]) for r in records] == [
        (B, "refund_request", "pending_review")
    ]
    assert all(v2_runtime["case_service"].count_order_events(order) == 0 for order in (A, B))


@pytest.mark.anyio
async def test_new_turn_can_explicitly_request_previously_prohibited_action(client, v2_runtime, monkeypatch):
    view = acceptance._fake_view(A, status="delivered", amount=500, delay=11)
    monkeypatch.setattr(
        v2_runtime["olist_service"], "get_order_status", lambda order: view if order == A else None
    )
    with acceptance._guard()[0], acceptance._guard()[1]:
        first = await client.post("/customer/chat", json={
            "session_id": "changed-mind", "message": f"订单 {A} 不申请退款，只查物流",
        })
        assert first.status_code == 200
        assert not v2_runtime["case_service"].list_by_session("changed-mind")
        second = await client.post("/customer/chat", json={
            "session_id": "changed-mind", "message": f"现在订单 {A} 因迟到申请退款",
        })
    assert second.status_code == 200, second.text
    records = v2_runtime["case_service"].list_by_session("changed-mind")
    assert [(r["action_type"], r["status"]) for r in records] == [("refund_request", "pending_review")]
    assert v2_runtime["case_service"].count_order_events(A) == 0


@pytest.mark.anyio
async def test_implicit_address_wording_still_reaches_the_semantic_planner(client, v2_runtime):
    message = f"我要把订单 {B} 地址调整为上海市浦东新区世纪大道100号"
    task = PlannedTask(intent="escalation", side_effect=True, action_type="change_address", text=message)
    with acceptance._guard()[0], acceptance._guard()[1], patch(
        "app.llm.intent_planner.IntentPlanner.plan", AsyncMock(return_value=TaskPlanResult(tasks=[task]))
    ) as planner:
        response = await client.post(
            "/customer/chat", json={"session_id": "implicit-address", "message": message}
        )
    assert response.status_code == 200, response.text
    planner.assert_awaited_once()
    snapshot = await v2_runtime["main"].conversation_agent.aget_state(
        {"configurable": {"thread_id": "implicit-address"}}
    )
    assert snapshot.values["task_plan"][0]["intent"] == "escalation"
    assert snapshot.values["task_plan"][0]["action_type"] == "change_address"


@pytest.mark.parametrize("status,label", [
    ("canceled", "已取消"), ("shipped", "运输中"), ("invoiced", "待发货"), ("processing", "处理中")
])
def test_current_status_wins_over_stale_delivered_timestamp(status, label):
    reply = _customer_order_status(
        f"当前状态：{status}。实际送达：2018-03-21 22:03:51；延迟 12 天。"
    )
    assert label in reply
    assert "已送达" not in reply
    if status in {"canceled", "invoiced", "processing"}:
        assert "晚了" not in reply


def test_delivered_status_still_reports_its_delay():
    reply = _customer_order_status("当前状态：delivered。实际送达：2018-03-21；延迟 12 天。")
    assert "已送达" in reply and "12 天" in reply


@pytest.mark.parametrize("timestamp", ["2018-03-21", "nan", "None", "未送达"])
def test_timestamp_without_current_status_does_not_assert_delivery(timestamp):
    assert "已送达" not in _customer_order_status(f"实际送达：{timestamp}。")
