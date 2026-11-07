from unittest.mock import AsyncMock

import pytest

from app.after_sales.complexity import assess_request
from app.agent import order_discovery as rules
from app.agent import semantic_order_grounding as semantic
from app.llm.customer_presenter import present
from tests import test_v2_conversation_case_decoupling as acceptance
from tests.test_order_discovery import A, B, order
from tests.test_semantic_order_grounding import SemanticAgent, decision

v2_runtime = acceptance.v2_runtime
client = acceptance.client


@pytest.mark.parametrize(
    "text,category",
    [
        ("健康与美容用品这笔，查一下是不是还在运输", "健康与美容用品"),
        ("电脑配件那笔，我现在申请取消，不是问规则", "电脑配件"),
    ],
)
def test_question_and_policy_denial_are_not_reference_denials(text, category):
    assert rules._selected(text, [order(A, category), order(B, "家居用品")], "") == A


@pytest.mark.parametrize("text", ["不是电脑配件", "不是第二个", "不选电脑配件", "电脑配件选错了"])
def test_actual_reference_denial_is_still_unbound(text):
    assert rules._selected(text, [order(A, "电脑配件"), order(B, "家居用品")], "") is None


@pytest.mark.anyio
async def test_case_progress_does_not_swallow_an_explicit_new_query():
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")],
        decision(indices=[1, 2], goal="case_progress", request="request"),
    )
    result = await semantic.ground(
        agent,
        "先让工作人员看退款，电脑配件只帮我查发没发",
        {},
        {A, B},
        [],
        cases=[{"status": "pending_review"}],
    )
    assert not result.get("answer")
    assert result["order_id"] == B
    assert "退款" not in result["message"]
    analysis = assess_request(result["message"], selected_order_id=B)
    assert not analysis.write_units
    assert [(u.intent, u.order_id) for u in analysis.read_units] == [("order_status", B)]


@pytest.mark.anyio
async def test_progress_query_does_not_infer_a_target_from_a_product_guess():
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")], decision(indices=[1, 2], goal="case_progress")
    )
    result = await semantic.ground(
        agent, "退款继续等着，查下键盘发没发", {}, {A, B}, [], cases=[{"status": "pending_review"}]
    )
    assert "order_id" not in result


@pytest.mark.anyio
async def test_only_case_progress_keeps_the_read_only_shortcut():
    agent = SemanticAgent([order(A)], decision(goal="case_progress"))
    result = await semantic.ground(
        agent, "我之前的退款审核怎么样了", {}, {A}, [], cases=[{"status": "pending_review"}]
    )
    assert "待审核" in result["answer"] and "order_id" not in result


@pytest.mark.parametrize("tail", ["分别是什么配送状态，只查一下", "都查一下物流状态"])
def test_shared_status_predicate_expands_distinct_targets(tail):
    result = assess_request(f"订单 {A} 家居用品和订单 {B} 健康与美容用品这两笔{tail}")
    assert [(u.intent, u.order_id) for u in result.read_units] == [("order_status", A), ("order_status", B)]
    assert not result.write_units


def compound_state():
    return {
        "task_plan": [
            {
                "intent": "escalation",
                "order_id": A,
                "action_type": "invoice_request",
                "side_effect": True,
                "text": "申请发票",
            },
            {"intent": "order_status", "order_id": B, "side_effect": False, "text": "查配送"},
        ],
        "customer_order_context": {"orders": [order(A, "家居用品"), order(B, "电脑配件")]},
        "customer_read_results": [
            {
                "task_index": 1,
                "answer": "internal",
                "tool_result": {"ok": True, "data": {"found": True, "answer": "订单 当前状态：processing。"}},
            }
        ],
        "customer_case_records": [
            {"order_id": A, "action_type": "invoice_request", "status": "pending_review"}
        ],
        "pending_side_effect": {"requires_confirmation": True, "type": "invoice_request"},
    }


@pytest.mark.anyio
async def test_pending_receipt_cannot_hide_a_completed_read():
    llm = AsyncMock()
    answer = await present(
        llm,
        "申请发票并查配送",
        compound_state(),
        {"status": "pending_review", "action_type": "invoice_request"},
    )
    assert "发票" in answer and "核查" in answer and "电脑配件" in answer and "处理中" in answer
    llm.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_tool_failure_cannot_be_rendered_as_a_known_status():
    state = compound_state()
    state["customer_read_results"][0]["tool_result"]["ok"] = False
    answer = await present(AsyncMock(), "申请发票并查配送", state, None)
    assert "暂未查到可靠" in answer and "处理中" not in answer


@pytest.mark.anyio
async def test_old_successful_case_cannot_complete_a_different_request():
    state = compound_state()
    state["customer_case_records"] = [{"order_id": B, "action_type": "invoice_request", "status": "executed"}]
    answer = await present(AsyncMock(), "申请发票并查配送", state, None)
    assert "尚未确认" in answer and "开票申请已受理" not in answer


@pytest.mark.anyio
async def test_compound_read_uses_each_order_result():
    state = compound_state()
    state["task_plan"][0] = {"intent": "order_status", "order_id": A, "side_effect": False}
    state["customer_read_results"].append(
        {
            "task_index": 0,
            "tool_result": {"ok": True, "data": {"found": True, "answer": "订单 当前状态：delivered。"}},
        }
    )
    answer = await present(AsyncMock(), "两笔分别什么状态", state, None)
    assert "家居用品" in answer and "已送达" in answer and "电脑配件" in answer and "处理中" in answer


@pytest.mark.anyio
async def test_api_compound_query_executes_both_targets_and_resets_results(client, v2_runtime, monkeypatch):
    import app.main as main
    from app.llm.customer_presenter import present as actual_present

    monkeypatch.setattr(main, "present", lambda llm, q, state, record: actual_present(llm, q, state, record))
    with acceptance._guard()[0], acceptance._guard()[1]:
        first = await client.post(
            "/customer/chat",
            json={
                "session_id": "multi-status",
                "message": f"订单 {A} 和订单 {B} 分别是什么配送状态，只查一下",
            },
        )
        assert first.status_code == 200
        snapshot = await main.agent.aget_state({"configurable": {"thread_id": "multi-status"}})
        results = snapshot.values["customer_read_results"]
        assert {r["order_id"] for r in results} == {A, B}
        assert len(results) == 2
        second = await client.post(
            "/customer/chat", json={"session_id": "multi-status", "message": f"订单 {B} 只查发货状态"}
        )
        assert second.status_code == 200
        snapshot = await main.agent.aget_state({"configurable": {"thread_id": "multi-status"}})
        assert len(snapshot.values["customer_read_results"]) == 1
        assert v2_runtime["case_service"].list_by_session("multi-status") == []


@pytest.mark.anyio
async def test_api_pending_case_reply_keeps_read_result_and_scope(client, v2_runtime, monkeypatch):
    import app.main as main
    from app.llm.customer_presenter import present as actual_present

    monkeypatch.setattr(main, "present", lambda llm, q, state, record: actual_present(llm, q, state, record))
    with acceptance._guard()[0], acceptance._guard()[1]:
        response = await client.post(
            "/customer/chat",
            json={
                "session_id": "case-and-read",
                "message": f"订单 {A} 申请发票，抬头测试研发中心；订单 {B} 只查发货状态",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert "发票" in body["answer"] and "核查" in body["answer"]
    assert "待发货" in body["answer"] or "未发货" in body["answer"]
    cases = v2_runtime["case_service"].list_by_session("case-and-read")
    assert len(cases) == 1 and cases[0]["action_type"] == "invoice_request"
    assert cases[0]["status"] == "pending_review"
    assert v2_runtime["case_service"].count_order_events(A) == 0


@pytest.mark.parametrize("background", ["发票先等着", "退款先放着", "取消暂时搁着"])
def test_deferred_background_is_not_a_new_write(background):
    result = assess_request(f"{background}，订单 {B} 只查配送")
    assert not result.write_units
    assert [(u.intent, u.order_id) for u in result.read_units] == [("order_status", B)]


@pytest.mark.parametrize("background", ["先不说投诉了", "暂时不谈退款", "先不提发票"])
def test_background_topic_does_not_attach_a_write_to_new_order(background):
    result = assess_request(f"{background}，订单 {B} 继续查配送")
    assert not result.write_units
    assert {u.order_id for u in result.read_units} == {B}


@pytest.mark.anyio
async def test_order_cancellation_status_is_not_a_missing_case_lookup():
    agent = SemanticAgent([order(A, "电脑配件")], decision(goal="case_progress"))
    result = await semantic.ground(agent, "电脑配件这笔取消已经生效了吧，我只确认现状", {}, {A}, [])
    assert result.get("order_id") == A
    assert result.get("action") == "track_order"


def test_historical_cancellation_status_does_not_authorize_cancellation():
    result = assess_request(f"订单 {A} 取消已经生效了吧，我只确认现状")
    assert not result.write_units
    assert result.read_units


@pytest.mark.parametrize("text", ["不是问规则", "我不想了解政策", "不用询问条件"])
def test_negated_policy_topic_does_not_add_an_unrequested_task(text):
    result = assess_request(f"订单 {A} 我申请取消，{text}")
    assert [(u.intent, u.action_type) for u in result.units] == [("escalation", "cancel_order")]


@pytest.mark.anyio
async def test_validated_page_query_survives_model_listing_all_orders():
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")], decision(kind="list", indices=[1, 2], goal="status")
    )
    result = await semantic.ground(agent, "这笔发货了吗", {}, {A, B}, [], selected_order_id=B)
    assert result.get("order_id") == B and result.get("action") == "track_order"
    assert result["context"]["reference_decision"]["decision"] == "list"
    assert result["context"]["routing_override"]["reason"] == "validated_page_reference"


@pytest.mark.anyio
async def test_page_read_shortcut_cannot_override_a_different_literal_target():
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")], decision(kind="list", indices=[1, 2], goal="status")
    )
    result = await semantic.ground(agent, "不是这笔，查家居用品的物流", {}, {A, B}, [], selected_order_id=B)
    assert result.get("order_id") != B


@pytest.mark.anyio
async def test_unique_requested_categories_ignore_unrelated_duplicate_category():
    rows = [
        order(A, "母婴用品"),
        order(B, "电脑配件"),
        order("1" * 32, "健康与美容用品"),
        order("2" * 32, "健康与美容用品"),
    ]
    agent = SemanticAgent(rows, decision(indices=[1, 2]))
    result = await semantic.ground(
        agent, "母婴用品和电脑配件这两笔分别是什么状态", {}, {r["order_id"] for r in rows}, []
    )
    assert not result.get("answer")
    targets = {u.order_id for u in assess_request(result["message"]).read_units}
    assert targets == {A, B}


@pytest.mark.anyio
async def test_ambiguous_requested_category_still_requires_confirmation():
    rows = [order(A, "母婴用品"), order(B, "电脑配件"), order("1" * 32, "电脑配件")]
    agent = SemanticAgent(rows, decision(indices=[1, 2, 3]))
    result = await semantic.ground(
        agent, "母婴用品和电脑配件分别查状态", {}, {r["order_id"] for r in rows}, []
    )
    assert "message" not in result and "answer" in result
