from unittest.mock import AsyncMock

import pytest

from app.agent import semantic_order_grounding as semantic
from app.agent.customer_order_agent import CustomerOrderAgent
from tests.test_order_discovery import A, B, ReadAgent, order
from tests.test_order_discovery import client as client
from tests.test_order_discovery import v2_runtime as v2_runtime


class SemanticAgent(ReadAgent):
    available = True

    def __init__(self, rows, decision):
        super().__init__(rows)
        self.interpret_reference = AsyncMock(return_value=decision)


def decision(kind="select", indices=None, goal="status", request="question"):
    return semantic.ReferenceDecision(
        decision=kind,
        indices=indices or [1],
        goal=goal,
        request_kind=request,
    )


@pytest.mark.anyio
async def test_semantic_product_guess_is_only_a_suggestion():
    agent = SemanticAgent(
        [order(A, "电脑配件"), order(B, "母婴用品")], decision(goal="refund", request="request")
    )
    result = await semantic.ground(agent, "键盘想退", {}, {A, B}, [])
    assert "order_id" not in result
    assert "可能" in result["answer"] and "不能确认具体商品" in result["answer"]
    assert result["context"]["suggested_ids"] == [A]
    assert len(result["actions"]) == 1


@pytest.mark.anyio
async def test_suggested_candidate_can_be_confirmed_in_next_turn():
    agent = SemanticAgent([order(A, "电脑配件"), order(B, "母婴用品")], decision("suggest"))
    first = await semantic.ground(agent, "键盘到哪了", {}, {A, B}, [])
    agent.interpret_reference.return_value = decision()
    second = await semantic.ground(agent, "对，就是这笔，查物流", first["context"], {A, B}, [])
    assert second["order_id"] == A and second["action"] == "track_order"
    assert len(agent.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("indices", [[0], [3], [1, 1]])
async def test_invalid_model_indices_never_become_orders(indices):
    agent = SemanticAgent([order(A), order(B)], decision(indices=indices))
    result = await semantic.ground(agent, "第二个查物流", {}, {A, B}, [])
    assert "order_id" not in result
    assert result["context"]["semantic_status"] == "invalid_index_rejected"


@pytest.mark.anyio
async def test_model_cannot_override_literal_ordinal_with_another_order():
    agent = SemanticAgent([order(A), order(B)], decision(indices=[2], goal="refund", request="request"))
    result = await semantic.ground(agent, "第一个申请退款", {}, {A, B}, [])
    assert "order_id" not in result


@pytest.mark.anyio
async def test_negative_current_turn_clears_pending_write_even_if_model_requests_it():
    agent = SemanticAgent(
        [order(A), order(B)], decision("list", indices=[1, 2], goal="refund", request="request")
    )
    first = await semantic.ground(agent, "我买的东西申请退款", {}, {A, B}, [])
    agent.interpret_reference.return_value = decision(indices=[2], goal="refund", request="request")
    result = await semantic.ground(agent, "第二个，不要退款", first["context"], {A, B}, [])
    assert not result.get("action")
    assert not result["context"].get("pending_action")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "message", ["第一个如果明天还没到就退款", "第一个朋友说可以退款", "第一个还没决定退款"]
)
async def test_conditional_or_quoted_request_cannot_write_despite_model_error(message):
    agent = SemanticAgent([order(A), order(B)], decision(goal="refund", request="request"))
    result = await semantic.ground(agent, message, {}, {A, B}, [])
    assert result.get("action") != "refund_request"


@pytest.mark.anyio
async def test_policy_question_routes_to_read_only_policy_path():
    agent = SemanticAgent([order(A), order(B)], decision(goal="policy"))
    result = await semantic.ground(agent, "第一个能不能退款", {}, {A, B}, [])
    assert result["action"] == "refund_policy"


@pytest.mark.anyio
async def test_model_unavailable_falls_back_without_double_lookup():
    agent = SemanticAgent([order(A), order(B)], decision())
    agent.interpret_reference.side_effect = ValueError("bad JSON")
    result = await semantic.ground(agent, "第二个查物流", {}, {A, B}, [])
    assert result["order_id"] == B
    assert len(agent.calls) == 1
    assert result["context"]["semantic_status"] == "model_error_rule_fallback"


@pytest.mark.anyio
async def test_unowned_tool_rows_cannot_be_presented_to_model():
    agent = SemanticAgent([order(A), order(B)], decision("list"))
    result = await semantic.ground(agent, "我买的东西有问题", {}, {A}, [])
    assert [row["order_id"] for row in result["context"]["orders"]] == [A]
    assert B not in str(agent.interpret_reference.call_args)


@pytest.mark.anyio
async def test_multiple_explicit_categories_are_grounded_without_losing_constraints():
    agent = SemanticAgent(
        [order(A), order(B, "电脑配件")], decision(indices=[1, 2], goal="unknown", request="request")
    )
    result = await semantic.ground(agent, "家居用品不退款，电脑配件申请取消", {}, {A, B}, [])
    assert A in result["message"] and B in result["message"]
    assert "不退款" in result["message"]
    assert not result.get("action")


@pytest.mark.anyio
async def test_existing_case_progress_cannot_create_another_case():
    agent = SemanticAgent([order(A)], decision(goal="case_progress"))
    result = await semantic.ground(agent, "退款到哪一步了", {}, {A}, [], cases=[{"status": "pending_review"}])
    assert "待审核" in result["answer"] and "order_id" not in result


@pytest.mark.anyio
async def test_model_receives_bounded_history_and_indices_not_order_ids():
    class Llm:
        ainvoke = AsyncMock(
            return_value='{"decision":"suggest","indices":[1],"goal":"status",'
            '"request_kind":"question","reference_quote":"键盘"}'
        )

    llm = Llm()
    agent = CustomerOrderAgent(llm, None)
    history = [{"sender": "customer", "content": "上一轮"} for _ in range(20)]
    result = await agent.interpret_reference("键盘呢", [order(A, "电脑配件")], history, {})
    payload = str(llm.ainvoke.call_args)
    assert A not in payload
    assert payload.count("上一轮") == 8
    assert result.decision == "suggest"


@pytest.mark.anyio
async def test_off_topic_goal_passes_to_existing_input_guard():
    agent = SemanticAgent([order(A)], decision(goal="outside_scope"))
    result = await semantic.ground(agent, "帮我写一首诗", {}, {A}, [])
    assert result.get("passthrough") and "order_id" not in result


def test_pure_greeting_does_not_force_order_discovery():
    assert not semantic.applicable("你好！", {})
    assert semantic.applicable("我想问个事", {})


@pytest.mark.anyio
@pytest.mark.parametrize(
    "text,selected",
    [
        ("The health and beauty order. Just check delivery, do not cancel it.", True),
        ("Not the health and beauty order, show the others.", False),
    ],
)
async def test_source_category_key_requires_positive_reference(text, selected):
    row = {**order(A, "健康美容"), "category_key": "health_beauty"}
    agent = SemanticAgent([row, order(B)], decision())
    result = await semantic.ground(agent, text, {}, {A, B}, [])
    assert (result.get("order_id") == A) is selected


@pytest.mark.anyio
async def test_explicit_correction_can_ground_positive_target():
    agent = SemanticAgent([order(A, "家居用品"), order(B, "电脑配件")], decision(indices=[2]))
    result = await semantic.ground(agent, "不是家居，我说的是电脑配件，查一下物流", {}, {A, B}, [])
    assert result["order_id"] == B and result["action"] == "track_order"


@pytest.mark.anyio
async def test_reference_goal_cannot_collapse_multiple_explicit_actions():
    text = "家居用品送晚了，我要退款，同时还要发票，抬头测试采购中心"
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")], decision(goal="refund", request="request")
    )
    result = await semantic.ground(agent, text, {}, {A, B}, [])
    assert result["order_id"] == A and result["message"] == text
    assert not result.get("action") and not result.get("read_only")


@pytest.mark.anyio
async def test_customer_graph_executes_grounded_references_but_ledger_keeps_original(
    client, v2_runtime, monkeypatch
):
    from app import support_conversations as conversations
    from tests import test_v2_conversation_case_decoupling as acceptance

    main = v2_runtime["main"]
    agent = SemanticAgent(
        [order(A, "家居用品"), order(B, "电脑配件")], decision(indices=[1, 2], goal="status")
    )
    monkeypatch.setattr(main, "customer_order_agent", agent)
    monkeypatch.setattr(main, "_allowed_customer_order_ids", lambda user: {A, B})
    monkeypatch.setenv("ORDER_GROUNDING_MODE", "semantic")
    calls = []
    manager = v2_runtime["actions"]._tool_manager
    original = manager.call

    async def observe(name, args, context):
        calls.append((name, args))
        return await original(name, args, context)

    monkeypatch.setattr(manager, "call", observe)
    text = "家居用品查订单状态；电脑配件查订单状态"
    with acceptance._guard()[0], acceptance._guard()[1]:
        response = await client.post("/customer/chat", json={"session_id": "bound-history", "message": text})
    assert response.status_code == 200
    targets = {args["order_id"] for name, args in calls if name == "get_order_status"}
    assert targets == {A, B}
    ledger = conversations.messages("bound-history")
    assert [m["content"] for m in ledger if m["sender"] == "customer"] == [text]
    assert not v2_runtime["case_service"].list_by_session("bound-history")
