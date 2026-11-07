import pytest

from app.agent.customer_order_agent import CustomerOrderAgent
from app.agent.order_discovery import applicable, ground
from tests import test_v2_conversation_case_decoupling as acceptance

v2_runtime = acceptance.v2_runtime
client = acceptance.client
A, B = acceptance.A, acceptance.B


class ReadAgent:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = []
        self.failure = False

    def _execute(self, name, arguments, allowed):
        self.calls.append((name, arguments, allowed))
        if self.failure:
            raise OSError("temporary")
        return {"orders": self.rows}


def order(oid, category="家居用品", date="2018-01-01"):
    return {
        "order_id": oid,
        "category": category,
        "purchased_at": date,
        "status": "shipped",
        "status_label": "运输中",
        "amount": 120,
    }


def test_vague_entry_reads_owned_orders_without_guessing_action():
    agent = ReadAgent([order(A), order(B, "电脑配件")])
    result = ground(agent, "我买的东西有问题", {}, {A, B})
    assert agent.calls == [("search_user_orders", {"limit": 5}, {A, B})]
    assert "order_id" not in result
    assert [x["order_id"] for x in result["actions"]] == [A, B]
    assert "订单号" in result["answer"]


@pytest.mark.parametrize("message", ["你好", "在吗？", "退款一般需要什么条件？"])
def test_greeting_and_policy_do_not_trigger_order_discovery(message):
    assert not applicable(message, {})


def test_ordinal_uses_display_order_and_preserves_pending_request():
    agent = ReadAgent([order(B, "电脑配件"), order(A)])
    first = ground(agent, "我买的东西要申请退款", {}, {A, B})
    assert all(action["id"] == "refund_request" for action in first["actions"])
    second = ground(agent, "第二个", first["context"], {A, B})
    assert second["order_id"] == A
    assert "申请退款" in second["message"]
    assert len(agent.calls) == 1


def test_bad_ordinal_cannot_fall_back_to_old_selection():
    agent = ReadAgent([order(A), order(B)])
    context = ground(agent, "我买的东西有问题", {}, {A, B})["context"]
    context["selected_order_id"] = A
    assert "order_id" not in ground(agent, "第五个查一下", context, {A, B})


@pytest.mark.parametrize(
    "message", ["不是第二个", "第二个，不要退款", "第二个，先别办", "不是第二个，是第一个"]
)
def test_selection_cannot_replay_pending_refund_over_current_denial(message):
    agent = ReadAgent([order(A), order(B)])
    context = ground(agent, "我买的东西要申请退款", {}, {A, B})["context"]
    result = ground(agent, message, context, {A, B})
    assert "order_id" not in result


def test_same_category_requires_disambiguation():
    agent = ReadAgent([order(A, "电脑配件", "2018-01-01"), order(B, "电脑配件", "2018-03-01")])
    context = ground(agent, "电脑上用的东西怎么还没到", {}, {A, B})["context"]
    assert "order_id" not in ground(agent, "电脑配件查一下", context, {A, B})
    assert ground(agent, "买得比较晚的那笔，查一下物流", context, {A, B})["order_id"] == B


def test_new_category_switches_away_from_previous_selection():
    agent = ReadAgent([order(A), order(B, "电脑配件")])
    context = ground(agent, "买的东西帮我看下", {}, {A, B})["context"]
    context = ground(agent, "家居用品那笔，查状态", context, {A, B})["context"]
    assert ground(agent, "电脑上用的那笔查发货", context, {A, B})["order_id"] == B


def test_multiple_category_mentions_are_not_a_single_confirmation():
    agent = ReadAgent([order(A), order(B, "电脑配件")])
    context = ground(agent, "买的东西帮我看看", {}, {A, B})["context"]
    result = ground(agent, "不是电脑，是家居用品", context, {A, B})
    assert "order_id" not in result


def test_ownership_change_invalidates_cached_candidates():
    agent = ReadAgent([order(A)])
    context = ground(agent, "买的东西有问题", {}, {A})["context"]
    agent.rows = [order(B)]
    result = ground(agent, "第一个查一下", context, {B})
    assert len(agent.calls) == 2
    assert result.get("order_id") == B
    assert A not in str(result)


def test_failure_is_not_empty_and_can_be_retried():
    agent = ReadAgent([order(A)])
    agent.failure = True
    first = ground(agent, "买的东西帮我看下", {}, {A})
    assert "暂时" in first["answer"] and first["context"]["failed"]
    agent.failure = False
    second = ground(agent, "那你再试一下", first["context"], {A})
    assert second["actions"] and len(agent.calls) == 2


def test_empty_account_does_not_offer_other_orders():
    result = ground(ReadAgent(), "我买的东西怎么没消息", {}, set())
    assert "没有查到订单" in result["answer"]
    assert not result.get("actions") and "order_id" not in result


def test_unique_read_can_proceed_but_unknown_issue_cannot():
    agent = ReadAgent([order(A)])
    assert ground(agent, "我买的东西到哪了", {}, {A})["order_id"] == A
    assert "order_id" not in ground(agent, "我买的东西有问题", {}, {A})


def test_truncated_preview_does_not_prove_category_uniqueness():
    allowed = {A, B, "1" * 32, "2" * 32, "3" * 32, "4" * 32}
    agent = ReadAgent([order(A)])
    assert "order_id" not in ground(agent, "家居用品查状态", {}, allowed)


def test_order_confirmation_does_not_convert_policy_question_into_shipping():
    agent = ReadAgent([order(A), order(B)])
    context = ground(agent, "我买的东西有问题", {}, {A, B})["context"]
    result = ground(agent, "第二个查一下能不能退款", context, {A, B})
    assert result["order_id"] == B
    assert result["read_only"] is False
    assert result["message"] == "第二个查一下能不能退款"


@pytest.mark.anyio
async def test_real_chat_stores_choices_in_checkpoint_and_reads_selected_order(
    client, v2_runtime, monkeypatch
):
    import app.config.di as di
    import app.main as main

    agent = CustomerOrderAgent(di.llm_client.chat_openai, v2_runtime["olist_service"])
    monkeypatch.setattr(main, "customer_order_agent", agent)
    monkeypatch.setattr(main, "_allowed_customer_order_ids", lambda user: {A, B})
    first = await client.post(
        "/customer/chat", json={"session_id": "discover", "message": "我买的东西有问题"}
    )
    assert first.status_code == 200
    assert not v2_runtime["case_service"].list_by_session("discover")
    choices = first.json()["ui_actions"]
    target = choices[1]["order_id"]
    calls = []
    manager = v2_runtime["actions"]._tool_manager
    original = manager.call

    async def spy(name, arguments, context):
        calls.append((name, arguments))
        return await original(name, arguments, context)

    monkeypatch.setattr(manager, "call", spy)
    with acceptance._guard()[0], acceptance._guard()[1]:
        response = await client.post(
            "/customer/chat", json={"session_id": "discover", "message": "第二个，查一下物流"}
        )
    assert response.status_code == 200
    assert any(name == "get_order_status" and args.get("order_id") == target for name, args in calls)
    assert not v2_runtime["case_service"].list_by_session("discover")
    state = await main.conversation_agent.aget_state({"configurable": {"thread_id": "discover"}})
    assert state.values["customer_order_context"]["selected_order_id"] == target


@pytest.mark.anyio
async def test_discovery_does_not_allow_cross_user_session_reuse(client, v2_runtime, monkeypatch):
    import app.config.di as di
    import app.main as main

    monkeypatch.setattr(
        main,
        "customer_order_agent",
        CustomerOrderAgent(di.llm_client.chat_openai, v2_runtime["olist_service"]),
    )
    response = await client.post(
        "/customer/chat", json={"session_id": "owned-discovery", "message": "买的东西有问题"}
    )
    assert response.status_code == 200
    response = await client.post(
        "/customer/chat",
        headers={"X-Demo-Customer": "customer-smoke"},
        json={"session_id": "owned-discovery", "message": "第二个申请退款"},
    )
    assert response.status_code == 403
    assert not v2_runtime["case_service"].list_by_session("owned-discovery")


@pytest.mark.anyio
async def test_display_order_survives_sqlite_checkpoint_reopen(client, v2_runtime, monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    import app.config.di as di
    import app.main as main
    from app.agent.graph import AgentGraph

    monkeypatch.setattr(
        main,
        "customer_order_agent",
        CustomerOrderAgent(di.llm_client.chat_openai, v2_runtime["olist_service"]),
    )
    monkeypatch.setattr(main, "_allowed_customer_order_ids", lambda user: {A, B})
    database = str(tmp_path / "discovery-checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        graph = AgentGraph(v2_runtime["actions"]).build(saver)
        monkeypatch.setattr(main, "conversation_agent", graph)
        first = await client.post(
            "/customer/chat", json={"session_id": "restart-discovery", "message": "我买的东西有问题"}
        )
        assert first.status_code == 200
        target = first.json()["ui_actions"][1]["order_id"]
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        graph = AgentGraph(v2_runtime["actions"]).build(saver)
        monkeypatch.setattr(main, "conversation_agent", graph)
        with acceptance._guard()[0], acceptance._guard()[1]:
            second = await client.post(
                "/customer/chat", json={"session_id": "restart-discovery", "message": "第二个，查是否发货"}
            )
        assert second.status_code == 200
        state = await graph.aget_state({"configurable": {"thread_id": "restart-discovery"}})
        assert state.values["active_order_id"] == target
        assert state.values["customer_order_context"]["selected_order_id"] == target
        assert not v2_runtime["case_service"].list_by_session("restart-discovery")


@pytest.mark.anyio
async def test_explicit_product_reference_can_override_stale_page_selection(client, v2_runtime, monkeypatch):
    import app.config.di as di
    import app.main as main

    home = "8fc207e94fa91a7649c5a5dab690272a"
    computer = "ad27779fb29796c8d798f29aefedf3d0"
    monkeypatch.setattr(main, "_allowed_customer_order_ids", lambda user: {home, computer})
    monkeypatch.setattr(
        main,
        "customer_order_agent",
        CustomerOrderAgent(di.llm_client.chat_openai, v2_runtime["olist_service"]),
    )
    with acceptance._guard()[0], acceptance._guard()[1]:
        response = await client.post(
            "/customer/chat",
            json={
                "session_id": "stale-selection",
                "message": "电脑配件那笔查一下物流",
                "page_context": {"selected_order_id": home},
            },
        )
    assert response.status_code == 200
    state = await main.conversation_agent.aget_state({"configurable": {"thread_id": "stale-selection"}})
    assert state.values["active_order_id"] == computer
    assert not v2_runtime["case_service"].list_by_session("stale-selection")
