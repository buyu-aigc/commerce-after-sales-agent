"""V2 acceptance tests: conversation and after-sales Case are decoupled."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

from app import support_conversations as conversations
from app.after_sales.shadow import ShadowService, SQLiteShadowStore
from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.case_workflow_service import CaseWorkflowService
from app.llm.types import InputGuardResult, OutputGuardResult, PlannedTask, TaskPlanResult
from app.olist.service import OlistService, OrderStatusView, SQLiteCaseService
from app.policy.engine import PolicyEngine
from app.policy.guard import PolicyGuard
from app.tool_call import InMemoryRuntimeStore, build_business_tool_manager

A = "203096f03d82e0dffbc41ebc2e2bcfb7"  # delivered, delayed
B = "8aec3a066f732dd927ec8fef1752415b"  # invoiced, pre-shipment
REVIEW_HEADERS = {"X-Review-Token": "local-review-demo"}


def _guard():
    async def check_input(self, message, history=None):
        return InputGuardResult(on_topic=True, reason="ok")

    async def check_output(self, answer):
        return OutputGuardResult(valid=True, reason="ok")

    return (
        patch("app.llm.guardrail.Guardrail.check_input", check_input),
        patch("app.llm.guardrail.Guardrail.check_output", check_output),
    )


def _plan(*intents: str):
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


def _plan_action(action_type: str):
    task = PlannedTask(intent="escalation", text="", side_effect=True, action_type=action_type)
    return patch(
        "app.llm.intent_planner.IntentPlanner.plan",
        AsyncMock(return_value=TaskPlanResult(tasks=[task])),
    )


@pytest.fixture
def v2_runtime(tmp_path, monkeypatch):
    import app.config.di as di
    import app.main as main_module

    case_service = SQLiteCaseService(tmp_path / "cases.db")
    olist_service = OlistService(projection_store=case_service)
    runtime_store = InMemoryRuntimeStore()
    tool_manager = build_business_tool_manager(
        olist_service=olist_service,
        knowledge_base=di.knowledge_base,
        support_retriever=di.support_retriever,
        case_service=case_service,
        cache_backend=runtime_store,
        runtime_store=runtime_store,
    )
    policy_engine = PolicyEngine()
    policy_guard = PolicyGuard(policy_engine, olist_service, case_service)
    actions = AgentActions(
        intent_planner=di.intent_planner,
        qa_generator=di.qa_generator,
        policy_generator=di.policy_generator,
        task_extractor=di.task_extractor,
        olist_service=olist_service,
        knowledge_base=di.knowledge_base,
        support_retriever=di.support_retriever,
        case_service=case_service,
        tool_manager=tool_manager,
        runtime_store=runtime_store,
        policy_engine=policy_engine,
        policy_guard=policy_guard,
    )
    checkpointer = InMemorySaver()
    conversation_graph = AgentGraph(actions).build(checkpointer)
    workflow = CaseWorkflowService(case_service, actions)
    workflow.configure(checkpointer, legacy_agent=None)
    actions.bind_case_workflow(workflow)

    monkeypatch.setattr(main_module, "agent", conversation_graph)
    monkeypatch.setattr(main_module, "conversation_agent", conversation_graph)
    monkeypatch.setattr(main_module, "case_service", case_service)
    monkeypatch.setattr(main_module, "case_workflow_service", workflow)
    monkeypatch.setattr(main_module, "olist_service", olist_service)
    monkeypatch.setattr(main_module, "llm_client", di.llm_client)
    monkeypatch.setattr(main_module, "present", AsyncMock(return_value="客服回复"))
    monkeypatch.setattr(conversations, "DB", tmp_path / "support_conversations.db")
    conversations.initialize()
    return {
        "main": main_module,
        "case_service": case_service,
        "olist_service": olist_service,
        "workflow": workflow,
        "actions": actions,
    }


@pytest.fixture
async def client(v2_runtime):
    transport = ASGITransport(app=v2_runtime["main"].app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _create_case(client: AsyncClient, session_id: str, action_type: str, order_id: str = A) -> str:
    with _guard()[0], _guard()[1], _plan_action(action_type):
        response = await client.post(
            "/customer/chat",
            json={"message": f"给订单 {order_id} 申请售后处理", "session_id": session_id},
        )
    assert response.status_code == 200, response.text
    case_id = response.json().get("case_id")
    assert case_id, response.text
    return case_id


@pytest.mark.anyio
async def test_a_pending_does_not_block_b_query(client: AsyncClient, v2_runtime):
    session_id = "v2-a-then-b"
    case_id = await _create_case(client, session_id, "refund_request", A)

    import app.main as main
    from app.config.di import llm_client
    from app.llm.customer_presenter import present
    from app.llm.offline import OfflineChatModel

    workflow = v2_runtime["workflow"]
    record = v2_runtime["case_service"].get(case_id)
    config = {"configurable": {"thread_id": record["workflow_thread_id"]}}
    before = await workflow._case_graph.aget_state(config)
    calls = []
    manager = v2_runtime["actions"]._tool_manager
    original = manager.call

    async def spy(name, args, context):
        result = await original(name, args, context)
        calls.append((name, args, result))
        return result

    with (
        patch.object(manager, "call", spy),
        patch.object(main, "present", present),
        patch.object(llm_client, "chat_openai", OfflineChatModel()),
        _guard()[0],
        _guard()[1],
        _plan("order_status"),
    ):
        response = await client.post(
            "/customer/chat",
            json={
                "message": f"查询订单 {B} 的物流状态",
                "session_id": session_id,
                "page_context": {"selected_order_id": B},
            },
        )
    assert response.status_code == 200, response.text
    successful = [
        result
        for name, args, result in calls
        if name == "get_order_status" and args.get("order_id") == B and result.ok
    ]
    assert successful, calls
    answer = response.json()["answer"]
    assert "已受理" not in answer and "正在核查" not in answer, answer
    assert B in str(successful[0].data)
    assert "发货" in answer or "确认" in answer or "开票" in answer, answer
    after = await workflow._case_graph.aget_state(config)
    assert before.next == after.next
    assert before.values["escalation_draft"] == after.values["escalation_draft"]
    state = await main.conversation_agent.aget_state({"configurable": {"thread_id": session_id}})
    assert not state.next
    assert state.values["active_order_id"] == B
    approved = await client.post(
        f"/review/cases/{case_id}/approve", headers=REVIEW_HEADERS, json={"expected_attempt": 1}
    )
    assert approved.status_code == 200
    state_after = await main.conversation_agent.aget_state({"configurable": {"thread_id": session_id}})
    assert state_after.values["active_order_id"] == B
    assert v2_runtime["case_service"].count_case_events(case_id) == 1


@pytest.mark.anyio
async def test_approve_a_by_case_id_is_idempotent(client: AsyncClient, v2_runtime):
    case_id = await _create_case(client, "v2-approve-a", "refund_request", A)
    approved = await client.post(
        f"/review/cases/{case_id}/approve",
        headers=REVIEW_HEADERS,
        json={
            "reviewer_id": "reviewer-1",
            "role": "after_sales_operator",
            "auth_scopes": ["after_sales:write"],
        },
    )
    assert approved.status_code == 200, approved.text
    record = v2_runtime["case_service"].get(case_id)
    assert record["status"] == "executed"
    assert v2_runtime["case_service"].count_case_events(case_id) == 1

    repeat = await client.post(
        f"/review/cases/{case_id}/approve",
        headers=REVIEW_HEADERS,
        json={
            "reviewer_id": "reviewer-2",
            "role": "after_sales_operator",
            "auth_scopes": ["after_sales:write"],
        },
    )
    assert repeat.status_code == 200, repeat.text
    assert v2_runtime["case_service"].count_case_events(case_id) == 1


@pytest.mark.anyio
async def test_two_cases_can_be_reviewed_in_reverse_order(client: AsyncClient, v2_runtime):
    session_id = "v2-two-cases"
    refund = await _create_case(client, session_id, "refund_request", A)
    complaint = await _create_case(client, session_id, "complaint_escalation", A)
    assert refund != complaint
    records = v2_runtime["case_service"].list_by_session(session_id)
    assert {item["case_id"] for item in records} == {refund, complaint}

    for case_id in (complaint, refund):
        response = await client.post(
            f"/review/cases/{case_id}/approve",
            headers=REVIEW_HEADERS,
            json={
                "reviewer_id": "reviewer-x",
                "role": "after_sales_operator",
                "auth_scopes": ["after_sales:write"],
            },
        )
        assert response.status_code == 200, response.text
        assert v2_runtime["case_service"].count_case_events(case_id) == 1
        assert v2_runtime["case_service"].get(case_id)["status"] == "executed"


@pytest.mark.anyio
async def test_reject_has_no_write_and_stale_approve_conflicts(client: AsyncClient, v2_runtime):
    case_id = await _create_case(client, "v2-reject", "refund_request", A)
    rejected = await client.post(
        f"/review/cases/{case_id}/reject",
        headers=REVIEW_HEADERS,
        json={
            "reviewer_id": "reviewer-r",
            "role": "after_sales_operator",
            "auth_scopes": ["after_sales:write"],
        },
    )
    assert rejected.status_code == 200, rejected.text
    assert v2_runtime["case_service"].count_case_events(case_id) == 0
    assert v2_runtime["case_service"].get(case_id)["status"] == "rejected"

    stale = await client.post(
        f"/review/cases/{case_id}/approve",
        headers=REVIEW_HEADERS,
        json={
            "reviewer_id": "reviewer-a",
            "role": "after_sales_operator",
            "auth_scopes": ["after_sales:write"],
        },
    )
    assert stale.status_code == 409
    assert v2_runtime["case_service"].count_case_events(case_id) == 0


@pytest.mark.anyio
async def test_other_customer_cannot_read_or_approve_case(client: AsyncClient, v2_runtime):
    case_id = await _create_case(client, "v2-owner", "refund_request", A)
    read = await client.get(f"/customer/cases/{case_id}", headers={"X-Demo-Customer": "customer-smoke"})
    assert read.status_code == 403
    review = await client.get(f"/review/cases/{case_id}", headers=REVIEW_HEADERS)
    assert review.status_code == 200


@pytest.mark.anyio
async def test_concurrent_approve_produces_one_write(client: AsyncClient, v2_runtime):
    case_id = await _create_case(client, "v2-concurrent", "refund_request", A)
    payload = {
        "reviewer_id": "reviewer-c",
        "role": "after_sales_operator",
        "auth_scopes": ["after_sales:write"],
    }
    first, second = await asyncio.gather(
        client.post(f"/review/cases/{case_id}/approve", headers=REVIEW_HEADERS, json=payload),
        client.post(f"/review/cases/{case_id}/approve", headers=REVIEW_HEADERS, json=payload),
    )
    assert first.status_code in {200, 409}
    assert second.status_code in {200, 409}
    assert v2_runtime["case_service"].count_case_events(case_id) == 1
    assert v2_runtime["case_service"].get(case_id)["status"] == "executed"


@pytest.mark.anyio
async def test_progress_lists_multiple_cases_for_owner(client: AsyncClient):
    session_id = "v2-progress"
    refund = await _create_case(client, session_id, "refund_request", A)
    complaint = await _create_case(client, session_id, "complaint_escalation", A)
    progress = await client.get(f"/customer/sessions/{session_id}/progress")
    assert progress.status_code == 200, progress.text
    body = progress.json()
    assert {item["case_id"] for item in body["cases"]} == {refund, complaint}
    other = await client.get(
        f"/customer/sessions/{session_id}/progress", headers={"X-Demo-Customer": "customer-smoke"}
    )
    assert other.status_code == 403


@pytest.mark.anyio
async def test_migrate_legacy_case_binds_after_checkpoint_check(v2_runtime):
    import sqlite3
    from types import SimpleNamespace

    case_service = v2_runtime["case_service"]
    workflow = v2_runtime["workflow"]
    created = case_service.submit_for_review(
        action_type="refund_request",
        order_id=A,
        message_text="legacy pending",
        session_id="legacy-session",
        user_id="demo-customer",
        expires_at=None,
    )
    case_id = str(created["result_id"])
    with sqlite3.connect(case_service.path) as conn:
        conn.execute(
            "UPDATE cases SET workflow_thread_id = NULL, workflow_version = 1 WHERE case_id = ?",
            (case_id,),
        )

    class FakeLegacy:
        async def aget_state(self, config):
            return SimpleNamespace(
                next=("await_confirmation",),
                values={"escalation_draft": {"review_case_id": case_id}},
            )

    workflow._legacy_agent = FakeLegacy()
    dry = await workflow.migrate_legacy(apply=False)
    assert dry["bound"] and dry["bound"][0]["case_id"] == case_id
    assert case_service.get(case_id)["workflow_thread_id"] is None
    applied = await workflow.migrate_legacy(apply=True)
    assert applied["bound"] and applied["bound"][0]["case_id"] == case_id
    assert case_service.get(case_id)["workflow_thread_id"] == "legacy-session"
    again = await workflow.migrate_legacy(apply=True)
    assert not again["bound"]
    assert any(item["case_id"] == case_id for item in again["skipped"])


@pytest.mark.anyio
async def test_restart_keeps_case_workflow_resumable(tmp_path):
    """The Case checkpoint must survive an application/checkpointer restart."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    import app.config.di as di

    def build_runtime():
        service = SQLiteCaseService(tmp_path / "restart_cases.db")
        olist = OlistService(projection_store=service)
        store = InMemoryRuntimeStore()
        manager = build_business_tool_manager(
            olist_service=olist,
            knowledge_base=di.knowledge_base,
            support_retriever=di.support_retriever,
            case_service=service,
            cache_backend=store,
            runtime_store=store,
        )
        actions = AgentActions(
            intent_planner=di.intent_planner,
            qa_generator=di.qa_generator,
            policy_generator=di.policy_generator,
            task_extractor=di.task_extractor,
            olist_service=olist,
            knowledge_base=di.knowledge_base,
            support_retriever=di.support_retriever,
            case_service=service,
            tool_manager=manager,
            runtime_store=store,
        )
        return service, actions

    state = {
        "session_id": "restart-session",
        "conversation_id": "restart-session",
        "tenant_id": "olist-demo",
        "user_id": "demo-customer",
        "messages": [{"role": "user", "content": "申请退款"}],
        "completed_tasks": [],
        "allowed_order_ids": [A],
        "auth_scopes": ["orders:read", "after_sales:assess", "after_sales:draft"],
        "channel": "customer_self_service",
    }
    draft = {
        "action_type": "refund_request",
        "order_id": A,
        "message_text": "申请退款",
        "task_index": 0,
        "decision": {"outcome": "needs_human_review", "risk_level": "medium"},
        "verification": {"required_next_step": "hitl"},
    }
    checkpoint_path = tmp_path / "checkpoints.db"

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        service, actions = build_runtime()
        workflow = CaseWorkflowService(service, actions)
        workflow.configure(saver)
        record = await workflow.dispatch_case(
            action_type="refund_request",
            order_id=A,
            message_text="申请退款",
            state=state,  # type: ignore[arg-type]
            draft=draft,
            expires_at=time.time() + 900,
            timeout_seconds=900,
            handoff_reasons=["write_action_requires_review"],
        )
        case_id = record["case_id"]

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        service, actions = build_runtime()
        workflow = CaseWorkflowService(service, actions)
        workflow.configure(saver)
        outcome = await workflow.resume_case(case_id, "approve", "restart-reviewer")
        assert outcome["ok"] is True, outcome
        assert service.count_case_events(case_id) == 1
        assert service.get(case_id)["status"] == "executed"


@pytest.mark.anyio
async def test_same_turn_query_b_and_escalate_a(client: AsyncClient, v2_runtime):
    plan = TaskPlanResult(
        tasks=[
            PlannedTask(
                intent="order_status",
                text=f"查询订单 {B} 的物流状态",
                side_effect=False,
                action_type="none",
            ),
            PlannedTask(
                intent="escalation",
                text=f"给订单 {A} 申请退款",
                side_effect=True,
                action_type="refund_request",
            ),
        ]
    )
    calls: list[str] = []
    original = v2_runtime["olist_service"].get_order_status

    def spy(order_id: str):
        calls.append(order_id)
        return original(order_id)

    v2_runtime["olist_service"].get_order_status = spy  # type: ignore[method-assign]
    with (
        _guard()[0],
        _guard()[1],
        patch("app.llm.intent_planner.IntentPlanner.plan", AsyncMock(return_value=plan)),
    ):
        response = await client.post(
            "/customer/chat",
            json={
                "message": f"先查订单 {B} 的物流，再给订单 {A} 申请退款",
                "session_id": "v2-same-turn",
            },
        )
    assert response.status_code == 200, response.text
    assert B in calls, f"B order tool was not called: {calls}"
    case_id = response.json().get("case_id")
    assert case_id
    packet = await client.get(f"/review/cases/{case_id}", headers=REVIEW_HEADERS)
    assert packet.status_code == 200
    assert packet.json()["has_pending"] is True


@pytest.mark.anyio
async def test_session_cases_endpoint_lists_both_cases(client: AsyncClient):
    session_id = "v2-case-list"
    refund = await _create_case(client, session_id, "refund_request", A)
    complaint = await _create_case(client, session_id, "complaint_escalation", A)
    response = await client.get(f"/review/sessions/{session_id}/cases", headers=REVIEW_HEADERS)
    assert response.status_code == 200, response.text
    assert {item["case_id"] for item in response.json()["cases"]} == {refund, complaint}


@pytest.mark.anyio
async def test_crash_between_decision_and_resume_is_recovered(client: AsyncClient, v2_runtime):
    case_id = await _create_case(client, "v2-crash-resume", "refund_request", A)
    service = v2_runtime["case_service"]
    workflow = v2_runtime["workflow"]

    # Simulate: CAS committed, process died before graph.ainvoke(resume=...).
    decided = service.decide_case(case_id, "approve", "crashed-reviewer")
    assert decided["ok"] is True
    assert service.get(case_id)["status"] == "approving"
    job_key = f"resume:{case_id}:1"
    assert service.get_workflow_job(job_key)["status"] == "queued"

    await workflow.recover_jobs()
    assert service.get(case_id)["status"] == "executed"
    assert service.count_case_events(case_id) == 1
    assert service.get_workflow_job(job_key)["status"] == "done"

    await workflow.recover_jobs()
    assert service.count_case_events(case_id) == 1


@pytest.mark.anyio
async def test_review_queue_includes_stuck_approving_case(client: AsyncClient, v2_runtime):
    import sqlite3

    case_id = await _create_case(client, "v2-stuck-queue", "refund_request", A)
    service = v2_runtime["case_service"]
    service.decide_case(case_id, "approve", "reviewer-stuck")
    with sqlite3.connect(service.path) as conn:
        conn.execute(
            "UPDATE cases SET reviewed_at = '2020-01-01T00:00:00Z' WHERE case_id = ?",
            (case_id,),
        )
    response = await client.get("/review/cases", headers=REVIEW_HEADERS)
    assert response.status_code == 200, response.text
    row = next(item for item in response.json()["cases"] if item["case_id"] == case_id)
    assert row["status"] == "approving"
    assert row["needs_attention"] is True
    assert row["stuck_seconds"] >= 120
    assert "case_payload" not in row


@pytest.mark.anyio
async def test_cancel_without_order_returns_owned_order_choices(client: AsyncClient):
    response = await client.post(
        "/customer/chat",
        json={"message": "我要取消订单", "session_id": "v2-order-choice"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["case_id"] is None
    assert "订单卡片" not in body["answer"]
    actions = body["ui_actions"]
    assert len(actions) >= 1
    assert all("取消订单" in action["message"] for action in actions)
    assert all(action["order_id"] for action in actions)

    chosen = actions[0]
    result = await client.post(
        "/customer/chat",
        json={
            "message": chosen["message"],
            "session_id": "v2-order-choice",
            "requested_action": chosen["id"],
            "page_context": {"surface": "assistant", "selected_order_id": chosen["order_id"]},
        },
    )
    assert result.status_code == 200, result.text
    assert result.json().get("case_id")


@pytest.mark.anyio
async def test_assistant_page_is_standalone(client: AsyncClient):
    response = await client.get("/assistant")
    assert response.status_code == 200
    assert "orders-rail" in response.text
    assert "display: none" in response.text


@pytest.mark.anyio
async def test_category_dislike_resolves_orders_instead_of_generic_menu(client: AsyncClient):
    response = await client.post(
        "/customer/chat",
        json={"message": "健康与美容用品不要了", "session_id": "v2-category-dislike"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "健康与美容" in body["answer"]
    assert "请选择要处理的订单" in body["answer"]
    assert body["ui_actions"]
    assert all("退货退款" in action["label"] for action in body["ui_actions"])
    assert all(action["order_id"] for action in body["ui_actions"])


@pytest.mark.anyio
async def test_customer_chat_exposes_agent_order_actions(client):
    import app.main as main_module

    candidates = [
        {
            "order_id": A,
            "order_id_masked": "203096f0...cfb7",
            "category": "健康与美容用品",
            "status": "delivered",
            "status_label": "已送达",
        }
    ]
    with patch.object(
        main_module.customer_order_agent,
        "resolve",
        AsyncMock(return_value={"answer": "请选择要处理的订单。", "candidates": candidates}),
    ):
        response = await client.post(
            "/customer/chat",
            json={"message": "健康与美容用品不要了", "session_id": "agent-action-test"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "请选择要处理的订单。"
    assert body["ui_actions"][0]["order_id"] == A
    assert body["ui_actions"][0]["id"] == "refund_request"


@pytest.mark.anyio
async def test_low_value_delayed_refund_auto_executes_without_review(
    client: AsyncClient, v2_runtime, monkeypatch
):

    original = v2_runtime["olist_service"].get_order_status

    def fake_order(order_id: str):
        if order_id != A:
            return original(order_id)
        return OrderStatusView(
            order_id=A,
            status="delivered",
            customer_state="SP",
            purchase_timestamp="2017-09-18 14:31:30",
            estimated_delivery_date="2017-09-28 00:00:00",
            delivered_customer_date="2017-10-09 22:23:46",
            delay_days=11,
            payment_value=43.0,
            review_score=2,
            category_summary="health_beauty",
        )

    monkeypatch.setattr(v2_runtime["olist_service"], "get_order_status", fake_order)

    with (
        _guard()[0],
        _guard()[1],
        _plan_action("refund_request"),
    ):
        response = await client.post(
            "/customer/chat",
            json={"message": f"订单 {A} 晚到了很多天，我要申请退款", "session_id": "auto-policy-e2e"},
        )
    assert response.status_code == 200, response.text
    case_id = response.json().get("case_id")
    assert case_id, response.json()
    record = v2_runtime["case_service"].get(case_id)
    assert record is not None
    assert record["status"] == "executed"
    assert v2_runtime["case_service"].list_pending() == []
    assert v2_runtime["case_service"].count_order_events(A) == 1


@pytest.mark.anyio
async def test_auto_execution_creates_an_executed_case_for_progress(client: AsyncClient, v2_runtime):
    session_id = "v2-auto-case"
    response = await client.post(
        "/customer/chat",
        json={
            "message": "我想取消这笔订单。",
            "session_id": session_id,
            "requested_action": "cancel_order",
            "page_context": {"surface": "order_detail", "selected_order_id": B},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    case_id = payload.get("case_id")
    assert case_id, payload

    record = v2_runtime["case_service"].get(case_id)
    assert record is not None
    assert record["status"] == "executed"
    assert v2_runtime["case_service"].count_case_events(case_id) == 1

    progress = await client.get(f"/customer/sessions/{session_id}/progress")
    assert progress.status_code == 200
    cases = progress.json()["cases"]
    assert any(item["case_id"] == case_id and item["status"] == "executed" for item in cases)


def _fake_view(
    order_id: str,
    *,
    status: str = "invoiced",
    amount: float = 45.0,
    delay: int = 0,
    review: int = 4,
) -> OrderStatusView:
    return OrderStatusView(
        order_id=order_id,
        status=status,
        customer_state="SP",
        purchase_timestamp="2017-09-18 14:31:30",
        estimated_delivery_date="2017-09-28 00:00:00",
        delivered_customer_date="2017-10-09 22:23:46" if status == "delivered" else "",
        delay_days=delay,
        payment_value=amount,
        review_score=review,
        category_summary="health_beauty",
    )


@pytest.mark.anyio
async def test_complex_request_partial_auto_executes_low_and_reviews_high(
    client: AsyncClient, v2_runtime, monkeypatch
):
    session_id = "v2-complex-partial"
    views = {
        A: _fake_view(A, status="invoiced", amount=45.0),
        B: _fake_view(B, status="delivered", amount=500.0, delay=11),
    }
    original = v2_runtime["olist_service"].get_order_status

    def fake_order(order_id: str):
        return views.get(order_id) or original(order_id)

    monkeypatch.setattr(v2_runtime["olist_service"], "get_order_status", fake_order)
    with _guard()[0], _guard()[1]:
        response = await client.post(
            "/customer/chat",
            json={
                "message": f"订单 {A} 帮我取消，订单 {B} 晚到了我要退款",
                "session_id": session_id,
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_analysis"]["tier"] == "complex"
    assert body["request_analysis"]["high_risk_count"] == 1
    assert body["request_analysis"]["partial_auto"] is True
    assert body["request_analysis"]["processed"] == 1
    assert body["request_analysis"]["pending_review"] == 1

    cases = {item["action_type"]: item for item in body["sub_cases"]}
    assert set(cases) == {"cancel_order", "refund_request"}
    assert cases["cancel_order"]["status"] == "executed"
    assert cases["cancel_order"]["risk_level"] == "low"
    assert cases["cancel_order"]["policy_decision"] == "auto"
    assert cases["refund_request"]["status"] == "pending_review"
    assert cases["refund_request"]["risk_level"] == "high"
    assert cases["refund_request"]["policy_decision"] == "review"

    progress = await client.get(f"/customer/sessions/{session_id}/progress")
    assert progress.status_code == 200
    summary = progress.json()["summary"]
    assert summary == {
        "processed": 1,
        "pending_review": 1,
        "rejected": 0,
        "failed": 0,
        "processing": 0,
        "total": 2,
        "partial_auto": True,
    }
    assert progress.json()["request"]["tier"] == "complex"
    assert len(progress.json()["groups"]["processed"]) == 1
    assert len(progress.json()["groups"]["pending_review"]) == 1


@pytest.mark.anyio
async def test_same_order_refund_and_invoice_are_two_policy_checked_cases(
    client: AsyncClient, v2_runtime, monkeypatch
):
    session_id = "v2-same-order-complex"
    view = _fake_view(A, status="delivered", amount=43.0, delay=11)
    monkeypatch.setattr(
        v2_runtime["olist_service"], "get_order_status", lambda order_id: view if order_id == A else None
    )
    with _guard()[0], _guard()[1]:
        response = await client.post(
            "/customer/chat",
            json={
                "message": f"订单 {A} 晚到了我要退款，另外开发票",
                "session_id": session_id,
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_analysis"]["tier"] == "complex"
    assert set(body["request_analysis"]["action_types"]) == {"refund_request", "invoice_request"}
    cases = {item["action_type"]: item for item in body["sub_cases"]}
    assert len(cases) == 2
    assert cases["refund_request"]["status"] == "executed"
    assert cases["refund_request"]["policy_decision"] == "auto"
    assert cases["invoice_request"]["status"] == "pending_review"
    assert cases["invoice_request"]["policy_decision"] == "review"


@pytest.mark.anyio
async def test_complex_request_missing_order_creates_no_guessed_case(
    client: AsyncClient, v2_runtime, monkeypatch
):
    session_id = "v2-complex-missing-order"
    view = _fake_view(A, status="invoiced", amount=45.0)
    monkeypatch.setattr(
        v2_runtime["olist_service"], "get_order_status", lambda order_id: view if order_id == A else None
    )
    with _guard()[0], _guard()[1]:
        response = await client.post(
            "/customer/chat",
            json={
                "message": f"我要退款，另外订单 {A} 帮我取消",
                "session_id": session_id,
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    actions = [item["action_type"] for item in body["sub_cases"]]
    assert actions == ["cancel_order"]
    assert body["sub_cases"][0]["status"] == "executed"
    assert "order_id" in body["request_analysis"]["missing_evidence"]
    assert body["request_analysis"]["auto_eligible_count"] == 1


@pytest.mark.anyio
async def test_high_risk_complaint_sub_case_never_auto_executes(client: AsyncClient, v2_runtime, monkeypatch):
    session_id = "v2-high-risk-no-auto"
    view = _fake_view(A, status="delivered", amount=45.0)
    monkeypatch.setattr(
        v2_runtime["olist_service"], "get_order_status", lambda order_id: view if order_id == A else None
    )
    with _guard()[0], _guard()[1]:
        response = await client.post(
            "/customer/chat",
            json={"message": f"订单 {A} 我要投诉", "session_id": session_id},
        )
    assert response.status_code == 200, response.text
    cases = response.json()["sub_cases"]
    assert len(cases) == 1
    assert cases[0]["action_type"] == "complaint_escalation"
    assert cases[0]["status"] == "pending_review"
    assert cases[0]["risk_level"] == "high"


@pytest.mark.anyio
async def test_shadow_mode_forces_review_and_compares_with_human_decision(
    client: AsyncClient, v2_runtime, monkeypatch, tmp_path
):
    session_id = "v2-shadow-mode"
    views = {
        A: _fake_view(A, status="invoiced", amount=45.0),
        B: _fake_view(B, status="delivered", amount=500.0, delay=11),
    }
    original = v2_runtime["olist_service"].get_order_status

    def fake_order(order_id: str):
        return views.get(order_id) or original(order_id)

    monkeypatch.setattr(v2_runtime["olist_service"], "get_order_status", fake_order)
    shadow = ShadowService(SQLiteShadowStore(tmp_path / "shadow.db"), enabled=True, sample_rate=1.0)
    v2_runtime["actions"]._shadow_service = shadow
    monkeypatch.setattr(v2_runtime["main"], "shadow_service", shadow)

    with _guard()[0], _guard()[1]:
        response = await client.post(
            "/customer/chat",
            json={
                "message": f"订单 {A} 帮我取消，订单 {B} 晚到了我要退款",
                "session_id": session_id,
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_analysis"]["tier"] == "complex"
    assert body["request_analysis"]["partial_auto"] is False
    cases = {item["action_type"]: item for item in body["sub_cases"]}
    assert set(cases) == {"cancel_order", "refund_request"}
    assert all(item["status"] == "pending_review" for item in cases.values())

    suggestions = {item["action_type"]: item for item in shadow.recent(limit=10)}
    assert suggestions["cancel_order"]["suggested_outcome"] == "auto_execute"
    assert suggestions["refund_request"]["suggested_outcome"] == "human_review"
    assert suggestions["cancel_order"]["case_id"] == cases["cancel_order"]["case_id"]
    assert shadow.metrics()["summary"]["high_risk_auto_suggestions"] == 0

    unauthorized = await client.get("/shadow/metrics")
    assert unauthorized.status_code == 403
    metrics_response = await client.get("/shadow/metrics", headers=REVIEW_HEADERS)
    assert metrics_response.status_code == 200
    assert metrics_response.json()["enabled"] is True
    assert metrics_response.json()["summary"]["linked_cases"] == 2

    approve = await client.post(
        f"/review/cases/{cases['cancel_order']['case_id']}/approve",
        headers=REVIEW_HEADERS,
        json={"reviewer_id": "shadow-reviewer", "reason": "low risk confirmed"},
    )
    assert approve.status_code == 200, approve.text
    reject = await client.post(
        f"/review/cases/{cases['refund_request']['case_id']}/reject",
        headers=REVIEW_HEADERS,
        json={"reviewer_id": "shadow-reviewer", "reason": "high value requires staff"},
    )
    assert reject.status_code == 200, reject.text

    metrics = shadow.metrics()
    summary = metrics["summary"]
    assert summary["auto_suggestions"] == 1
    assert summary["auto_approved"] == 1
    assert summary["false_auto"] == 0
    assert summary["review_rejected"] == 1
    assert summary["routing_agreement"] == 1.0
    assert metrics["gates"]["high_risk_auto_zero"] is True
    assert metrics["gates"]["false_auto_zero"] is True

    records = await client.get("/shadow/records", headers=REVIEW_HEADERS)
    assert records.status_code == 200
    assert records.json()["records"]
    assert all(len(item["order_id"]) < 32 for item in records.json()["records"])
