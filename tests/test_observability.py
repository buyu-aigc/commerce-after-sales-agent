"""P0-4 observability tests: tool/policy/recovery metrics and runtime endpoint."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

from app import observability
from app import support_conversations as conversations
from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.case_workflow_service import CaseWorkflowService
from app.olist.service import OlistService, SQLiteCaseService
from app.policy.engine import PolicyEngine
from app.policy.guard import PolicyGuard
from app.tool_call import InMemoryRuntimeStore, ToolCallContext, build_business_tool_manager

A = "203096f03d82e0dffbc41ebc2e2bcfb7"  # delivered, delayed, auto-eligible
B = "8aec3a066f732dd927ec8fef1752415b"  # invoiced, low value, cancel auto
C = "360787554f41824600bdcba9687cd4f0"  # high value pre-shipment, review
REVIEW_HEADERS = {"X-Review-Token": "local-review-demo"}
CUSTOMER_HEADERS = {"X-Demo-Customer": "demo-customer"}


@pytest.fixture(autouse=True)
def reset_metrics():
    observability.reset()
    yield
    observability.reset()


@pytest.fixture
def obs_runtime(tmp_path, monkeypatch):
    import app.config.di as di
    import app.main as main_module
    import app.trace_store as trace_store

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
    graph = AgentGraph(actions).build(checkpointer)
    workflow = CaseWorkflowService(case_service, actions)
    workflow.configure(checkpointer, legacy_agent=None)
    actions.bind_case_workflow(workflow)

    monkeypatch.setattr(main_module, "agent", graph)
    monkeypatch.setattr(main_module, "conversation_agent", graph)
    monkeypatch.setattr(main_module, "case_service", case_service)
    monkeypatch.setattr(main_module, "case_workflow_service", workflow)
    monkeypatch.setattr(main_module, "olist_service", olist_service)
    monkeypatch.setattr(main_module, "runtime_store", runtime_store)
    monkeypatch.setattr(main_module, "present", AsyncMock(return_value="客服回复"))
    monkeypatch.setattr(conversations, "DB", tmp_path / "support_conversations.db")
    monkeypatch.setattr(trace_store, "TRACE_DB", tmp_path / "agent_traces.db")
    conversations.initialize()
    return {
        "main": main_module,
        "case_service": case_service,
        "tool_manager": tool_manager,
        "workflow": workflow,
    }


@pytest.fixture
async def client(obs_runtime):
    transport = ASGITransport(app=obs_runtime["main"].app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.mark.anyio
async def test_tool_call_metrics_percentiles_and_error_rate(caplog):
    caplog.set_level(logging.INFO, logger="aipm.observability")
    for latency, ok in ((10.0, True), (20.0, True), (30.0, False)):
        observability.record_tool_call(
            tool="get_order_status",
            latency_ms=latency,
            ok=ok,
            error_code=None if ok else "schema_validation_failed",
            result_count=1,
            redacted_args={"order_id": "203096...cfb7"},
            tenant_id="olist-demo",
            user_id="demo-customer",
            session_id="tool-metrics",
        )
    snapshot = observability.tool_snapshot()
    assert snapshot["total"] == 3
    assert snapshot["errors"] == 1
    assert snapshot["error_rate"] == 0.3333
    assert snapshot["p50_latency_ms"] == 20.0
    assert snapshot["p95_latency_ms"] == 20.0
    by_tool = snapshot["by_tool"]["get_order_status"]
    assert by_tool["calls"] == 3
    assert by_tool["errors"] == 1
    assert by_tool["p50_latency_ms"] == 20.0
    assert '"event": "tool_call"' in caplog.text
    assert '"result_count": 1' in caplog.text


@pytest.mark.anyio
async def test_tool_manager_audit_is_exported(obs_runtime):
    manager = obs_runtime["tool_manager"]
    context = ToolCallContext(
        user_id="demo-customer",
        role="customer",
        tenant_id="olist-demo",
        session_id="tool-audit",
        auth_scopes=["orders:read"],
        allowed_order_ids=[A],
    )
    ok = await manager.call("get_order_status", {"order_id": A}, context)
    failed = await manager.call("get_order_status", {"order_id": "bad"}, context)
    assert ok.ok is True
    assert failed.ok is False

    snapshot = observability.tool_snapshot()
    assert snapshot["by_tool"]["get_order_status"]["calls"] == 2
    assert snapshot["by_tool"]["get_order_status"]["errors"] == 1
    audit = manager.audit_events[-2]
    assert audit["redacted_args"]["order_id"] != A
    assert audit["result_summary"]


def test_policy_decision_and_auto_outcome_metrics():
    observability.record_policy_decision(
        {
            "policy_id": "demo_delay_refund_auto_v1",
            "decision": "auto",
            "policy_version": "1.1.0",
            "reason": "auto allowed",
        },
        action_type="refund_request",
        session_id="policy-metrics",
        user_id="demo-customer",
    )
    observability.record_policy_decision(
        {
            "policy_id": "demo_delay_refund_auto_v1",
            "decision": "review",
            "policy_version": "1.1.0",
            "reason": "amount cap exceeded",
            "guardrail": ["amount_cap_exceeded"],
        },
        action_type="refund_request",
        session_id="policy-metrics",
        user_id="demo-customer",
    )
    observability.record_auto_outcome({"policy_id": "demo_delay_refund_auto_v1"}, executed=True)
    observability.record_auto_outcome({"policy_id": "demo_delay_refund_auto_v1"}, executed=False)

    policy = observability.policy_snapshot()
    assert policy["total"] == 2
    assert policy["auto"] == 1
    assert policy["review"] == 1
    assert policy["auto_rate"] == 0.5
    assert policy["auto_executed"] == 1
    assert policy["auto_failed"] == 1
    assert policy["false_auto"] == 1
    assert policy["guardrail_downgrades"] == {"amount_cap_exceeded": 1}
    assert policy["by_policy"]["demo_delay_refund_auto_v1"] == {"auto": 1, "review": 1}
    assert policy["policy_versions"] == {"1.1.0": 2}


@pytest.mark.anyio
async def test_workflow_recovery_is_counted(obs_runtime):
    await obs_runtime["workflow"].recover_jobs()
    recovery = observability.recovery_snapshot()
    assert recovery["runs"] == 1
    assert recovery["attempted"] == 0

    observability.record_recovery("resume", "case-1", success=True)
    observability.record_recovery("init", "case-2", success=False)
    recovery = observability.recovery_snapshot()
    assert recovery["attempted"] == 2
    assert recovery["succeeded"] == 1
    assert recovery["failed"] == 1
    assert recovery["by_job_type"]["resume"]["succeeded"] == 1
    assert recovery["by_job_type"]["init"]["failed"] == 1


@pytest.mark.anyio
async def test_policy_decisions_are_recorded_from_real_graph(client: AsyncClient):
    auto = await client.post(
        "/customer/chat",
        headers=CUSTOMER_HEADERS,
        json={
            "message": "我想取消这笔订单。",
            "session_id": "obs-auto",
            "requested_action": "cancel_order",
            "page_context": {"surface": "order_detail", "selected_order_id": B},
        },
    )
    assert auto.status_code == 200, auto.text
    review = await client.post(
        "/customer/chat",
        headers=CUSTOMER_HEADERS,
        json={
            "message": "我想为这笔订单申请退款。",
            "session_id": "obs-review",
            "requested_action": "refund_request",
            "page_context": {"surface": "order_detail", "selected_order_id": A},
        },
    )
    assert review.status_code == 200, review.text

    policy = observability.policy_snapshot()
    assert policy["auto"] == 1
    assert policy["review"] == 1
    assert policy["auto_executed"] == 1
    assert policy["auto_failed"] == 0
    assert policy["false_auto"] == 0


@pytest.mark.anyio
async def test_runtime_endpoint_requires_token_and_has_all_sections(
    client: AsyncClient, obs_runtime
):
    from app.trace_store import record_trace

    observability.record_tool_call(tool="get_order_status", latency_ms=12.5, ok=True, result_count=1)
    observability.record_policy_decision(
        {"policy_id": "demo_default_review_v1", "decision": "review", "policy_version": "1.1.0"},
        action_type="refund_request",
    )
    record_trace(
        session_id="obs-endpoint",
        user_message="hello",
        result={"route_intent": "policy", "final_answer": "ok"},
        latency_ms=12.5,
        status="ok",
    )

    denied = await client.get("/observability/runtime")
    assert denied.status_code == 403

    response = await client.get("/observability/runtime", headers=REVIEW_HEADERS)
    assert response.status_code == 200
    body = response.json()
    for key in ("generated_at", "tools", "policy", "recovery", "traces", "cases"):
        assert key in body
    assert body["tools"]["total"] == 1
    assert body["policy"]["review"] == 1
    assert body["recovery"]["runs"] == 0
    assert body["traces"]["p50_latency_ms"] == 12.5
    assert body["traces"]["p95_latency_ms"] == 12.5
    assert "p50_latency_ms" in body["cases"]


@pytest.mark.anyio
async def test_trace_summary_exposes_p50_and_p95(client: AsyncClient, obs_runtime):
    from app.trace_store import record_trace, trace_summary

    for latency in (10.0, 20.0, 30.0):
        record_trace(
            session_id=f"p50-{latency}",
            user_message="hello",
            result={"route_intent": "policy", "final_answer": "ok"},
            latency_ms=latency,
            status="ok",
        )
    summary = trace_summary()
    assert summary["p50_latency_ms"] == 20.0
    assert summary["p95_latency_ms"] == 20.0
