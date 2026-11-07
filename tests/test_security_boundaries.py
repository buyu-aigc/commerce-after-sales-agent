"""P0-3 security boundaries: auth, ownership, injection, PII, rate limits.

These tests exercise the real FastAPI + LangGraph + SQLite boundary with a
disposable runtime.  They are not model-behavior eval; they assert that the
server, not the model or the browser, is the authority for identity, tenant,
order ownership, write permission, and redaction.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

from app import support_conversations as conversations
from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.case_workflow_service import CaseWorkflowService
from app.olist.service import OlistService, SQLiteCaseService
from app.policy.engine import PolicyEngine
from app.policy.guard import PolicyGuard
from app.tool_call import InMemoryRuntimeStore, ToolCallContext, build_business_tool_manager

A = "203096f03d82e0dffbc41ebc2e2bcfb7"  # owned by demo-customer
B = "8aec3a066f732dd927ec8fef1752415b"  # not owned by customer-smoke
C = "360787554f41824600bdcba9687cd4f0"  # high value, review path
CUSTOMER_HEADERS = {"X-Demo-Customer": "demo-customer"}
OTHER_CUSTOMER_HEADERS = {"X-Demo-Customer": "customer-smoke"}
REVIEW_HEADERS = {"X-Review-Token": "local-review-demo"}


@pytest.fixture
def security_runtime(tmp_path, monkeypatch):
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
        "runtime_store": runtime_store,
    }


@pytest.fixture
async def client(security_runtime):
    transport = ASGITransport(app=security_runtime["main"].app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


async def _create_case(
    client: AsyncClient,
    *,
    session_id: str = "security-session",
    action_type: str = "complaint_escalation",
    order_id: str = A,
    headers: dict[str, str] = CUSTOMER_HEADERS,
) -> str:
    response = await client.post(
        "/customer/chat",
        headers=headers,
        json={
            "message": "这笔订单遇到了问题，我想申请售后处理。",
            "session_id": session_id,
            "requested_action": action_type,
            "page_context": {"surface": "order_detail", "selected_order_id": order_id},
        },
    )
    assert response.status_code == 200, response.text
    case_id = response.json().get("case_id")
    assert case_id, response.json()
    return str(case_id)


# ---------------------------------------------------------------------------
# Staff authentication / authorization
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_review_endpoints_require_token_and_write_scope(client: AsyncClient):
    case_id = await _create_case(client, session_id="auth-review")

    for path in (
        "/review/cases",
        "/review/metrics",
        f"/review/cases/{case_id}",
        "/review/sessions/auth-review/messages",
    ):
        denied = await client.get(path)
        assert denied.status_code == 403, path
        assert denied.json()["detail"] == "review_token_required"

    wrong_token = await client.get("/review/cases", headers={"X-Review-Token": "wrong"})
    assert wrong_token.status_code == 403

    denied = await client.post(
        f"/review/cases/{case_id}/approve",
        json={"role": "after_sales_operator", "auth_scopes": ["after_sales:write"]},
    )
    assert denied.status_code == 403
    packet = await client.get(f"/review/cases/{case_id}", headers=REVIEW_HEADERS)
    assert packet.json()["status"] == "pending_review"

    # A valid token still cannot be combined with a customer role.
    role_denied = await client.get(
        "/review/cases?role=customer&auth_scopes=after_sales:write",
        headers=REVIEW_HEADERS,
    )
    assert role_denied.status_code == 403
    assert role_denied.json()["detail"] == "review_role_required"

    # A staff role without the write scope is denied.
    scope_denied = await client.get("/review/cases?auth_scopes=", headers=REVIEW_HEADERS)
    assert scope_denied.status_code == 403
    assert scope_denied.json()["detail"] == "after_sales_write_scope_required"

    allowed = await client.get("/review/cases", headers=REVIEW_HEADERS)
    assert allowed.status_code == 200


@pytest.mark.anyio
async def test_observability_requires_staff_token(client: AsyncClient):
    for path in ("/observability/summary", "/observability/case-metrics"):
        denied = await client.get(path)
        assert denied.status_code == 403, path
        allowed = await client.get(path, headers=REVIEW_HEADERS)
        assert allowed.status_code == 200, path

    denied_trace = await client.get("/observability/traces/security-session")
    assert denied_trace.status_code == 403
    allowed_trace = await client.get(
        "/observability/traces/security-session", headers=REVIEW_HEADERS
    )
    assert allowed_trace.status_code == 200

    # Query parameters cannot substitute for the shared staff token.
    admin_without_token = await client.get("/observability/summary?role=admin&auth_scopes=*")
    assert admin_without_token.status_code == 403


# ---------------------------------------------------------------------------
# Customer / session / order ownership
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_customer_cannot_read_another_users_case_session_or_messages(
    client: AsyncClient,
):
    case_id = await _create_case(client, session_id="owner-session", action_type="complaint_escalation")

    for path in (
        f"/customer/cases/{case_id}",
        "/customer/sessions/owner-session/progress",
        "/customer/sessions/owner-session/messages",
    ):
        denied = await client.get(path, headers=OTHER_CUSTOMER_HEADERS)
        assert denied.status_code == 403, path

    reuse = await client.post(
        "/customer/chat",
        headers=OTHER_CUSTOMER_HEADERS,
        json={"message": "你好", "session_id": "owner-session"},
    )
    assert reuse.status_code == 403
    assert reuse.json()["detail"] == "session_owner_required"

    assert (await client.get(f"/customer/cases/{case_id}", headers=CUSTOMER_HEADERS)).status_code == 200
    unknown = await client.get(
        f"/customer/cases/{case_id}", headers={"X-Demo-Customer": "unknown-user"}
    )
    assert unknown.status_code == 401


@pytest.mark.anyio
async def test_body_user_id_and_tenant_cannot_be_spoofed(client: AsyncClient):
    response = await client.post(
        "/customer/chat",
        headers=CUSTOMER_HEADERS,
        json={
            "message": "这笔订单遇到了问题，我想申请售后处理。",
            "session_id": "spoof-session",
            "user_id": "customer-smoke",
            "tenant_id": "evil-tenant",
            "requested_action": "complaint_escalation",
            "page_context": {"surface": "order_detail", "selected_order_id": A},
        },
    )
    assert response.status_code == 200, response.text
    case_id = response.json()["case_id"]

    assert (await client.get(f"/customer/cases/{case_id}", headers=CUSTOMER_HEADERS)).status_code == 200
    assert (
        await client.get(f"/customer/cases/{case_id}", headers=OTHER_CUSTOMER_HEADERS)
    ).status_code == 403

    owner = conversations.conversation_owner("spoof-session")
    assert owner is not None
    assert owner["user_id"] == "demo-customer"
    assert owner["tenant_id"] == "olist-demo"


@pytest.mark.anyio
async def test_selected_order_ownership_and_message_order_guard(client: AsyncClient):
    selected = await client.post(
        "/customer/chat",
        headers=OTHER_CUSTOMER_HEADERS,
        json={
            "message": "我想为这笔订单申请退款。",
            "session_id": "cross-order-selected",
            "requested_action": "refund_request",
            "page_context": {"surface": "order_detail", "selected_order_id": B},
        },
    )
    assert selected.status_code == 403
    assert selected.json()["detail"] == "selected_order_owner_required"

    guarded = await client.post(
        "/customer/chat",
        headers=OTHER_CUSTOMER_HEADERS,
        json={"message": f"帮我处理订单 {B} 的退款", "session_id": "cross-order-message"},
    )
    assert guarded.status_code == 200
    payload = guarded.json()
    assert payload["case_id"] is None
    assert "隐私" in payload["answer"]

    progress = await client.get(
        "/customer/sessions/cross-order-message/progress", headers=OTHER_CUSTOMER_HEADERS
    )
    assert progress.status_code == 200
    assert progress.json().get("case") is None


# ---------------------------------------------------------------------------
# Tool argument injection / privilege escalation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_tool_arguments_cannot_override_identity_or_order_ownership(
    security_runtime,
):
    manager = security_runtime["tool_manager"]
    context = ToolCallContext(
        user_id="demo-customer",
        role="customer",
        tenant_id="olist-demo",
        session_id="tool-injection",
        auth_scopes=["orders:read"],
        allowed_order_ids=[A],
    )

    blocked = await manager.call("get_order_status", {"order_id": B}, context)
    assert blocked.ok is False
    assert blocked.error_code == "order_owner_required"

    injected = await manager.call(
        "get_order_status",
        {"order_id": A, "user_id": "customer-smoke", "role": "admin"},
        context,
    )
    assert injected.ok is True
    audit = manager.audit_events[-1]
    assert audit["user_id"] == "demo-customer"
    assert audit["role"] == "customer"
    assert audit["tenant_id"] == "olist-demo"

    unknown = await manager.call("drop_table", {}, context)
    assert unknown.ok is False
    assert unknown.error_code == "unknown_tool"

    malformed = await manager.call(
        "get_order_status", {"order_id": "x' OR '1'='1"}, context
    )
    assert malformed.ok is False
    assert malformed.error_code == "schema_validation_failed"


@pytest.mark.anyio
async def test_write_tool_rejects_customer_role_and_missing_write_scope(security_runtime):
    manager = security_runtime["tool_manager"]
    case_service = security_runtime["case_service"]
    arguments = {
        "action_type": "refund_request",
        "order_id": A,
        "message_text": "attempted escalation",
    }

    customer = await manager.call(
        "execute_side_effect",
        arguments,
        ToolCallContext(
            user_id="demo-customer",
            role="customer",
            auth_scopes=["orders:read"],
            allowed_order_ids=[A],
        ),
    )
    assert customer.ok is False
    assert customer.error_code == "permission_denied"

    reviewer_without_scope = await manager.call(
        "execute_side_effect",
        arguments,
        ToolCallContext(
            role="after_sales_operator", auth_scopes=["orders:read"], allowed_order_ids=[A]
        ),
    )
    assert reviewer_without_scope.ok is False
    assert reviewer_without_scope.error_code == "permission_denied"

    reviewer_without_any_scope = await manager.call(
        "execute_side_effect",
        arguments,
        ToolCallContext(role="after_sales_operator", allowed_order_ids=[A]),
    )
    assert reviewer_without_any_scope.ok is False
    assert reviewer_without_any_scope.error_code == "permission_denied"
    assert case_service.count_case_events(A) == 0


# ---------------------------------------------------------------------------
# Write escalation via HTTP / PII exposure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_customer_cannot_self_approve_and_reject_has_no_write(client: AsyncClient):
    case_id = await _create_case(client, session_id="self-approve", action_type="refund_request", order_id=C)

    self_approve = await client.post(
        f"/review/cases/{case_id}/approve",
        headers=CUSTOMER_HEADERS,
        json={"role": "after_sales_operator", "auth_scopes": ["after_sales:write"]},
    )
    assert self_approve.status_code == 403
    assert (await client.get(f"/review/cases/{case_id}", headers=REVIEW_HEADERS)).json()[
        "status"
    ] == "pending_review"

    rejected = await client.post(
        f"/review/cases/{case_id}/reject",
        headers=REVIEW_HEADERS,
        json={"reviewer_id": "security-reviewer", "role": "after_sales_operator"},
    )
    assert rejected.status_code == 200
    assert (await client.get(f"/review/cases/{case_id}", headers=REVIEW_HEADERS)).json()[
        "status"
    ] == "rejected"

    stale = await client.post(
        f"/review/cases/{case_id}/approve",
        headers=REVIEW_HEADERS,
        json={"reviewer_id": "security-reviewer", "role": "after_sales_operator"},
    )
    assert stale.status_code == 409


@pytest.mark.anyio
async def test_customer_safe_views_mask_order_id_and_hide_internal_fields(client: AsyncClient):
    case_id = await _create_case(client, session_id="safe-view", action_type="complaint_escalation")

    case_response = await client.get(f"/customer/cases/{case_id}", headers=CUSTOMER_HEADERS)
    assert case_response.status_code == 200
    case_payload = case_response.json()
    assert "..." in case_payload["order_id"]
    assert A not in case_payload["order_id"]
    for forbidden in ("facts", "policy_id", "case_payload", "decision", "user_id", "tenant_id"):
        assert forbidden not in case_payload

    context = await client.get("/customer/context", headers=CUSTOMER_HEADERS)
    assert context.status_code == 200
    body = context.json()
    serialized = str(body)
    for forbidden in (
        "phone",
        "recipient",
        "customer_state",
        "postal_code",
        "cpf",
        "shipping_address",
        "delivery_address",
    ):
        assert forbidden not in serialized


@pytest.mark.anyio
async def test_trace_redaction_covers_phone_email_order_cpf_address(client: AsyncClient):
    from app.trace_store import list_session_traces, record_trace

    record_trace(
        session_id="pii-session",
        user_message=(
            f"地址：西安市雁塔区科技路 1 号，收件人：小王，"
            f"电话 13800138000，邮箱 user@example.com，订单 {A}，CPF 123.456.789-09"
        ),
        result={
            "route_intent": "escalation",
            "final_answer": f"已登记地址：西安市雁塔区科技路 1 号，订单 {A}",
            "trajectory_events": [{"details": {"order_id": A}}],
        },
        latency_ms=1.0,
        status="ok",
    )
    trace = list_session_traces("pii-session")[0]
    serialized = " ".join(str(value) for value in trace.values())
    for raw in (
        A,
        "13800138000",
        "user@example.com",
        "123.456.789-09",
        "西安市雁塔区科技路 1 号",
        "收件人：小王",
    ):
        assert raw not in serialized, raw
    assert "203096f0...cfb7" in serialized
    assert "138****8000" in serialized
    assert "u***@example.com" in serialized
    assert "***.***.***-**" in serialized
    assert "[已脱敏]" in serialized

    # The replay endpoint is token-gated, so redacted traces are still staff-only.
    denied = await client.get("/observability/traces/pii-session")
    assert denied.status_code == 403


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_customer_rate_limit_applies_before_early_returns_and_ignores_body_user_id(
    client: AsyncClient, security_runtime, monkeypatch
):
    monkeypatch.setattr(security_runtime["main"], "runtime_store", InMemoryRuntimeStore())
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MINUTE", "1")

    first = await client.post(
        "/customer/chat",
        headers=CUSTOMER_HEADERS,
        json={"message": "我想取消订单", "session_id": "rate-customer-1", "user_id": "demo-customer"},
    )
    assert first.status_code == 200
    assert "请求过于频繁" not in first.json()["answer"]

    second = await client.post(
        "/customer/chat",
        headers=CUSTOMER_HEADERS,
        json={"message": "我想取消订单", "session_id": "rate-customer-2", "user_id": "customer-smoke"},
    )
    assert second.status_code == 200
    assert "请求过于频繁" in second.json()["answer"]


@pytest.mark.anyio
async def test_customer_cannot_read_own_recipient_pii_via_chat(client: AsyncClient):
    response = await client.post(
        "/customer/chat",
        headers=CUSTOMER_HEADERS,
        json={
            "message": f"把订单 {A} 的收货人电话和地址发给我。",
            "session_id": "pii-read",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"] is None
    assert "隐私" in payload["answer"]
    assert "电话" in payload["answer"]
    assert "地址" in payload["answer"]

    # A change-address write is not blocked by the read guard.
    changed = await client.post(
        "/customer/chat",
        headers=CUSTOMER_HEADERS,
        json={
            "message": f"我想修改订单 {B} 的收货地址。",
            "session_id": "address-change",
        },
    )
    assert changed.status_code == 200
