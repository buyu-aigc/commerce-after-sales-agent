import asyncio
import hashlib
import json
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import observability
from app import support_conversations as conversations
from app.after_sales.decision import AfterSalesDecisionEngine
from app.agent import order_discovery, semantic_order_grounding
from app.agent.actions import _customer_category_label, _customer_order_status_label
from app.agent.state import AgentState
from app.config.di import (
    agent_graph_builder,
    case_service,
    case_workflow_service,
    customer_order_agent,
    guardrail,
    llm_client,
    olist_service,
    runtime_status,
    runtime_store,
    shadow_service,
)
from app.config.di import policy_engine as _policy_engine
from app.evaluation_snapshot import load_evaluation_snapshot
from app.intent.turn_contract import ACTIVE_TURN, CustomerTurnPlanner
from app.llm.customer_presenter import present
from app.logger import format_state, setup_logger
from app.models import (
    CaseMetricsResponse,
    ChatRequest,
    ChatResponse,
    CustomerAction,
    CustomerAppealRequest,
    CustomerCaseResponse,
    DemoReviewRequest,
    DemoScenarioRequest,
    ReviewActionRequest,
    ReviewCaseListResponse,
    ReviewSessionResponse,
    RuntimeObservabilityResponse,
    RuntimeStatusResponse,
    TraceReplayResponse,
    TraceSummaryResponse,
)
from app.olist.category_labels import CATEGORY_LABELS
from app.olist.status import PRE_SHIPMENT_STATUSES
from app.policy.guard import PolicyGuard
from app.trace_store import case_metrics, list_session_traces, record_trace, trace_summary

agent: StateGraph | None = None
conversation_agent: StateGraph | None = None
logger = setup_logger("agent")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ORDER_ID_RE = re.compile(r"\b[0-9a-fA-F]{32}\b")
DEFAULT_CUSTOMER_ORDER_IDS: dict[str, set[str]] = {
    "demo-customer": {
        "203096f03d82e0dffbc41ebc2e2bcfb7",  # delivered, delayed
        "8aec3a066f732dd927ec8fef1752415b",  # low-value, pre-shipment
        "360787554f41824600bdcba9687cd4f0",  # high-value, pre-shipment
        "1b9ecfe83cdc259250e1a8aca174f0ad",  # canceled
        "f98ae4bbfd1a32ea2104c8cf1f64dabf",  # delivered, no delay
    },
    "customer-demo": {
        "203096f03d82e0dffbc41ebc2e2bcfb7",
        "8aec3a066f732dd927ec8fef1752415b",
        "360787554f41824600bdcba9687cd4f0",
        "1b9ecfe83cdc259250e1a8aca174f0ad",
        "f98ae4bbfd1a32ea2104c8cf1f64dabf",
    },
    "customer-smoke": {"203096f03d82e0dffbc41ebc2e2bcfb7"},
}

# The guided demo deliberately uses the same public Olist snapshots as the
# product surface, but starts from a clean local case store for every run.
# It is disabled outside a local/sandbox deployment (DEMO_MODE=0).
DEMO_SCENARIOS: dict[str, dict[str, str]] = {
    "delayed_refund": {
        "title": "配送异常，申请退款",
        "summary": "订单延迟且评价较低，系统整理证据并交给售后审核。",
        "order_id": "203096f03d82e0dffbc41ebc2e2bcfb7",
        "message": "这笔订单晚到了很多天，商品体验也不好，我想申请退款。",
        "requested_action": "refund_request",
        "expected_outcome": "review",
    },
    "track_delivery": {
        "title": "查询配送进度",
        "summary": "标准查询直接返回订单事实，不占用人工审核。",
        "order_id": "8aec3a066f732dd927ec8fef1752415b",
        "message": "帮我看看这笔订单现在配送到哪里了。",
        "requested_action": "track_order",
        "expected_outcome": "resolved",
    },
    "delivered_address": {
        "title": "已送达后修改地址",
        "summary": "订单已送达，系统说明限制并阻止无效写入。",
        "order_id": "203096f03d82e0dffbc41ebc2e2bcfb7",
        "message": (
            "这笔订单已经送达，但我想改到新地址：西安市雁塔区科技路 1 号，收件人小王，电话 13800138000。"
        ),
        "requested_action": "change_address",
        "expected_outcome": "blocked",
    },
}
DEMO_SESSIONS: dict[str, str] = {}

DEFAULT_ROLE_SCOPES: dict[str, list[str]] = {
    "customer": [
        "orders:read",
        "policy:read",
        "support_examples:read",
        "after_sales:assess",
        "after_sales:draft",
    ],
    "support_agent": [
        "orders:read",
        "analytics:read",
        "policy:read",
        "support_examples:read",
        "after_sales:assess",
        "after_sales:draft",
    ],
    "ops_manager": [
        "orders:read",
        "analytics:read",
        "policy:read",
        "support_examples:read",
        "after_sales:ops_report",
    ],
    "after_sales_operator": [
        "orders:read",
        "analytics:read",
        "policy:read",
        "support_examples:read",
        "after_sales:assess",
        "after_sales:draft",
        "after_sales:write",
    ],
    "admin": ["*"],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: manage AsyncSqliteSaver lifecycle."""
    global agent, conversation_agent
    await asyncio.to_thread(conversations.initialize)
    async with AsyncSqliteSaver.from_conn_string("data/checkpoints.db") as checkpointer:
        agent = agent_graph_builder.build(checkpointer=checkpointer)
        conversation_agent = agent_graph_builder.build(checkpointer=checkpointer)
        case_workflow_service.configure(checkpointer, legacy_agent=agent)
        await case_workflow_service.recover_jobs()
        logger.info("Checkpointer ready (AsyncSqliteSaver: data/checkpoints.db)")

        async def recover_periodically():
            while True:
                await asyncio.sleep(5)
                try:
                    await case_workflow_service.recover_jobs()
                except Exception:
                    logger.exception("Case recovery scan failed; retrying")

        recovery_task = asyncio.create_task(recover_periodically())
        try:
            yield
        finally:
            recovery_task.cancel()
            with suppress(asyncio.CancelledError):
                await recovery_task
    logger.info("Checkpointer closed")


app = FastAPI(title="Olist Marketplace Support Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="static")


@app.get("/", response_class=FileResponse)
def app_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "customer" / "index.html")


@app.get("/customer", response_class=FileResponse)
def customer_app() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "customer" / "index.html")


@app.get("/demo", response_class=FileResponse)
def demo_app() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "guided-demo" / "index.html")


@app.get("/demo/scenarios")
def demo_scenarios() -> dict:
    """Describe the fixed walkthroughs without exposing internal tool details."""
    _require_demo_mode()
    return {
        "scenarios": [
            {"id": scenario_id, **{key: value for key, value in scenario.items() if key != "message"}}
            for scenario_id, scenario in DEMO_SCENARIOS.items()
        ]
    }


@app.post("/demo/run")
async def run_demo_scenario(request: DemoScenarioRequest) -> dict:
    """Execute a disposable scenario through the real customer workflow.

    This endpoint is intentionally unavailable with ``DEMO_MODE=0``. It is a
    convenience layer for a guided local showcase, not an alternate business
    API or a way to bypass staff authorization in a deployed service.
    """
    _require_demo_mode()
    scenario = DEMO_SCENARIOS.get(request.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="demo_scenario_not_found")

    # A showcase must be repeatable. The store only contains local sandbox
    # effects, and each click starts a fresh graph session against the same
    # immutable public-data snapshot.
    await asyncio.to_thread(case_service.reset)
    session_id = f"guided-demo-{uuid.uuid4().hex}"
    DEMO_SESSIONS.clear()
    DEMO_SESSIONS[session_id] = request.scenario_id
    response = await customer_chat(
        ChatRequest(
            message=scenario["message"],
            session_id=session_id,
            requested_action=scenario["requested_action"],
            page_context={"surface": "guided_demo", "selected_order_id": scenario["order_id"]},
        ),
        x_demo_customer="customer-demo",
    )
    return await _demo_session_payload(session_id, scenario, customer_answer=response.answer)


@app.post("/demo/sessions/{session_id}/review")
async def review_demo_scenario(session_id: str, request: DemoReviewRequest) -> dict:
    """Resume a guided-demo HITL checkpoint as the local demo reviewer."""
    _require_demo_mode()
    scenario_id = DEMO_SESSIONS.get(session_id)
    if scenario_id is None:
        raise HTTPException(status_code=404, detail="demo_session_not_found")
    status = "review_approved" if request.decision == "approve" else "review_rejected"
    resume_value = "__review_approve__" if request.decision == "approve" else "no"
    case = _single_actionable_case(session_id)
    if case is not None and case_workflow_service.configured and case.get("workflow_thread_id"):
        response = await _resume_case_review(
            str(case["case_id"]),
            decision="approve" if request.decision == "approve" else "reject",
            reviewer_id="guided-demo-reviewer",
        )
    else:
        response = await _resume_review_session(
            session_id,
            resume_value=resume_value,
            status=status,
            reviewer_id="guided-demo-reviewer",
        )
    messages = await asyncio.to_thread(conversations.messages, session_id)
    customer_answer = next(
        (str(item.get("content", "")) for item in reversed(messages) if item.get("sender") == "staff"),
        response.answer,
    )
    return await _demo_session_payload(
        session_id,
        DEMO_SCENARIOS[scenario_id],
        customer_answer=customer_answer,
        review_decision=request.decision,
    )


@app.get("/technical", response_class=FileResponse)
def technical_app() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "demo.html")


@app.get("/customer/context")
def customer_context(x_demo_customer: str | None = Header(default=None)) -> dict:
    """Return a minimal, ownership-filtered order context for the customer UI.

    ``X-Demo-Customer`` is only a local stand-in for the actor resolved by an
    authenticated BFF in production.  The browser never supplies a user id in
    the request body for this route or ``/customer/chat``.
    """
    return _customer_context_payload(_resolve_customer_actor(x_demo_customer))


@app.get("/customer/orders")
def customer_orders(x_demo_customer: str | None = Header(default=None)) -> dict:
    """Compatibility endpoint for the order rail in the customer product UI."""
    payload = _customer_context_payload(_resolve_customer_actor(x_demo_customer))
    return {"orders": payload["orders"]}


@app.get("/customer/sessions/{session_id}/progress")
async def customer_progress(
    session_id: str,
    x_demo_customer: str | None = Header(default=None),
) -> dict:
    customer_user_id = _resolve_customer_actor(x_demo_customer)
    owner = await asyncio.to_thread(conversations.conversation_owner, session_id)
    if owner is not None and str(owner.get("user_id")) != customer_user_id:
        raise HTTPException(status_code=403, detail="conversation_owner_required")
    records = await asyncio.to_thread(case_service.list_by_session, session_id)
    if not records:
        graph = conversation_agent or agent
        snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
        if not snapshot.values:
            return {"case": None}
        if snapshot.values.get("user_id") != customer_user_id:
            raise HTTPException(status_code=403, detail="conversation_owner_required")
        values = snapshot.values or {}
        case_id = (values.get("escalation_draft") or {}).get("review_case_id")
        record = await asyncio.to_thread(case_service.get, case_id) if case_id else None
        records = [record] if record else []
    if not records:
        return {"case": None}
    if str(records[0].get("user_id") or "demo-customer") != customer_user_id:
        raise HTTPException(status_code=403, detail="case_owner_required")
    safe_cases = []
    for record in records:
        safe = _customer_case_response(record).model_dump()
        safe.pop("message_text", None)
        safe_cases.append(safe)
    summary = _progress_summary(safe_cases)
    return {
        "case": safe_cases[-1],
        "cases": safe_cases,
        "summary": summary,
        "groups": _progress_groups(safe_cases),
        "request": _request_progress(safe_cases, summary),
    }


@app.get("/assistant", response_class=FileResponse)
def assistant_app() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "assistant" / "index.html")


@app.get("/review", response_class=FileResponse)
def review_app() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "review" / "index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    *,
    graph=None,
    decouple_cases: bool = False,
    rate_limit_checked: bool = False,
) -> ChatResponse:
    """Internal/debug entrypoint. Product traffic should use /customer/chat."""
    session_id: str = request.session_id or str(uuid.uuid4())
    graph_to_use = graph or agent
    config: dict = {"configurable": {"thread_id": session_id}}
    auth_scopes = _resolve_auth_scopes(request.role, request.auth_scopes)

    t_start = time.perf_counter()
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("REQUEST | session=%s | message=%s", session_id, request.message[:100])

    if not rate_limit_checked:
        rate_limit = await _check_rate_limit(request)
        if rate_limit is not None and not rate_limit.allowed:
            return _rate_limited_chat_response(
                request,
                rate_limit,
                session_id=session_id,
                latency_ms=(time.perf_counter() - t_start) * 1000,
            )

    # Check if there's a pending interrupt for this session
    snapshot = await graph_to_use.aget_state(config)
    has_interrupt = bool(snapshot.next)
    history_messages = snapshot.values.get("messages") if snapshot.values else None
    if _is_customer_request(request):
        visible_history = await asyncio.to_thread(conversations.messages, session_id)
        if visible_history:
            history_messages = [
                {"role": "user" if item["sender"] == "customer" else "assistant", "content": item["content"]}
                for item in visible_history[-8:]
            ]
            # The ledger keeps the customer's original text. Execute this turn's
            # validated, grounded request without losing internal order bindings.
            # A staff notification (e.g. a case the previous click executed) can be
            # delivered after this turn's message was recorded; the turn's own message
            # must still come last, because the graph treats the last message as the turn.
            last_customer = next(
                (
                    i
                    for i in range(len(history_messages) - 1, -1, -1)
                    if history_messages[i]["role"] == "user"
                ),
                None,
            )
            if last_customer is not None:
                history_messages.pop(last_customer)
            history_messages.append({"role": "user", "content": request.message})

    # Input guardrail with conversation history for context. A validated contract plan has
    # already classified the turn (outside_scope gets a fixed reply), so a second model
    # call must not veto it; deterministic blocking still applies.
    t0 = time.perf_counter()
    if ACTIVE_TURN.get() is not None:
        input_check = guardrail.check_input_deterministic(request.message)
    else:
        input_check = await guardrail.check_input(request.message, history=history_messages)
    logger.info(
        "GUARD INPUT  [%.2fs] | on_topic=%s | reason=%s",
        time.perf_counter() - t0,
        input_check.on_topic,
        input_check.reason,
    )

    if not input_check.on_topic:
        logger.info("DONE [%.2fs] | rejected by input guardrail", time.perf_counter() - t_start)
        rejected_result = {
            "route_intent": "input_guard",
            "final_answer": (
                "I can only help with e-commerce after-sales support: order status, "
                "policy questions, and refund/cancellation/address/invoice/complaint requests."
            ),
        }
        record_trace(
            session_id=session_id,
            user_message=request.message,
            result=rejected_result,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            status="input_guard_rejected",
        )
        return ChatResponse(
            answer=rejected_result["final_answer"],
            session_id=session_id,
            sources=[],
            request_analysis={"route_intent": "input_guard"},
        )

    # Invoke graph - resume from interrupt or start new run
    t_graph = time.perf_counter()
    if has_interrupt:
        if await _pending_confirmation_expired(session_id, snapshot.values):
            logger.info("GRAPH RESUME | pending side effect expired, canceling")
            result = await graph_to_use.ainvoke(Command(resume="__hitl_timeout__"), config)
            record_trace(
                session_id=session_id,
                user_message=request.message,
                result=result,
                latency_ms=(time.perf_counter() - t_start) * 1000,
                status="hitl_timeout_canceled",
            )
            return ChatResponse(
                answer=str(result["final_answer"]),
                session_id=session_id,
                sources=_response_sources(result),
            )
        if _is_customer_request(request):
            pending_result = _pending_confirmation_response(snapshot.values, customer_view=True)
            record_trace(
                session_id=session_id,
                user_message=request.message,
                result=pending_result,
                latency_ms=(time.perf_counter() - t_start) * 1000,
                status="pending_staff_review",
            )
            return ChatResponse(
                answer=str(pending_result["final_answer"]),
                session_id=session_id,
                sources=[],
            )
        if not _is_confirmation_reply(request.message):
            pending_result = _pending_confirmation_response(snapshot.values)
            record_trace(
                session_id=session_id,
                user_message=request.message,
                result=pending_result,
                latency_ms=(time.perf_counter() - t_start) * 1000,
                status="pending_confirmation",
            )
            return ChatResponse(
                answer=str(pending_result["final_answer"]),
                session_id=session_id,
                sources=[],
            )
        logger.info("GRAPH RESUME | resuming from interrupt")
        result = await graph_to_use.ainvoke(Command(resume=request.message), config)
    else:
        # A checkpoint contains both durable conversation context and transient
        # per-turn execution buffers. Only carry the small amount of context
        # that can help a follow-up; old task results must not enter a new turn.
        previous_dialog = dict((snapshot.values or {}).get("dialog_state") or {})
        explicit_order = ORDER_ID_RE.search(request.message)
        selected_order_id = (
            request.page_context.selected_order_id
            or (explicit_order.group(0) if explicit_order else "")
            or str(previous_dialog.get("active_order_id") or "")
        )
        if _is_customer_request(request) and selected_order_id not in _allowed_customer_order_ids(
            request.user_id
        ):
            selected_order_id = ""
        dialog_state = {
            "active_order_id": selected_order_id,
            "filled_slots": {"order_id": selected_order_id} if selected_order_id else {},
            "turn_count": _history_user_turn_count(history_messages),
        }
        initial_state: AgentState = {
            "session_id": session_id,
            "conversation_id": session_id,
            "decouple_cases": decouple_cases,
            "case_id": "",
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "role": request.role,
            "auth_scopes": auth_scopes,
            "channel": request.channel,
            "page_context": request.page_context.model_dump(),
            "requested_action": request.requested_action or "",
            "validated_turn_plan": ACTIVE_TURN.get(),
            "active_order_id": selected_order_id,
            "allowed_order_ids": sorted(_allowed_customer_order_ids(request.user_id))
            if _is_customer_request(request)
            else [],
            "messages": history_messages
            if _is_customer_request(request) and history_messages
            else [{"role": "user", "content": request.message}],
            "route_intent": "",
            "task_plan": [],
            "completed_tasks": [],
            "current_task_index": 0,
            "current_task": {},
            "current_task_text": "",
            "current_slots": {},
            "current_context": {},
            "dialog_state": dialog_state,
            "filled_slots": dict(dialog_state["filled_slots"]),
            "missing_slots": [],
            "handoff_reasons": [],
            "turn_count": int(dialog_state["turn_count"]),
            "failure_count": 0,
            "answer_parts": [],
            "customer_read_results": [],
            "workflow_complete": False,
            "trajectory_events": [],
            "artifacts": {},
            "retrieved_insights": [],
            "retrieved_policy": [],
            "retrieved_support_docs": [],
            "after_sales_cases": [],
            "escalation_draft": {},
            "pending_side_effect": {},
            "user_response": "",
            "final_answer": "",
            "shadow_mode": bool(shadow_service.should_shadow(session_id, request.tenant_id))
            if _is_customer_request(request)
            else False,
        }
        logger.info("GRAPH INPUT%s", format_state(initial_state))
        result = await graph_to_use.ainvoke(initial_state, config)

    logger.info(
        "GRAPH OUTPUT [%.2fs]%s",
        time.perf_counter() - t_graph,
        format_state(result),
    )

    # Output guardrail - LLM validates the response before returning to user
    t1 = time.perf_counter()
    output_check = await guardrail.check_output(result.get("final_answer", ""))
    logger.info(
        "GUARD OUTPUT [%.2fs] | valid=%s | reason=%s",
        time.perf_counter() - t1,
        output_check.valid,
        output_check.reason,
    )

    if not output_check.valid:
        logger.warning("GUARD OUTPUT rejected response: %s", output_check.reason)
        record_trace(
            session_id=session_id,
            user_message=request.message,
            result=result,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            status="output_guard_rejected",
        )
        return ChatResponse(
            answer="I'm sorry, I ran into an issue processing your request. Could you try again or rephrase?",
            session_id=session_id,
            sources=_response_sources(result),
        )

    logger.info("DONE [%.2fs] | total request time", time.perf_counter() - t_start)

    record_trace(
        session_id=session_id,
        user_message=request.message,
        result=result,
        latency_ms=(time.perf_counter() - t_start) * 1000,
        status="ok",
    )

    return ChatResponse(
        answer=result["final_answer"],
        session_id=session_id,
        sources=_response_sources(result),
        case_id=str(
            result.get("case_id") or (result.get("escalation_draft") or {}).get("review_case_id") or ""
        )
        or None,
        request_analysis=_safe_request_analysis(result.get("request_analysis")),
    )


CONFIRMATION_ACTIONS = {"confirm_application", "decline_application", "reselect_application"}
_WRITE_ACTION_NAMES = {
    "refund_request": "退款申请",
    "cancel_order": "取消订单",
    "change_address": "修改收货地址",
    "invoice_request": "开票申请",
    "complaint_escalation": "投诉",
    "open_support_case": "售后跟进",
    "withdraw_case": "撤回申请",
}
_WITHDRAWABLE_STATUSES = {"pending_review", "appealed_pending_review"}


def _withdrawable(task: dict) -> bool:
    record = case_service.get(str(task.get("case_id") or ""))
    return (
        bool(record) and record.get("status") in _WITHDRAWABLE_STATUSES and not record.get("review_decision")
    )


def _withdrawn_case_name(task: dict) -> str:
    record = case_service.get(str(task.get("case_id") or "")) or {}
    return _WRITE_ACTION_NAMES.get(str(record.get("action_type") or ""), "申请")


def _perform_withdrawal(task: dict, user_id: str, session_id: str) -> str:
    """Atomic: only a case nobody has decided can be withdrawn; afterwards approval conflicts."""
    name = _withdrawn_case_name(task)
    outcome = case_service.withdraw_case(
        str(task.get("case_id") or ""), user_id=user_id, session_id=session_id
    )
    if outcome.get("ok"):
        return f"已撤回您的{name}，工作人员不会再处理它。"
    if outcome.get("conflict"):
        return f"这项{name}已经进入审核或已处理完成，无法撤回；可以查看进度或联系人工客服。"
    return "没有找到当前会话里可以撤回的申请，本次没有改动任何申请。"


def _settle_withdrawals(plan: dict, user_id: str, session_id: str) -> tuple[dict, list[str]]:
    """Withdrawals never run through the case-creating workflow.

    With confirmation cards on, any withdrawal still here was not withdrawable (no card);
    without cards, the explicitly requested and evidenced withdrawal runs now.
    """
    withdrawals = [t for t in plan["tasks"] if t.get("intent") == "case_withdraw"]
    if not withdrawals:
        return plan, []
    notes = [
        f"这项{_withdrawn_case_name(t)}已经进入审核或已处理完成，无法撤回；可以查看进度或联系人工客服。"
        if _write_confirmation_enabled()
        else _perform_withdrawal(t, user_id, session_id)
        for t in withdrawals
    ]
    return {**plan, "tasks": [t for t in plan["tasks"] if t.get("intent") != "case_withdraw"]}, notes


def _write_confirmation_enabled() -> bool:
    """Writes wait for an explicit click on a confirmation card (opt-in while it is evaluated)."""
    return os.environ.get("CUSTOMER_WRITE_CONFIRMATION", "0").strip().lower() in {"1", "true", "yes", "on"}


def _confirmation_ttl_seconds() -> float:
    try:
        return max(30.0, float(os.environ.get("CUSTOMER_CONFIRMATION_TTL_SECONDS", "600")))
    except ValueError:
        return 600.0


def _customer_order_label(order: dict | None) -> str:
    """The order as the customer sees it: category, amount, status, delay."""
    if not order:
        return "这笔订单"
    parts = [
        f"R${order['amount']}" if order.get("amount") not in (None, "") else "",
        _customer_order_status_label(str(order.get("status") or "")),
    ]
    delay = order.get("delay_days")
    if isinstance(delay, int) and delay > 0:
        parts.append(f"晚到{delay}天")
    category = _customer_category_label(str(order.get("category") or ""))
    return f"{category}（{'，'.join(p for p in parts if p)}）"


def _order_options(plan: dict, orders: list[dict]) -> list[CustomerAction]:
    """Object choices as buttons: only the orders the model could not tell apart, when it named them."""
    listing = any(t.get("intent") == "customer_orders" for t in plan["tasks"])
    candidates: list[str] = []
    for task in plan["tasks"]:
        for order_id in task.get("candidate_order_ids") or []:
            if order_id not in candidates:
                candidates.append(order_id)
    shown = [o for o in orders if o.get("order_id") in candidates] if candidates and not listing else orders
    return [
        CustomerAction(
            id="track_order",
            label=f"{i + 1}. {_customer_order_label(o)} · {str(o.get('purchased_at') or '')[:10]}",
            order_id=o["order_id"],
            message="我说的是这笔订单，继续刚才的请求。",
        )
        for i, o in enumerate(shown)
    ]


def _confirmation_reply(cards: list[dict]) -> str:
    lines = []
    for card in cards:
        order = _customer_order_label(card.get("order"))
        if card["task"].get("action_type") == "withdraw_case":
            lines.append(f"请确认是否撤回{order}的{_withdrawn_case_name(card['task'])}。")
        else:
            lines.append(
                f"请确认是否为{order}提交"
                f"{_WRITE_ACTION_NAMES.get(card['task'].get('action_type'), '售后申请')}。"
            )
    if any(card["task"].get("action_type") != "withdraw_case" for card in cards):
        lines.append("点击确认后才会提交；系统会按售后规则处理，符合条件的直接办理，其余转人工审核。")
    return "\n".join(lines)


def _confirmation_actions(cards: list[dict]) -> list[CustomerAction]:
    actions = []
    for card in cards:
        task = card["task"]
        shared = {
            "order_id": task.get("order_id"),
            "confirmation_id": card["confirmation_id"],
            "action_type": task.get("action_type"),
        }
        if task.get("action_type") == "withdraw_case":
            confirm_label, decline_label = "确认撤回", "不撤回"
        else:
            name = _WRITE_ACTION_NAMES.get(task.get("action_type"), "售后申请")
            confirm_label, decline_label = f"确认提交{name}", "先不提交"
        actions.append(
            CustomerAction(
                id="confirm_application",
                label=confirm_label,
                style="primary",
                message=confirm_label,
                **shared,
            )
        )
        actions.append(
            CustomerAction(id="decline_application", label=decline_label, message=decline_label, **shared)
        )
        if task.get("action_type") != "withdraw_case":
            # "Wrong order" is not "don't do it": keep the action, let the customer pick the object.
            # No order_id, so the page does not switch to the rejected order.
            actions.append(
                CustomerAction(
                    id="reselect_application",
                    label="不是这笔",
                    message="不是这笔订单",
                    confirmation_id=card["confirmation_id"],
                    action_type=task.get("action_type"),
                )
            )
    return actions


async def _offer_other_orders(
    graph, task: dict, orders: list[dict], user_id: str, session_id: str
) -> list[dict]:
    """The card named the wrong order: offer the same action on the orders the customer may have meant.

    Same-category orders first (the usual confusion), otherwise any owned order where the action would
    be submitted. Each offer is a new single-use card; nothing runs until one is confirmed.
    """
    rejected = task.get("order_id")
    category = next((o.get("category") for o in orders if o.get("order_id") == rejected), None)

    def offers_from(pool: list[dict]) -> list[dict]:
        swapped = [{**task, "order_id": o["order_id"]} for o in pool if o.get("order_id") != rejected]
        return [t for t in swapped if _would_submit(t)]

    offers = offers_from([o for o in orders if o.get("category") == category]) or offers_from(orders)
    if not offers:
        return []
    # Picking the object is the point here, so every offer is a card, whatever its route.
    _, cards = await _defer_writes_for_confirmation(
        graph, {"tasks": offers[:4]}, orders, user_id, session_id, tiered=False
    )
    return cards


def _reselect_reply(task: dict, cards: list[dict]) -> str:
    name = _WRITE_ACTION_NAMES.get(str(task.get("action_type") or ""), "售后申请")
    if not cards:
        return f"没有找到其他可以提交{name}的订单，本次没有提交任何申请。"
    return f"好的，换一笔订单。\n{_confirmation_reply(cards)}"


def _confirmation_scope() -> str:
    """Which writes wait for a click: "auto_only" (default) or "all" (every write)."""
    scope = os.environ.get("CUSTOMER_CONFIRMATION_SCOPE", "auto_only").strip().lower()
    return scope if scope in {"auto_only", "all"} else "auto_only"


def _goes_to_review(task: dict) -> bool:
    """True only when the workflow is certain to hand this write to a reviewer.

    Same decision engine and policy guard as the write path. The flags this check cannot know
    yet (a repeat claim, an earlier refund failure) only ever add review, so "review" here is
    review there. Anything else (auto, an unknown order, an error) keeps the confirmation card:
    the click stays on every write that would take effect without a person looking at it.
    """
    if task.get("intent") == "case_withdraw":
        return False
    order_id = str(task.get("order_id") or "")
    order = olist_service.get_order_status(order_id)
    if order is None:
        return False
    action_type = str(task.get("action_type") or "")
    try:
        decision = (
            AfterSalesDecisionEngine()
            .assess(
                action_type=action_type,
                order=order,
                user_request=str(task.get("text") or ""),
                policy_sections=[],
            )
            .decision
        )
        task_risk, decision_risk = str(task.get("risk_level") or ""), str(decision.risk_level or "")
        risk = "high" if "high" in {task_risk, decision_risk} else (task_risk or decision_risk)
        policy = PolicyGuard(_policy_engine, olist_service, case_service).check(
            action_type, order_id, risk_level=risk
        )
    except Exception:
        return False
    return policy.get("decision") == "review"


def _would_submit(task: dict) -> bool:
    """Read-only pre-check with the same decision engine the write path uses.

    Only a write the workflow would submit (directly or for review) is worth a card.
    Rejections and missing information depend on order facts alone, so they are left to
    the normal path, which explains the rejection or asks for what is missing.
    """
    if task.get("intent") == "case_withdraw":
        return _withdrawable(task)
    order = olist_service.get_order_status(str(task.get("order_id") or ""))
    if order is None:
        return False
    outcome = (
        AfterSalesDecisionEngine()
        .assess(
            action_type=str(task.get("action_type") or ""),
            order=order,
            user_request=str(task.get("text") or ""),
            policy_sections=[],
        )
        .decision.outcome
    )
    return outcome not in {"reject", "ask_clarification"}


async def _defer_writes_for_confirmation(
    graph, plan: dict, orders: list[dict], user_id: str, session_id: str, *, tiered: bool = True
) -> tuple[dict, list[dict]]:
    """Hold this turn's writes until the customer confirms them; read-only tasks still run now.

    The click, not the model's reading of the wording, is what authorizes a write that takes
    effect on its own. With the default "auto_only" scope a write that is certain to go to a
    reviewer is submitted now: a person checks it anyway, and the customer can still withdraw it.
    """
    skip_review = tiered and _confirmation_scope() == "auto_only"
    writes = [
        t
        for t in plan["tasks"]
        if t.get("side_effect") and _would_submit(t) and not (skip_review and _goes_to_review(t))
    ]
    if not writes:
        return plan, []
    config = {"configurable": {"thread_id": session_id}}
    now = time.time()
    snapshot = await graph.aget_state(config)
    pending = {
        token: record
        for token, record in dict((snapshot.values or {}).get("pending_confirmations") or {}).items()
        if float(record.get("expires_at", 0)) > now
    }
    by_id = {o.get("order_id"): o for o in orders}
    cards = []
    for task in writes:
        token = secrets.token_hex(32)
        pending[token] = {
            "user_id": user_id,
            "session_id": session_id,
            "created_at": now,
            "expires_at": now + _confirmation_ttl_seconds(),
            "task": dict(task),
        }
        cards.append({"confirmation_id": token, "task": dict(task), "order": by_id.get(task.get("order_id"))})
    await graph.aupdate_state(
        config,
        {
            "pending_confirmations": pending,
            "user_id": user_id,
            "tenant_id": "olist-demo",
            "session_id": session_id,
        },
        as_node="finalize_answer",
    )
    held = {id(t) for t in writes}
    return {**plan, "tasks": [t for t in plan["tasks"] if id(t) not in held]}, cards


async def _consume_customer_confirmation(graph, request: ChatRequest, user_id: str, session_id: str) -> dict:
    """Single use: the token is removed before anything executes. Only the stored task can run."""
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    now = time.time()
    pending = dict((snapshot.values or {}).get("pending_confirmations") or {})
    record = pending.pop(request.confirmation_id or "", None)
    if snapshot.values:
        await graph.aupdate_state(
            config,
            {
                "pending_confirmations": {
                    t: r for t, r in pending.items() if float(r.get("expires_at", 0)) > now
                }
            },
            as_node="finalize_answer",
        )
    if (
        not record
        or record.get("user_id") != user_id
        or record.get("session_id") != session_id
        or float(record.get("expires_at", 0)) <= now
    ):
        return {
            "status": "invalid",
            "answer": "这个确认已失效或不属于当前会话，本次没有提交任何申请。如需办理，请重新说明。",
        }
    if request.requested_action == "decline_application":
        return {"status": "declined", "answer": "好的，这项申请先不提交。需要时可以随时再告诉我。"}
    if request.requested_action == "reselect_application":
        return {"status": "reselect", "task": dict(record["task"])}
    task = {**record["task"], "authorization": "explicit", "authorization_source": "confirmed_card"}
    return {
        "status": "confirmed",
        "plan": {
            "version": 1,
            "origin": "model",
            "user_id": user_id,
            "session_id": session_id,
            "message_sha256": hashlib.sha256(request.message.encode()).hexdigest(),
            "tasks": [task],
            "decisions": [{"task_id": task.get("task_id"), "validation": "confirmed_card"}],
            "selection_only": False,
        },
    }


@app.post("/customer/chat", response_model=ChatResponse)
async def customer_chat(
    request: ChatRequest,
    x_demo_customer: str | None = Header(default=None),
) -> ChatResponse:
    """Customer-facing self-service entrypoint with minimal read/draft scopes."""
    customer_user_id = _resolve_customer_actor(x_demo_customer)
    # Rate limit at the product boundary, before clarification/tool early
    # returns, and key it on the server-resolved actor rather than body fields.
    rate_limit = await _check_rate_limit(request, user_id=customer_user_id, tenant_id="olist-demo")
    if rate_limit is not None and not rate_limit.allowed:
        return _rate_limited_chat_response(
            request,
            rate_limit,
            session_id=request.session_id or str(uuid.uuid4()),
        )
    ownership_denial = _customer_order_access_denied(request, customer_user_id=customer_user_id)
    if ownership_denial is not None:
        return ownership_denial
    pii_denial = _customer_pii_request_denied(request, customer_user_id=customer_user_id)
    if pii_denial is not None:
        return pii_denial
    page_context = _validated_customer_page_context(request, customer_user_id)
    session_id = request.session_id or str(uuid.uuid4())
    graph = conversation_agent or agent
    decoupled = bool(
        case_workflow_service.configured and conversation_agent is not None and graph is conversation_agent
    )
    owner = await asyncio.to_thread(conversations.conversation_owner, session_id)
    if owner is not None and str(owner.get("user_id")) != customer_user_id:
        raise HTTPException(status_code=403, detail="session_owner_required")
    previous = await graph.aget_state({"configurable": {"thread_id": session_id}})
    if previous.values and previous.values.get("user_id") != customer_user_id:
        raise HTTPException(status_code=403, detail="session_owner_required")
    await asyncio.to_thread(conversations.ensure_conversation, session_id, "olist-demo", customer_user_id)
    await asyncio.to_thread(conversations.append_message, session_id, "customer", request.message)
    contract_mode = os.environ.get("CUSTOMER_PLANNING_MODE", "legacy") == "contract"
    server_turn_plan = None
    contract_orders = []
    confirmation_cards: list[dict] = []
    withdrawal_notes: list[str] = []
    confirming = request.requested_action in CONFIRMATION_ACTIONS
    if confirming and not contract_mode:
        raise HTTPException(status_code=422, detail="unsupported_customer_action")
    if contract_mode:
        if not decoupled:
            raise HTTPException(status_code=503, detail="contract_requires_decoupled_case_workflow")
        if await asyncio.to_thread(conversations.human_mode, session_id):
            return ChatResponse(answer="您的消息已发送给客服，请稍候。", session_id=session_id)
        owned_context = await asyncio.to_thread(_customer_context_payload, customer_user_id)
        contract_orders = list(owned_context["orders"])
        if confirming:
            # A click on a confirmation card: no model call, and only the stored task may run.
            outcome = await _consume_customer_confirmation(graph, request, customer_user_id, session_id)
            if outcome["status"] == "reselect":
                cards = await _offer_other_orders(
                    graph, outcome["task"], contract_orders, customer_user_id, session_id
                )
                answer = _reselect_reply(outcome["task"], cards)
                await asyncio.to_thread(conversations.append_message, session_id, "assistant", answer)
                return ChatResponse(
                    answer=answer,
                    session_id=session_id,
                    ui_actions=_confirmation_actions(cards) or _customer_follow_up_actions(None),
                )
            if (
                outcome["status"] == "confirmed"
                and outcome["plan"]["tasks"][0].get("intent") == "case_withdraw"
            ):
                outcome = {
                    "status": "withdrawal",
                    "answer": _perform_withdrawal(outcome["plan"]["tasks"][0], customer_user_id, session_id),
                }
            if outcome["status"] != "confirmed":
                await asyncio.to_thread(
                    conversations.append_message, session_id, "assistant", outcome["answer"]
                )
                return ChatResponse(
                    answer=outcome["answer"],
                    session_id=session_id,
                    ui_actions=_customer_follow_up_actions(None),
                )
            server_turn_plan = outcome["plan"]
        else:
            history = await asyncio.to_thread(conversations.messages, session_id)
            owned_cases = [
                c
                for c in await asyncio.to_thread(case_service.list_by_session, session_id)
                if c.get("user_id") == customer_user_id
            ]
            server_turn_plan = await CustomerTurnPlanner(llm_client.chat_openai).plan(
                message=request.message,
                orders=contract_orders,
                cases=owned_cases,
                allowed=_allowed_customer_order_ids(customer_user_id),
                user_id=customer_user_id,
                session_id=session_id,
                history=history[:-1],
                pending=(previous.values or {}).get("validated_turn_plan"),
                page_context={**page_context.model_dump(), "requested_action": request.requested_action},
            )
            if _write_confirmation_enabled():
                server_turn_plan, confirmation_cards = await _defer_writes_for_confirmation(
                    graph, server_turn_plan, contract_orders, customer_user_id, session_id
                )
            server_turn_plan, withdrawal_notes = _settle_withdrawals(
                server_turn_plan, customer_user_id, session_id
            )
        await graph.aupdate_state(
            {"configurable": {"thread_id": session_id}},
            {
                "customer_order_context": {
                    "orders": contract_orders,
                    "allowed": sorted(_allowed_customer_order_ids(customer_user_id)),
                },
                "user_id": customer_user_id,
                "tenant_id": "olist-demo",
                "session_id": session_id,
            },
            as_node="finalize_answer",
        )
        if (confirmation_cards or withdrawal_notes) and not server_turn_plan["tasks"]:
            answer = "\n".join(
                [
                    *withdrawal_notes,
                    *([_confirmation_reply(confirmation_cards)] if confirmation_cards else []),
                ]
            )
            await asyncio.to_thread(conversations.append_message, session_id, "assistant", answer)
            return ChatResponse(
                answer=answer,
                session_id=session_id,
                ui_actions=_confirmation_actions(confirmation_cards) or _customer_follow_up_actions(None),
            )
    discovery_context = dict(previous.values.get("customer_order_context") or {})
    semantic_mode = (
        os.environ.get("ORDER_GROUNDING_MODE", "semantic") == "semantic" and customer_order_agent.available
    )
    discovery_applicable = (
        semantic_order_grounding.applicable(request.message, discovery_context)
        if semantic_mode
        else order_discovery.applicable(request.message, discovery_context)
    )
    if (
        not contract_mode
        and decoupled
        and not request.requested_action
        and (
            semantic_mode
            or not page_context.selected_order_id
            or any(cue in request.message for cue in ("电脑", "母婴", "家居", "健康与美容"))
        )
        and discovery_applicable
        and not await asyncio.to_thread(conversations.human_mode, session_id)
    ):
        if semantic_mode:
            history = await asyncio.to_thread(conversations.messages, session_id)
            records = await asyncio.to_thread(case_service.list_by_session, session_id)
            grounded = await semantic_order_grounding.ground(
                customer_order_agent,
                request.message,
                discovery_context,
                _allowed_customer_order_ids(customer_user_id),
                history[:-1],
                selected_order_id=page_context.selected_order_id,
                cases=[record for record in records if record.get("user_id") == customer_user_id],
            )
        else:
            grounded = await asyncio.to_thread(
                order_discovery.ground,
                customer_order_agent,
                request.message,
                discovery_context,
                _allowed_customer_order_ids(customer_user_id),
            )
        await graph.aupdate_state(
            {"configurable": {"thread_id": session_id}},
            {
                "customer_order_context": grounded["context"],
                "user_id": customer_user_id,
                "tenant_id": "olist-demo",
                "session_id": session_id,
            },
            as_node="finalize_answer",
        )
        if grounded.get("answer"):
            answer = str(grounded["answer"])
            await asyncio.to_thread(conversations.append_message, session_id, "assistant", answer)
            return ChatResponse(
                answer=answer,
                session_id=session_id,
                ui_actions=[CustomerAction(**action) for action in grounded.get("actions", [])],
            )
        if not grounded.get("passthrough"):
            page_context = page_context.model_copy(update={"selected_order_id": grounded["order_id"]})
            request = request.model_copy(
                update={
                    "message": grounded["message"],
                    "requested_action": grounded.get("action")
                    or ("track_order" if grounded.get("read_only") else None),
                }
            )
    if not contract_mode and _should_use_customer_order_agent(request.message, page_context):
        resolution = await customer_order_agent.resolve(
            request.message,
            allowed_order_ids=_allowed_customer_order_ids(customer_user_id),
        )
        if resolution and resolution.get("answer"):
            actions = _customer_agent_actions(request.message, list(resolution.get("candidates") or []))
            answer = str(resolution["answer"])
            await asyncio.to_thread(conversations.append_message, session_id, "assistant", answer)
            return ChatResponse(
                answer=answer,
                session_id=session_id,
                ui_actions=actions,
                case_id=None,
            )
    clarification = (
        None
        if contract_mode
        else _customer_order_clarification(customer_user_id, request.message, page_context)
    )
    if clarification is not None:
        answer, actions = clarification
        await asyncio.to_thread(conversations.append_message, session_id, "assistant", answer)
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            ui_actions=actions,
            case_id=None,
        )
    if await asyncio.to_thread(conversations.human_mode, session_id):
        return ChatResponse(answer="您的消息已发送给客服，请稍候。", session_id=session_id)
    customer_request = request.model_copy(
        update={
            "session_id": session_id,
            "role": "customer",
            "channel": "customer_self_service",
            "auth_scopes": _resolve_auth_scopes("customer", request.auth_scopes),
            "user_id": customer_user_id,
            "tenant_id": "olist-demo",
            "page_context": page_context,
            # A confirmed card already carries its server-built plan; it is not a page action.
            "requested_action": None if confirming else _validated_customer_action(request.requested_action),
        }
    )
    t_total = time.perf_counter()
    plan_token = ACTIVE_TURN.set(server_turn_plan)
    try:
        response = await chat(
            customer_request,
            graph=graph,
            decouple_cases=decoupled,
            rate_limit_checked=True,
        )
    finally:
        ACTIVE_TURN.reset(plan_token)
    if response.request_analysis.get("route_intent") == "input_guard":
        # Nothing ran this turn: presenting the conversation state would repeat an older answer.
        answer = "我可以协助查询本人订单、了解售后政策或提交售后申请。本次没有办理任何事项。"
        await asyncio.to_thread(conversations.append_message, session_id, "assistant", answer)
        return ChatResponse(
            answer=answer, session_id=session_id, ui_actions=_customer_follow_up_actions(None)
        )
    t_chat = time.perf_counter()
    snapshot = await graph.aget_state({"configurable": {"thread_id": response.session_id}})
    state = {**dict(snapshot.values or {}), "final_answer": response.answer}
    case_id = str(response.case_id or state.get("case_id") or "")
    record = (
        case_service.get(case_id)
        if case_id
        else (None if decoupled else case_service.get_by_session(response.session_id))
    )
    if record is not None and not case_id:
        case_id = str(record.get("case_id") or "")
    if record is None and state.get("pending_side_effect", {}).get("requires_confirmation"):
        record = {
            "status": "pending_review",
            "action_type": state["pending_side_effect"].get("type", "open_support_case"),
        }
    request_analysis = dict(state.get("request_analysis") or {})
    request_id = str(request_analysis.get("request_id") or "")
    try:
        session_records = await asyncio.to_thread(case_service.list_by_session, response.session_id)
    except Exception:
        session_records = []
    turn_records = [
        item for item in session_records if not request_id or _case_request_id(item) == request_id
    ]
    if not turn_records:
        turn_records = session_records
    state["customer_case_records"] = [
        item
        for item in session_records
        if request_id and _case_request_id(item) == request_id and item.get("user_id") == customer_user_id
    ]
    t_present = time.perf_counter()
    response.answer = await present(llm_client.chat_openai, request.message, state, record)
    t_done = time.perf_counter()
    response.sources = []
    response.case_id = case_id or None
    safe_cases = [_public_case(item) for item in turn_records]
    progress = _progress_summary(safe_cases)
    response.sub_cases = safe_cases
    response.request_analysis = {
        **_safe_request_analysis(request_analysis),
        **progress,
        "partial_auto": bool(progress["processed"] and progress["pending_review"]),
    }
    response.timings = {
        "chat_ms": round((t_chat - t_total) * 1000, 2),
        "present_ms": round((t_done - t_present) * 1000, 2),
        "total_ms": round((t_done - t_total) * 1000, 2),
    }
    logger.info(
        "CUSTOMER_TIMING | session=%s | chat_ms=%.2f | present_ms=%.2f | total_ms=%.2f",
        session_id,
        response.timings["chat_ms"],
        response.timings["present_ms"],
        response.timings["total_ms"],
    )
    response.ui_actions = _customer_follow_up_actions(page_context.selected_order_id)
    if server_turn_plan and any(
        t["intent"] in {"customer_orders", "clarify"} for t in server_turn_plan["tasks"]
    ):
        response.ui_actions = _order_options(server_turn_plan, contract_orders)
    if withdrawal_notes:
        response.answer = "\n".join([*withdrawal_notes, response.answer])
    if confirmation_cards:
        response.answer = f"{response.answer}\n\n{_confirmation_reply(confirmation_cards)}"
        response.ui_actions = [*_confirmation_actions(confirmation_cards), *response.ui_actions]
    await asyncio.to_thread(
        conversations.append_message, session_id, "assistant", response.answer, case_id=case_id or None
    )
    return response


class StaffMessage(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


@app.get("/review/conversations")
async def review_conversations(x_review_token: str | None = Header(default=None)) -> dict:
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    rows = await asyncio.to_thread(conversations.recent_sessions)
    for row in rows:
        record = await asyncio.to_thread(case_service.get_by_session, row["session_id"])
        row.update(
            {
                "order_id": record.get("order_id", "") if record else "",
                "action_type": record.get("action_type", "咨询会话") if record else "咨询会话",
                "created_at": row["updated_at"],
            }
        )
    return {"cases": rows}


@app.get("/review/metrics")
def review_metrics(x_review_token: str | None = Header(default=None)) -> dict:
    """Authenticated evaluation baselines and clearly scoped runtime traces."""
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    return load_evaluation_snapshot(runtime_summary=trace_summary(), runtime_cases=case_metrics())


@app.get("/shadow/metrics")
def shadow_metrics(x_review_token: str | None = Header(default=None)) -> dict:
    """Staff-only shadow comparison: suggestions vs human decisions."""
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    return shadow_service.metrics()


@app.get("/shadow/records")
def shadow_records(
    limit: int = Query(default=50, ge=1, le=200),
    risk_level: str = Query(default=""),
    action_type: str = Query(default=""),
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> dict:
    """Recent shadow observations, with order ids masked for staff review."""
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    return {
        "records": shadow_service.recent(
            limit=limit,
            risk_level=risk_level,
            action_type=action_type,
        )
    }


class HandoffMode(BaseModel):
    human: bool


async def _customer_conversation_owner(session_id: str, customer_user_id: str):
    owner = await asyncio.to_thread(conversations.conversation_owner, session_id)
    if owner is not None:
        if str(owner.get("user_id")) != customer_user_id:
            raise HTTPException(status_code=403, detail="conversation_owner_required")
        return owner
    graph = conversation_agent or agent
    snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    if snapshot.values.get("user_id") != customer_user_id:
        raise HTTPException(status_code=403, detail="conversation_owner_required")
    return snapshot


async def _existing_conversation(session_id: str):
    graph = conversation_agent or agent
    snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
    if not snapshot.values and conversation_agent is not None:
        snapshot = await agent.aget_state({"configurable": {"thread_id": session_id}})
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return snapshot


@app.get("/customer/sessions/{session_id}/messages")
async def customer_messages(
    session_id: str,
    x_demo_customer: str | None = Header(default=None),
) -> dict:
    await _customer_conversation_owner(session_id, _resolve_customer_actor(x_demo_customer))
    return {
        "messages": await asyncio.to_thread(conversations.messages, session_id),
        "human": await asyncio.to_thread(conversations.human_mode, session_id),
    }


@app.get("/review/sessions/{session_id}/messages")
async def staff_messages(session_id: str, x_review_token: str | None = Header(default=None)) -> dict:
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    await _existing_conversation(session_id)
    return {
        "messages": await asyncio.to_thread(conversations.messages, session_id),
        "human": await asyncio.to_thread(conversations.human_mode, session_id),
    }


@app.post("/review/sessions/{session_id}/messages")
async def staff_reply(
    session_id: str, body: StaffMessage, x_review_token: str | None = Header(default=None)
) -> dict:
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    await _existing_conversation(session_id)
    await asyncio.to_thread(conversations.set_human_mode, session_id, True)
    await asyncio.to_thread(conversations.append_message, session_id, "staff", body.content)
    record_trace(
        session_id=session_id,
        user_message="[staff_reply]",
        result={"route_intent": "staff_handoff", "final_answer": "staff_reply_sent"},
        latency_ms=0,
        status="staff_reply_sent",
    )
    return {"sent": True}


@app.post("/review/sessions/{session_id}/handoff")
async def staff_handoff(
    session_id: str, body: HandoffMode, x_review_token: str | None = Header(default=None)
) -> dict:
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    await _existing_conversation(session_id)
    await asyncio.to_thread(conversations.set_human_mode, session_id, body.human)
    record_trace(
        session_id=session_id,
        user_message="[staff_handoff]",
        result={"route_intent": "staff_handoff", "final_answer": "human" if body.human else "assistant"},
        latency_ms=0,
        status="staff_handoff_enabled" if body.human else "staff_handoff_released",
    )
    return {"human": body.human}


@app.post("/review/sessions/{session_id}/suggestion")
async def staff_suggestion(session_id: str, x_review_token: str | None = Header(default=None)) -> dict:
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    await _existing_conversation(session_id)
    history = await asyncio.to_thread(conversations.messages, session_id)
    snapshot = await agent.aget_state({"configurable": {"thread_id": session_id}})
    latest = next((m["content"] for m in reversed(history) if m["sender"] == "customer"), "")
    draft = await present(
        llm_client.chat_openai, latest, dict(snapshot.values), case_service.get_by_session(session_id)
    )
    signals = [word for word in ("投诉", "生气", "失望", "太慢", "着急", "等了", "不满意") if word in latest]
    return {
        "draft": draft,
        "signals": signals,
        "attention": "建议先回应客户不满或催促，再说明下一步" if signals else "未发现明显不满或催促用词",
        "signal_method": "本轮文本关键词提示，不代表心理状态判断",
    }


@app.get("/customer/cases/{case_id}", response_model=CustomerCaseResponse)
async def get_customer_case(
    case_id: str,
    x_demo_customer: str | None = Header(default=None),
) -> CustomerCaseResponse:
    """Return a customer-safe status view for an after-sales case."""
    customer_user_id = _resolve_customer_actor(x_demo_customer)
    record = case_service.get(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="case_not_found")
    if str(record.get("user_id") or "demo-customer") != customer_user_id:
        raise HTTPException(status_code=403, detail="case_owner_required")
    return _customer_case_response(record)


@app.post("/customer/cases/{case_id}/appeal", response_model=CustomerCaseResponse)
async def appeal_customer_case(
    case_id: str,
    request: CustomerAppealRequest,
    x_demo_customer: str | None = Header(default=None),
) -> CustomerCaseResponse:
    """Reopen a rejected or timed-out case for staff review with new customer evidence."""
    customer_user_id = _resolve_customer_actor(x_demo_customer)
    try:
        record = case_service.appeal_case(case_id, customer_user_id, request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="case_not_found_or_not_owned")
    if record.get("status") == "appealed_pending_review":
        try:
            record = await case_workflow_service.ensure_workflow(case_id)
        except Exception:
            logger.exception("Appeal initialization queued for recovery: %s", case_id)
    record_trace(
        session_id=str(record.get("session_id") or case_id),
        user_message=f"[customer_appeal:{case_id}] {request.reason}",
        result={
            "route_intent": "appeal",
            "final_answer": "customer appeal submitted",
            "trajectory_events": [
                {
                    "node": "customer_case_appeal",
                    "intent": "appeal",
                    "status": str(record.get("status")),
                    "details": {
                        "case_id": case_id,
                        "appeal_count": int(record.get("appeal_count") or 0),
                    },
                }
            ],
        },
        latency_ms=0,
        status="customer_case_appeal",
    )
    return _customer_case_response(record)


@app.get("/review/cases", response_model=ReviewCaseListResponse)
async def list_review_cases(
    limit: int = Query(default=50, ge=1, le=200),
    role: str = Query(default="after_sales_operator"),
    auth_scopes: list[str] | None = Query(default=None),
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> ReviewCaseListResponse:
    """Return pending and appealed after-sales cases for asynchronous staff handling."""
    _authorize_review(role, auth_scopes, x_review_token)
    now = time.time()
    cases: list[dict] = []
    for raw in case_service.list_review_queue(limit):
        record = dict(raw)
        record["age_seconds"] = _iso_age_seconds(record.get("created_at"), now)
        stuck_since = record.get("reviewed_at") or record.get("updated_at")
        record["stuck_seconds"] = (
            _iso_age_seconds(stuck_since, now) if str(record.get("status")) == "approving" else 0
        )
        record["needs_attention"] = (
            str(record.get("status")) == "approving" and int(record["stuck_seconds"]) >= 120
        )
        record.pop("case_payload", None)
        cases.append(record)
    return ReviewCaseListResponse(cases=cases)


@app.get("/review/sessions/{session_id}", response_model=ReviewSessionResponse)
async def get_review_session(
    session_id: str,
    role: str = Query(default="after_sales_operator"),
    auth_scopes: list[str] | None = Query(default=None),
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> ReviewSessionResponse:
    """Return the staff-facing HITL review packet for a paused case."""
    _authorize_review(role, auth_scopes, x_review_token)
    case = _single_actionable_case(session_id)
    if case is not None and case_workflow_service.configured and case.get("workflow_thread_id"):
        packet = case_workflow_service.review_packet(str(case["case_id"])) or {}
        return ReviewSessionResponse(
            session_id=session_id,
            has_pending=bool(packet.get("has_pending")),
            pending_side_effect=dict(packet.get("pending_side_effect") or {}),
            escalation_draft=dict(packet.get("escalation_draft") or {}),
            after_sales_cases=list(packet.get("after_sales_cases") or []),
            completed_tasks=list(packet.get("completed_tasks") or []),
            trajectory_events=[],
            customer_safe_summary=f"待审核 Case：{case.get('case_id')}",
        )
    snapshot = await _existing_conversation(session_id)
    values = dict(snapshot.values or {})
    return _review_session_response(session_id, bool(snapshot.next), values)


@app.post("/review/sessions/{session_id}/approve", response_model=ChatResponse)
async def approve_review_session(
    session_id: str,
    request: ReviewActionRequest | None = None,
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> ChatResponse:
    """Approve a paused side-effect case as an after-sales operator."""
    role = request.role if request else "after_sales_operator"
    auth_scopes = request.auth_scopes if request else None
    _authorize_review(role, auth_scopes, x_review_token)
    case = _single_actionable_case(session_id)
    if case is not None and case_workflow_service.configured and case.get("workflow_thread_id"):
        return await _resume_case_review(
            str(case["case_id"]),
            decision="approve",
            reviewer_id=(request.reviewer_id if request else "demo-reviewer"),
            reason=(request.reason if request else ""),
        )
    return await _resume_review_session(
        session_id,
        resume_value="__review_approve__",
        status="review_approved",
        reviewer_id=(request.reviewer_id if request else "demo-reviewer"),
    )


@app.post("/review/sessions/{session_id}/reject", response_model=ChatResponse)
async def reject_review_session(
    session_id: str,
    request: ReviewActionRequest | None = None,
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> ChatResponse:
    """Reject a paused side-effect case and clear the HITL checkpoint."""
    role = request.role if request else "after_sales_operator"
    auth_scopes = request.auth_scopes if request else None
    _authorize_review(role, auth_scopes, x_review_token)
    case = _single_actionable_case(session_id)
    if case is not None and case_workflow_service.configured and case.get("workflow_thread_id"):
        return await _resume_case_review(
            str(case["case_id"]),
            decision="reject",
            reviewer_id=(request.reviewer_id if request else "demo-reviewer"),
            reason=(request.reason if request else ""),
        )
    return await _resume_review_session(
        session_id,
        resume_value="no",
        status="review_rejected",
        reviewer_id=(request.reviewer_id if request else "demo-reviewer"),
    )


@app.get("/review/sessions/{session_id}/cases")
async def list_session_cases(
    session_id: str,
    role: str = Query(default="after_sales_operator"),
    auth_scopes: list[str] | None = Query(default=None),
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> dict:
    _authorize_review(role, auth_scopes, x_review_token)
    records = case_service.list_by_session(session_id)
    return {"cases": [_case_queue_summary(record) for record in records]}


@app.get("/review/cases/{case_id}")
async def get_review_case(
    case_id: str,
    role: str = Query(default="after_sales_operator"),
    auth_scopes: list[str] | None = Query(default=None),
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> dict:
    _authorize_review(role, auth_scopes, x_review_token)
    packet = case_workflow_service.review_packet(case_id)
    if packet is None:
        raise HTTPException(status_code=404, detail="case_not_found")
    packet["order_facts"] = _case_order_facts(packet.get("order_id"))
    packet["needs_attention"] = _case_needs_attention(packet)
    return packet


@app.post("/review/cases/{case_id}/approve", response_model=ChatResponse)
async def approve_review_case(
    case_id: str,
    request: ReviewActionRequest | None = None,
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> ChatResponse:
    role = request.role if request else "after_sales_operator"
    auth_scopes = request.auth_scopes if request else None
    _authorize_review(role, auth_scopes, x_review_token)
    return await _resume_case_review(
        case_id,
        decision="approve",
        reviewer_id=(request.reviewer_id if request else "demo-reviewer"),
        reason=(request.reason if request else ""),
        expected_attempt=(request.expected_attempt if request else None),
    )


@app.post("/review/cases/{case_id}/reject", response_model=ChatResponse)
async def reject_review_case(
    case_id: str,
    request: ReviewActionRequest | None = None,
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> ChatResponse:
    role = request.role if request else "after_sales_operator"
    auth_scopes = request.auth_scopes if request else None
    _authorize_review(role, auth_scopes, x_review_token)
    return await _resume_case_review(
        case_id,
        decision="reject",
        reviewer_id=(request.reviewer_id if request else "demo-reviewer"),
        reason=(request.reason if request else ""),
        expected_attempt=(request.expected_attempt if request else None),
    )


def _single_actionable_case(session_id: str) -> dict | None:
    cases = case_service.list_by_session(
        session_id,
        statuses=("pending_review", "appealed_pending_review", "approving"),
    )
    if len(cases) > 1:
        raise HTTPException(
            status_code=409,
            detail="multiple_pending_cases_require_case_id",
        )
    return cases[0] if cases else None


async def _resume_case_review(
    case_id: str,
    *,
    decision: str,
    reviewer_id: str,
    reason: str = "",
    expected_attempt: int | None = None,
) -> ChatResponse:
    current = case_service.get(case_id)
    if current and int(current.get("workflow_attempt") or 1) > 1 and expected_attempt is None:
        raise HTTPException(status_code=409, detail="workflow_attempt_required")
    outcome = await case_workflow_service.resume_case(
        case_id, decision, reviewer_id, expected_attempt=expected_attempt
    )
    if not outcome.get("ok"):
        if outcome.get("error") == "not_found":
            raise HTTPException(status_code=404, detail="case_not_found")
        raise HTTPException(status_code=409, detail="case_decision_conflict")
    try:
        shadow_service.record_human_decision(
            case_id=case_id,
            decision=decision,
            reviewer_id=reviewer_id,
            reason=reason,
        )
    except Exception:
        pass
    record = dict(outcome.get("record") or {})
    result = dict(outcome.get("result") or {})
    if outcome.get("already") and not result:
        return ChatResponse(
            answer=f"该 Case 已处理（{record.get('status')}）。",
            session_id=str(record.get("session_id") or ""),
            case_id=case_id,
            sources=[],
        )
    answer = str(result.get("final_answer") or "Case 审核完成。")
    # The workflow service owns durable, deduplicated customer notifications.
    return ChatResponse(
        answer=answer,
        session_id=str(record.get("session_id") or ""),
        case_id=case_id,
        sources=_response_sources(result),
    )


def _iso_age_seconds(raw: object, now: float) -> int:
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return max(0, int(now - parsed.timestamp()))
    except (TypeError, ValueError):
        return 0


def _case_needs_attention(record: dict) -> bool:
    if str(record.get("status")) != "approving":
        return False
    since = record.get("reviewed_at") or record.get("updated_at")
    return _iso_age_seconds(since, time.time()) >= 120


def _case_queue_summary(record: dict) -> dict:
    now = time.time()
    summary = {
        "case_id": record.get("case_id"),
        "session_id": record.get("session_id"),
        "status": record.get("status"),
        "action_type": record.get("action_type"),
        "order_id": record.get("order_id"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "reviewed_at": record.get("reviewed_at"),
        "expires_at": record.get("expires_at"),
        "workflow_thread_id": record.get("workflow_thread_id"),
        "workflow_attempt": record.get("workflow_attempt"),
        "has_pending": record.get("status") in {"pending_review", "appealed_pending_review"},
        "age_seconds": _iso_age_seconds(record.get("created_at"), now),
        "stuck_seconds": (
            _iso_age_seconds(record.get("reviewed_at") or record.get("updated_at"), now)
            if str(record.get("status")) == "approving"
            else 0
        ),
    }
    summary["needs_attention"] = (
        str(record.get("status")) == "approving" and int(summary["stuck_seconds"]) >= 120
    )
    return summary


def _case_order_facts(order_id: object) -> dict:
    order = olist_service.get_order_status(str(order_id or ""))
    if order is None:
        return {}
    return {
        "order_id": order.order_id,
        "order_status": order.status,
        "payment_value": order.payment_value,
        "delay_days": order.delay_days,
        "review_score": order.review_score,
    }


def _response_sources(result: dict) -> list[str]:
    artifacts = result.get("artifacts", {})
    if isinstance(artifacts, dict):
        sources = artifacts.get("insight_sources", []) or artifacts.get("policy_sources", [])
        return [str(source) for source in sources]
    insights = result.get("retrieved_insights", [])
    if insights and isinstance(insights[0], dict):
        return [str(item.get("name", "")) for item in insights]
    policies = result.get("retrieved_policy", [])
    if policies and isinstance(policies[0], dict):
        return [str(item.get("section_title", "")) for item in policies]
    return [str(source) for source in (insights or policies)]


def _customer_case_response(record: dict) -> CustomerCaseResponse:
    order_id = str(record.get("order_id", ""))
    payload = _case_payload(record)
    sub_task = _case_sub_task(record)
    policy = payload.get("policy_decision")
    if not isinstance(policy, dict):
        draft = payload.get("draft")
        policy = draft.get("policy_decision") if isinstance(draft, dict) else None
    policy_decision = str(policy.get("decision") or "") if isinstance(policy, dict) else ""
    if not policy_decision:
        decision = payload.get("decision")
        if not isinstance(decision, dict):
            draft = payload.get("draft")
            decision = draft.get("decision") if isinstance(draft, dict) else None
        policy_decision = str(decision.get("outcome") or "") if isinstance(decision, dict) else ""
    status = str(record.get("status", ""))
    return CustomerCaseResponse(
        case_id=str(record.get("case_id", "")),
        status=status,
        action_type=str(record.get("action_type", "")),
        order_id=f"{order_id[:8]}...{order_id[-4:]}" if len(order_id) >= 12 else order_id,
        message_text=str(record.get("message_text", ""))[:800],
        created_at=str(record.get("created_at") or ""),
        updated_at=record.get("updated_at"),
        expires_at=record.get("expires_at"),
        appeal_count=int(record.get("appeal_count") or 0),
        request_id=str(sub_task.get("request_id") or payload.get("request_id") or ""),
        request_tier=str(sub_task.get("request_tier") or payload.get("request_tier") or ""),
        risk_level=str(sub_task.get("risk_level") or ""),
        policy_decision=policy_decision,
        auto_executed=status == "executed"
        and (str(record.get("reviewer_id") or "") == "system_auto" or bool(payload.get("auto_execute"))),
        sub_task_index=_optional_int(sub_task.get("index")),
        sub_task_total=_optional_int(sub_task.get("total")),
        display_status=_display_status(status),
    )


def _case_payload(record: dict) -> dict:
    raw = record.get("case_payload")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _case_sub_task(record: dict) -> dict:
    payload = _case_payload(record)
    sub_task = payload.get("sub_task")
    if not isinstance(sub_task, dict):
        draft = payload.get("draft")
        sub_task = draft.get("sub_task") if isinstance(draft, dict) else {}
    return sub_task if isinstance(sub_task, dict) else {}


def _case_request_id(record: dict) -> str:
    payload = _case_payload(record)
    return str(_case_sub_task(record).get("request_id") or payload.get("request_id") or "")


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _display_status(status: str) -> str:
    if status == "executed":
        return "processed"
    if status in {"pending_review", "appealed_pending_review", "approving"}:
        return "pending_review"
    if status == "rejected":
        return "rejected"
    if status in {"execution_failed", "initialization_failed"}:
        return "failed"
    if status == "withdrawn":
        return "withdrawn"
    return "processing"


def _public_case(record: dict) -> dict:
    safe = _customer_case_response(record).model_dump()
    safe.pop("message_text", None)
    return safe


def _progress_summary(cases: list[dict]) -> dict:
    counts = {
        "processed": 0,
        "pending_review": 0,
        "rejected": 0,
        "failed": 0,
        "processing": 0,
    }
    for item in cases:
        bucket = str(item.get("display_status") or _display_status(str(item.get("status") or "")))
        if bucket == "withdrawn":
            # Only present when it occurs, so existing consumers see the same shape.
            counts["withdrawn"] = counts.get("withdrawn", 0) + 1
            continue
        counts[bucket if bucket in counts else "processing"] += 1
    counts["total"] = len(cases)
    counts["partial_auto"] = bool(counts["processed"] and counts["pending_review"])
    return counts


def _progress_groups(cases: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {
        "pending_review": [],
        "processed": [],
        "rejected": [],
        "failed": [],
        "processing": [],
    }
    for item in cases:
        bucket = str(item.get("display_status") or _display_status(str(item.get("status") or "")))
        groups.setdefault(bucket if bucket in groups or bucket == "withdrawn" else "processing", []).append(
            item
        )
    return groups


def _request_progress(cases: list[dict], summary: dict) -> dict:
    tiers = {str(item.get("request_tier") or "") for item in cases if item.get("request_tier")}
    risks = {str(item.get("risk_level") or "") for item in cases if item.get("risk_level")}
    request_ids = {str(item.get("request_id") or "") for item in cases if item.get("request_id")}
    totals = [int(item.get("sub_task_total") or 0) for item in cases]
    return {
        "request_id": sorted(request_ids)[-1] if request_ids else "",
        "tier": "complex" if "complex" in tiers else ("simple" if tiers else ""),
        "risk_level": (
            "high" if "high" in risks else ("medium" if "medium" in risks else ("low" if risks else ""))
        ),
        "sub_task_total": max(totals) if totals else len(cases),
        "processed": summary["processed"],
        "pending_review": summary["pending_review"],
        "partial_auto": summary["partial_auto"],
    }


def _safe_request_analysis(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    keys = (
        "request_id",
        "tier",
        "risk_level",
        "intent_count",
        "action_types",
        "order_count",
        "amount_total",
        "evidence_completeness",
        "missing_evidence",
        "risk_flags",
        "decomposition_mode",
        "planning_mode",
        "sub_task_count",
        "write_task_count",
        "read_task_count",
        "auto_eligible_count",
        "high_risk_count",
    )
    return {key: value.get(key) for key in keys if key in value}


def _resolve_customer_actor(x_demo_customer: str | None) -> str:
    """Resolve the demo actor at the server boundary.

    A production BFF would obtain this value from a verified session/JWT and
    pass an immutable principal downstream.  Keeping the demo selector in a
    header makes the trust boundary explicit and avoids accepting ``user_id``
    from the chat JSON payload.
    """
    actor = (x_demo_customer or "demo-customer").strip()
    if actor not in DEFAULT_CUSTOMER_ORDER_IDS:
        raise HTTPException(status_code=401, detail="demo_customer_not_authenticated")
    return actor


def _validated_customer_page_context(request: ChatRequest, customer_user_id: str):
    context = request.page_context.model_copy()
    selected_order_id = (context.selected_order_id or "").strip().lower()
    if selected_order_id and selected_order_id not in _allowed_customer_order_ids(customer_user_id):
        raise HTTPException(status_code=403, detail="selected_order_owner_required")
    context.selected_order_id = selected_order_id or None
    return context


def _validated_customer_action(action: str | None) -> str | None:
    if action is None:
        return None
    allowed = {
        "track_order",
        "list_active_shipments",
        "list_attention_orders",
        "order_summary",
        "refund_policy",
        "refund_request",
        "cancel_order",
        "change_address",
        "invoice_request",
        "complaint_escalation",
        "open_support_case",
    }
    if action not in allowed:
        raise HTTPException(status_code=422, detail="unsupported_customer_action")
    return action


def _customer_context_payload(customer_user_id: str) -> dict[str, object]:
    """Build the compact personal order view used to ground the assistant.

    It intentionally exposes only customer-safe order fields.  Full facts are
    retrieved on demand after an owned order is selected or explicitly
    referenced, keeping raw history and PII out of the model context.
    """
    orders: list[dict[str, object]] = []
    active_shipments = 0
    attention_count = 0
    for order_id in _allowed_customer_order_ids(customer_user_id):
        order = olist_service.get_order_status(order_id)
        if order is None:
            continue
        in_transit = order.status in {"approved", "invoiced", "processing", "shipped"}
        needs_attention = bool(
            (isinstance(order.delay_days, int) and order.delay_days > 0)
            or order.refund_status
            or order.address_change_status
        )
        if in_transit:
            active_shipments += 1
        if needs_attention:
            attention_count += 1
        orders.append(
            {
                "order_id": order.order_id,
                "status": order.status,
                "category": order.category_summary,
                "category_label": _customer_category_label(order.category_summary),
                "amount": order.payment_value,
                "currency": "BRL",
                "purchased_at": order.purchase_timestamp,
                "estimated_delivery_at": order.estimated_delivery_date,
                "delivered_at": order.delivered_customer_date,
                "delay_days": order.delay_days,
                "refund_status": order.refund_status,
                "address_change_status": order.address_change_status,
                "needs_attention": needs_attention,
                "in_transit": in_transit,
                "actions": [action.model_dump() for action in _customer_follow_up_actions(order.order_id)],
            }
        )
    orders.sort(key=lambda item: str(item.get("purchased_at") or ""), reverse=True)
    return {
        "customer": {"id": customer_user_id, "display_name": "体验账户"},
        "summary": {
            "order_count": len(orders),
            "active_shipments": active_shipments,
            "attention_count": attention_count,
        },
        "orders": orders,
    }


_ORDER_ACTION_HINTS = (
    ("取消", "cancel_order", "取消订单"),
    ("退款", "refund_request", "退款申请"),
    ("退货", "refund_request", "退货退款"),
    ("不要了", "refund_request", "退货退款"),
    ("不想要", "refund_request", "退货退款"),
    ("地址", "change_address", "修改地址"),
    ("发票", "invoice_request", "发票申请"),
    ("投诉", "complaint_escalation", "投诉处理"),
)
_CATEGORY_ALIASES = (
    ("健康与美容", "health_beauty", "健康与美容用品"),
    ("母婴", "baby", "母婴用品"),
    ("家居", "bed_bath_table", "家居用品"),
    ("电脑", "computers_accessories", "电脑配件"),
)
_CATEGORY_LABELS = dict(CATEGORY_LABELS)


_CUSTOMER_ORDER_AGENT_HINTS = (
    "取消",
    "退款",
    "退货",
    "不要",
    "不想要",
    "地址",
    "发票",
    "投诉",
    "订单",
    "物流",
    "状态",
)
_CUSTOMER_POLICY_QUESTION_CUES = (
    "政策",
    "规则",
    "条件",
    "流程",
    "时效",
    "手续费",
    "费用",
    "多久",
    "怎么",
    "如何",
    "能不能",
    "可以吗",
    "是否",
    "一般",
    "通常",
    "需要满足",
    "要求",
)


def _should_use_customer_order_agent(message: str, page_context: object) -> bool:
    if getattr(page_context, "selected_order_id", None):
        return False
    if re.search(r"\b[0-9a-f]{32}\b", message.lower()):
        return False
    # "退款需要满足什么条件" is a policy question.  In live mode the order
    # tool agent would answer it from model knowledge, bypassing the retrieved
    # policy pack and leaving no trace.  Let the normal policy workflow answer.
    if any(cue in message for cue in _CUSTOMER_POLICY_QUESTION_CUES):
        return False
    return any(hint in message for hint in _CUSTOMER_ORDER_AGENT_HINTS)


def _customer_agent_actions(message: str, candidates: list[dict]) -> list[CustomerAction]:
    if "取消" in message:
        action_id, action_label = "cancel_order", "取消订单"
    elif any(word in message for word in ("退款", "退货", "不要", "不想要")):
        action_id, action_label = "refund_request", "退货退款"
    elif "地址" in message:
        action_id, action_label = "change_address", "修改地址"
    elif "发票" in message:
        action_id, action_label = "invoice_request", "发票申请"
    elif "投诉" in message:
        action_id, action_label = "complaint_escalation", "投诉处理"
    else:
        action_id, action_label = "open_support_case", "售后处理"
    actions = []
    for order in candidates[:3]:
        order_id = str(order.get("order_id") or "")
        if not order_id:
            continue
        actions.append(
            CustomerAction(
                id=action_id,
                label=(
                    f"{action_label} · {order.get('category') or '订单'} · "
                    f"{order.get('order_id_masked') or order_id[-6:]} · "
                    f"{order.get('status_label') or order.get('status') or ''}"
                ).strip(),
                style="primary" if not actions else "secondary",
                order_id=order_id,
                message=f"{action_label} 订单 {order_id}",
            )
        )
    return actions


def _customer_order_clarification(
    customer_user_id: str,
    message: str,
    page_context: object,
) -> tuple[str, list[CustomerAction]] | None:
    """Resolve an action intent to a concrete order before calling the agent.

    A standalone assistant should not ask the customer to type a 32-character
    order id.  When the intent is clear but the order is not, return owned
    order choices as native actions; the next message carries the exact id.
    """
    if getattr(page_context, "selected_order_id", None):
        return None
    if re.search(r"\b[0-9a-f]{32}\b", message.lower()):
        return None
    if any(cue in message for cue in _CUSTOMER_POLICY_QUESTION_CUES):
        return None
    matched = next((item for item in _ORDER_ACTION_HINTS if item[0] in message), None)
    if matched is None:
        return None
    _, action_id, action_label = matched
    orders = list(_customer_context_payload(customer_user_id).get("orders", []))
    category_label = None
    for alias, category, label in _CATEGORY_ALIASES:
        if alias in message:
            category_label = label
            scoped = [order for order in orders if order.get("category") == category]
            if scoped:
                orders = scoped
            break
    if action_id == "cancel_order":
        orders = [order for order in orders if order.get("status") != "canceled"]
    if not orders:
        return None
    status_labels = {
        "delivered": "已送达",
        "invoiced": "已开票",
        "canceled": "已取消",
        "shipped": "运输中",
        "processing": "处理中",
        "approved": "已确认",
    }
    actions = []
    for index, order in enumerate(orders[:3]):
        order_id = str(order.get("order_id"))
        short_id = f"{order_id[:8]}...{order_id[-4:]}"
        status = status_labels.get(str(order.get("status")), str(order.get("status")))
        raw_category = str(order.get("category") or "")
        category = _CATEGORY_LABELS.get(raw_category, raw_category or "订单")
        actions.append(
            CustomerAction(
                id=action_id,
                label=f"{action_label} · {category} · {short_id} · {status}",
                style="primary" if index == 0 else "secondary",
                order_id=order_id,
                message=f"{action_label} {order_id}",
            )
        )
    if category_label:
        return (
            f"理解，您不想要{category_label}了。请选择要处理的订单，我会按{action_label}继续核查：",
            actions,
        )
    return (
        f"可以，请选择要处理的订单，我会按{action_label}继续核查：",
        actions,
    )


def _customer_follow_up_actions(order_id: str | None = None) -> list[CustomerAction]:
    """Native order controls. They carry structured action intent, not IDs in prose."""
    if not order_id:
        return [
            CustomerAction(
                id="list_active_shipments",
                label="查看在途订单",
                message="我有哪些订单还在运输中？",
            ),
            CustomerAction(
                id="list_attention_orders",
                label="查看待处理订单",
                message="哪些订单需要我留意？",
            ),
            CustomerAction(
                id="order_summary",
                label="订单小结",
                message="帮我总结一下当前订单记录。",
            ),
            CustomerAction(
                id="refund_policy",
                label="了解退款政策",
                message="退款一般需要满足什么条件？",
            ),
        ]
    common = [
        CustomerAction(
            id="track_order", label="查看物流", order_id=order_id, message="帮我查看这笔订单的物流进展。"
        ),
    ]
    order = olist_service.get_order_status(order_id)
    status = order.status if order else ""
    if status in PRE_SHIPMENT_STATUSES:
        common.extend(
            [
                CustomerAction(
                    id="cancel_order",
                    label="取消订单",
                    order_id=order_id,
                    message="我想取消这笔订单。",
                    style="primary",
                ),
                CustomerAction(
                    id="change_address",
                    label="修改地址",
                    order_id=order_id,
                    message="我想修改这笔订单的收货地址。",
                ),
            ]
        )
    elif status in {"shipped", "delivered"}:
        common.extend(
            [
                CustomerAction(
                    id="refund_request",
                    label="申请退款",
                    order_id=order_id,
                    message="我想为这笔订单申请退款。",
                    style="primary",
                ),
                CustomerAction(
                    id="complaint_escalation",
                    label="申请售后",
                    order_id=order_id,
                    message="这笔订单遇到了问题，我想申请售后处理。",
                ),
            ]
        )
    elif status == "canceled":
        common.extend(
            [
                CustomerAction(
                    id="refund_request",
                    label="查询退款",
                    order_id=order_id,
                    message="我想查询这笔已取消订单的退款处理情况。",
                    style="primary",
                ),
                CustomerAction(
                    id="invoice_request",
                    label="申请发票",
                    order_id=order_id,
                    message="我想为这笔订单申请发票。",
                ),
            ]
        )
    else:
        common.append(
            CustomerAction(
                id="open_support_case",
                label="申请售后",
                order_id=order_id,
                message="这笔订单遇到了问题，我想申请售后处理。",
                style="primary",
            )
        )
    return common


def _resolve_auth_scopes(role: str, provided_scopes: list[str] | None) -> list[str]:
    if provided_scopes is not None:
        if role == "customer":
            allowed = set(DEFAULT_ROLE_SCOPES["customer"])
            return [scope for scope in provided_scopes if scope in allowed]
        return list(provided_scopes)
    return list(DEFAULT_ROLE_SCOPES.get(role, []))


_PII_FIELDS = ("电话", "手机号", "收货人", "收件人", "身份证", "cpf")
_PII_READ_CUES = ("发给我", "给我", "告诉我", "查询", "查一下", "多少", "是什么", "提供", "发一下")
_PII_CHANGE_CUES = ("改", "修改", "换")


def _customer_pii_request_denied(
    request: ChatRequest,
    *,
    customer_user_id: str,
) -> ChatResponse | None:
    """Refuse read access to recipient/phone/address fields.

    A change-address request contains the word "地址" but is a write intent;
    only read-style PII requests are blocked here.  Order facts exposed by the
    normal context intentionally exclude these fields.
    """
    message = request.message
    if not any(field in message for field in _PII_FIELDS):
        return None
    if any(cue in message for cue in _PII_CHANGE_CUES):
        return None
    if not any(cue in message for cue in _PII_READ_CUES):
        return None
    session_id = request.session_id or str(uuid.uuid4())
    answer = (
        "为了保护隐私，我不能提供收货人电话、地址等个人信息。"
        "如果是为了修改地址，请说明新地址并按售后流程提交。"
    )
    record_trace(
        session_id=session_id,
        user_message=message,
        result={"route_intent": "pii_guard", "final_answer": answer},
        latency_ms=0,
        status="customer_pii_request_blocked",
    )
    return ChatResponse(answer=answer, session_id=session_id, sources=[])


def _customer_order_access_denied(
    request: ChatRequest,
    *,
    customer_user_id: str,
) -> ChatResponse | None:
    requested_order_ids = _extract_order_ids(request.message)
    if not requested_order_ids:
        return None
    allowed_order_ids = _allowed_customer_order_ids(customer_user_id)
    unauthorized = [order_id for order_id in requested_order_ids if order_id not in allowed_order_ids]
    if not unauthorized:
        return None

    session_id = request.session_id or str(uuid.uuid4())
    answer = "为了保护订单隐私，我只能处理当前账号名下的订单。请确认登录账号或订单号后再试。"
    record_trace(
        session_id=session_id,
        user_message=request.message,
        result={
            "route_intent": "auth_guard",
            "final_answer": answer,
            "trajectory_events": [
                {
                    "node": "customer_order_access_guard",
                    "intent": "auth_guard",
                    "status": "blocked_unauthorized_order",
                    "details": {
                        "user_id": customer_user_id,
                        "requested_count": len(requested_order_ids),
                    },
                }
            ],
        },
        latency_ms=0,
        status="unauthorized_order_access",
    )
    return ChatResponse(answer=answer, session_id=session_id, sources=[])


def _extract_order_ids(text: str) -> list[str]:
    return [match.group(0).lower() for match in ORDER_ID_RE.finditer(text)]


def _history_user_turn_count(messages: list[dict[str, str]] | None) -> int:
    """Count visible customer turns without carrying graph work buffers forward."""

    return sum(1 for message in messages or [] if message.get("role") == "user")


def _allowed_customer_order_ids(user_id: str) -> set[str]:
    allowed = set(DEFAULT_CUSTOMER_ORDER_IDS.get(user_id, set()))
    configured = os.environ.get("DEMO_CUSTOMER_ORDER_IDS", "")
    allowed.update(order_id.strip().lower() for order_id in configured.split(",") if order_id.strip())
    return allowed


def _authorize_review(
    role: str,
    provided_scopes: list[str] | None,
    review_token: str | None,
) -> None:
    expected_token = os.environ.get("REVIEW_API_TOKEN", "local-review-demo")
    if review_token != expected_token:
        raise HTTPException(status_code=403, detail="review_token_required")
    scopes = _resolve_auth_scopes(role, provided_scopes)
    if role not in {"after_sales_operator", "admin"}:
        raise HTTPException(status_code=403, detail="review_role_required")
    if "*" not in scopes and "after_sales:write" not in scopes:
        raise HTTPException(status_code=403, detail="after_sales_write_scope_required")


def _is_confirmation_reply(message: str) -> bool:
    normalized = message.lower().strip()
    return normalized in {
        "yes",
        "yeah",
        "y",
        "confirm",
        "ok",
        "okay",
        "no",
        "n",
        "cancel",
        "确认",
        "创建",
        "升级",
        "取消",
        "不用",
        "不要",
        "算了",
        "放弃",
        "否",
    }


def _pending_confirmation_response(state: dict, *, customer_view: bool = False) -> dict[str, object]:
    pending = state.get("pending_side_effect", {}) if state else {}
    action_type = pending.get("type", "side_effect")
    expires_at = pending.get("expires_at")
    expires_hint = ""
    if isinstance(expires_at, (int, float)):
        remaining = max(int(expires_at - time.time()), 0)
        expires_hint = f"\n该确认将在 {remaining} 秒后超时自动取消。"
    if customer_view:
        final_answer = (
            f"你的售后申请正在等待工作人员审核：{action_type}。\n"
            "涉及退款、取消、改地址或投诉升级的动作需要由售后人员在审核台确认；"
            "你无需回复 yes/no 来触发执行。"
            f"{expires_hint}"
        )
    else:
        final_answer = (
            f"当前 session 还有一个待确认的副作用动作：{action_type}。\n"
            "请先回复 yes/确认 执行，或 no/取消 放弃；如果要开始新任务，请换一个 Session ID。"
            f"{expires_hint}"
        )
    return {
        "route_intent": "pending_confirmation",
        "final_answer": final_answer,
        "trajectory_events": [
            {
                "node": "chat",
                "intent": "pending_confirmation",
                "status": "blocked_new_message",
                "details": {"action_type": action_type},
            }
        ],
    }


async def _check_rate_limit(
    request: ChatRequest,
    *,
    user_id: str | None = None,
    tenant_id: str | None = None,
):
    limit = _env_int("AGENT_RATE_LIMIT_PER_MINUTE", 1000)
    if limit <= 0:
        return None
    key = f"tenant:{tenant_id or request.tenant_id}:user:{user_id or request.user_id}:minute"
    return await runtime_store.check_rate_limit(key, limit=limit, window_seconds=60)


def _rate_limited_chat_response(
    request: ChatRequest,
    decision,
    *,
    session_id: str | None = None,
    latency_ms: float = 0.0,
) -> ChatResponse:
    session_id = session_id or request.session_id or str(uuid.uuid4())
    answer = f"当前请求过于频繁，请稍后再试。预计 {decision.reset_after_seconds} 秒后恢复。"
    record_trace(
        session_id=session_id,
        user_message=request.message,
        result={"route_intent": "rate_limited", "final_answer": answer},
        latency_ms=latency_ms,
        status="rate_limited",
    )
    return ChatResponse(answer=answer, session_id=session_id, sources=[])


async def _pending_confirmation_expired(session_id: str, state: dict) -> bool:
    pending = state.get("pending_side_effect", {}) if state else {}
    if not pending.get("requires_confirmation"):
        return False
    if os.environ.get("RUNTIME_STORE_STRICT_HITL_TTL", "1").strip().lower() not in {"0", "false", "no"}:
        runtime_pending = await runtime_store.get_pending_confirmation(session_id)
        if runtime_pending is None:
            return True
    expires_at = pending.get("expires_at")
    return isinstance(expires_at, (int, float)) and time.time() >= float(expires_at)


async def _resume_review_session(
    session_id: str,
    *,
    resume_value: str,
    status: str,
    reviewer_id: str,
) -> ChatResponse:
    config: dict = {"configurable": {"thread_id": session_id}}
    snapshot = await agent.aget_state(config)
    if not snapshot.next:
        return ChatResponse(answer="该 session 当前没有待审核售后动作。", session_id=session_id, sources=[])

    t_start = time.perf_counter()
    if await _pending_confirmation_expired(session_id, snapshot.values):
        result = await agent.ainvoke(Command(resume="__hitl_timeout__"), config)
        status = "hitl_timeout_canceled"
        case_service.mark_review_result(session_id, "timeout_canceled", reviewer_id="system_timeout")
    else:
        result = await agent.ainvoke(Command(resume=resume_value), config)
        if status == "review_approved":
            case_service.mark_review_result(session_id, "executed", reviewer_id=reviewer_id)
        elif status == "review_rejected":
            case_service.mark_review_result(session_id, "rejected", reviewer_id=reviewer_id)
    record_trace(
        session_id=session_id,
        user_message=f"[staff_review:{status}:{reviewer_id}]",
        result=result,
        latency_ms=(time.perf_counter() - t_start) * 1000,
        status=status,
    )
    await _append_customer_review_update(session_id, result)
    return ChatResponse(
        answer=str(result.get("final_answer", "")),
        session_id=session_id,
        sources=_response_sources(result),
    )


async def _append_customer_review_update(
    session_id: str,
    result: dict,
    *,
    case_id: str | None = None,
    event_key: str | None = None,
    record: dict | None = None,
) -> None:
    """Publish a customer-safe status update after a staff review.

    The graph result contains evidence and receipts intended for the operator.
    Customers receive only the concise, grounded presentation, and the unique
    event_key makes retries idempotent.
    """
    record = record or (case_service.get(case_id) if case_id else case_service.get_by_session(session_id))
    messages = list(result.get("messages", []) or [])
    question = next(
        (
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
            and str(message.get("content", "")) not in {"__review_approve__", "no"}
        ),
        "售后申请处理进度",
    )
    customer_state = {**result, "final_answer": str(result.get("final_answer", ""))}
    customer_message = await present(llm_client.chat_openai, question, customer_state, record)
    await asyncio.to_thread(
        conversations.append_message,
        session_id,
        "staff",
        customer_message,
        case_id=case_id,
        event_key=event_key,
    )


def _review_session_response(session_id: str, has_pending: bool, state: dict) -> ReviewSessionResponse:
    draft = dict(state.get("escalation_draft", {}) or {})
    pending = dict(state.get("pending_side_effect", {}) or {})
    cases = list(state.get("after_sales_cases", []) or [])
    completed = list(state.get("completed_tasks", []) or [])
    events = list(state.get("trajectory_events", []) or [])
    action_type = draft.get("action_type") or pending.get("type") or ""
    order_id = str(draft.get("order_id") or state.get("active_order_id") or "")
    if not order_id:
        for message in reversed(state.get("messages", [])):
            match = ORDER_ID_RE.search(str(message.get("content", "")))
            if match:
                order_id = match.group(0)
                break
    order = olist_service.get_order_status(order_id) if order_id else None
    order_facts = (
        {
            "order_id": order.order_id,
            "order_status": order.status,
            "payment_value": order.payment_value,
            "delay_days": order.delay_days,
            "review_score": order.review_score,
        }
        if order
        else {}
    )
    summary = "当前没有待审核售后 case。"
    if has_pending and action_type:
        summary = f"待审核动作：{action_type}；订单：{order_id[:8] if order_id else 'unknown'}。"
    return ReviewSessionResponse(
        session_id=session_id,
        has_pending=has_pending,
        pending_side_effect=pending,
        escalation_draft=draft,
        after_sales_cases=cases,
        completed_tasks=completed,
        trajectory_events=events[-20:],
        customer_safe_summary=summary,
        order_facts=order_facts,
    )


def _require_demo_mode() -> None:
    """Keep the convenience review endpoint out of a deployed service."""
    enabled = os.environ.get("DEMO_MODE", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        raise HTTPException(status_code=404, detail="demo_mode_disabled")


async def _demo_session_payload(
    session_id: str,
    scenario: dict[str, str],
    *,
    customer_answer: str,
    review_decision: str | None = None,
) -> dict:
    """Return a small, visitor-safe view of a real workflow execution."""
    graph = conversation_agent or agent
    snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
    state = dict(snapshot.values or {})
    review = _review_session_response(session_id, bool(snapshot.next), state)
    draft = dict(review.escalation_draft or {})
    case = dict(review.after_sales_cases[-1] if review.after_sales_cases else {})
    decision = dict(draft.get("decision") or case.get("decision") or {})
    verification = dict(draft.get("verification") or case.get("verification") or {})
    record = case_service.get_by_session(session_id)

    timeline_labels = {
        "plan_tasks": "理解用户诉求",
        "extract_slots": "关联订单信息",
        "retrieve_context": "核对订单与政策",
        "build_after_sales_case": "生成售后处理建议",
        "hitl_gate": "转交人工审核",
        "finalize_escalation": "同步审核结果",
        "execute_write_action": "写入售后申请",
        "finalize_answer": "生成客户回复",
    }
    timeline = []
    for event in review.trajectory_events:
        node = str(event.get("node", ""))
        label = timeline_labels.get(node)
        if label:
            item = {"label": label, "status": str(event.get("status", "completed"))}
            if item not in timeline:
                timeline.append(item)
    if not timeline:
        timeline.append({"label": "完成本次咨询", "status": "completed"})

    return {
        "session_id": session_id,
        "scenario": {
            "title": scenario.get("title", "售后处理"),
            "summary": scenario.get("summary", ""),
            "order_id": review.order_facts.get("order_id") or scenario.get("order_id", ""),
        },
        "customer": {
            "request": scenario.get("message", ""),
            "answer": customer_answer,
        },
        "status": str(
            (record or {}).get("status") or ("pending_review" if review.has_pending else "resolved")
        ),
        "review_required": review.has_pending,
        "review_decision": review_decision,
        "review_packet": {
            "action_type": draft.get("action_type") or case.get("action_type", ""),
            "order_facts": review.order_facts,
            "risk_level": decision.get("risk_level", ""),
            "confidence": decision.get("confidence"),
            "handoff_reasons": decision.get("handoff_reasons", []),
            "evidence": decision.get("evidence", []),
            "policy_refs": decision.get("policy_refs", []),
            "customer_message_points": decision.get("customer_message_points", []),
            "verifier_passed": verification.get("passed"),
            "next_step": verification.get("required_next_step", ""),
        },
        "timeline": timeline,
    }


def _is_customer_request(request: ChatRequest) -> bool:
    return request.role == "customer" or request.channel == "customer_self_service"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


@app.get("/observability/summary", response_model=TraceSummaryResponse)
def observability_summary(
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> TraceSummaryResponse:
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    return TraceSummaryResponse(**trace_summary())


@app.get("/observability/case-metrics", response_model=CaseMetricsResponse)
def observability_case_metrics(
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> CaseMetricsResponse:
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    return CaseMetricsResponse(**case_metrics())


@app.get("/observability/runtime", response_model=RuntimeObservabilityResponse)
def observability_runtime(
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> RuntimeObservabilityResponse:
    """Live tool/policy/recovery telemetry for the staff observability view."""
    _authorize_review("after_sales_operator", ["after_sales:write"], x_review_token)
    snapshot = observability.runtime_snapshot()
    return RuntimeObservabilityResponse(
        **snapshot,
        traces=trace_summary(),
        cases=case_metrics(),
    )


@app.get("/observability/traces/{session_id}", response_model=TraceReplayResponse)
def replay_session_traces(
    session_id: str,
    limit: int = 20,
    role: str = Query(default="after_sales_operator"),
    auth_scopes: list[str] | None = Query(default=None),
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
) -> TraceReplayResponse:
    _authorize_review(role, auth_scopes, x_review_token)
    return TraceReplayResponse(session_id=session_id, traces=list_session_traces(session_id, limit=limit))


@app.get("/runtime/status", response_model=RuntimeStatusResponse)
def get_runtime_status() -> RuntimeStatusResponse:
    return RuntimeStatusResponse(**runtime_status)
