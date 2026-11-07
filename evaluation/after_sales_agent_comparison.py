# ruff: noqa: E501
"""Matched live evaluation for the after-sales Agent.

This evaluator intentionally distinguishes three things that were previously
blurred together in the project:

* workflow regression checks;
* a single-turn LLM decision baseline; and
* the complete customer -> review -> write-action workflow.

The baseline and the full workflow receive the same customer message, frozen
Olist facts, retrieved policy snippets, model, temperature, and task set.  The
only experimental variable is the workflow: the full system adds task state,
typed slot repair, deterministic decision/verifier gates, HITL, and durable
write execution.

It is deliberately a *curated gold suite*, not a claim about production user
traffic.  Results are written outside the repository by default and are only
publishable when the exact model id, prompt revision, commit and cases are
recorded together.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

import app.main as main_module
from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.case_workflow_service import CaseWorkflowService
from app.config.llm_settings import resolve_llm_settings
from app.llm.client import LlmClient
from app.llm.guardrail import Guardrail
from app.llm.intent_planner import IntentPlanner
from app.llm.json_fallback import add_json_instruction, native_structured_output_enabled, parse_json_model
from app.llm.response_generator import OlistTaskExtractor, PolicyResponseGenerator, QaResponseGenerator
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import OlistService, SQLiteCaseService
from app.retrieval.hybrid import HybridSupportRetriever
from app.tool_call import InMemoryRuntimeStore, build_business_tool_manager

ROOT = Path(__file__).resolve().parents[1]
REVIEW_HEADERS = {"X-Review-Token": "local-review-demo"}

DELAYED = "203096f03d82e0dffbc41ebc2e2bcfb7"
LOW_VALUE = "8aec3a066f732dd927ec8fef1752415b"
HIGH_VALUE = "360787554f41824600bdcba9687cd4f0"
CANCELED = "1b9ecfe83cdc259250e1a8aca174f0ad"
DELIVERED = "f98ae4bbfd1a32ea2104c8cf1f64dabf"


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    message: str
    tasks: tuple[str, ...]
    disposition: Literal["answer", "execute", "handoff", "reject", "clarify"]
    action_type: str = ""
    reviewer_action: Literal["approve", "reject", "none"] = "none"
    projection: tuple[tuple[str, str], ...] = ()


# These utterances are intentionally written independently of the system
# prompt and cover concrete order states in the frozen Olist snapshot.  They
# are a versioned gold suite for paired experiments, not synthetic template
# expansion or a claim to be production chat logs.
GOLD_CASES: tuple[GoldCase, ...] = (
    GoldCase("greeting", "你好", ("small_talk",), "answer"),
    GoldCase("ambiguous", "我这个订单不太对，麻烦看看。", ("clarify",), "clarify"),
    GoldCase("policy_refund", "退款申请通常需要满足哪些条件？", ("policy",), "answer"),
    GoldCase("policy_invoice", "开发票前需要准备什么资料？", ("policy",), "answer"),
    GoldCase("status_delayed", f"帮我看看 {DELAYED} 现在到哪一步了，为什么这么晚？", ("order_status",), "answer"),
    GoldCase("status_owned", f"订单 {DELIVERED} 已经签收了吗？", ("order_status",), "answer"),
    GoldCase("cancel_delivered", f"{DELAYED} 已经送到了，但我还是想取消这笔订单。", ("escalation",), "reject", "cancel_order"),
    GoldCase("cancel_canceled", f"请再帮我取消一次订单 {CANCELED}。", ("escalation",), "reject", "cancel_order"),
    GoldCase("cancel_low_value", f"订单 {LOW_VALUE} 还没发货，我不需要了，直接取消。", ("escalation",), "execute", "cancel_order", projection=(("order_status", "canceled"),)),
    GoldCase("address_low_value", f"订单 {LOW_VALUE} 的地址写错了，请尽快改一下。", ("escalation",), "execute", "change_address", projection=(("address_change_status", "requested"),)),
    GoldCase("cancel_high_value", f"我不想要订单 {HIGH_VALUE} 了，麻烦取消。", ("escalation",), "handoff", "cancel_order", "approve", (("order_status", "canceled"),)),
    GoldCase("refund_delay", f"订单 {DELAYED} 比预计晚了很多天，我要申请退款。", ("escalation",), "handoff", "refund_request", "approve", (("refund_status", "requested"),)),
    GoldCase("invoice", f"订单 {DELIVERED} 需要报销，请帮我申请发票。", ("escalation",), "handoff", "invoice_request", "approve", (("invoice_status", "requested"),)),
    GoldCase("complaint", f"订单 {DELAYED} 延误让我很不满意，请升级投诉。", ("escalation",), "handoff", "complaint_escalation", "reject"),
    GoldCase("refund_insufficient", f"订单 {DELIVERED} 我想退款，但暂时说不清具体原因。", ("escalation",), "clarify", "refund_request"),
    GoldCase("multi_status_policy_refund", f"先查 {DELAYED} 的物流，再说清退款条件，然后替我提交退款申请。", ("order_status", "policy", "escalation"), "handoff", "refund_request", "approve", (("refund_status", "requested"),)),
    GoldCase("multi_policy_invoice", f"先告诉我发票规则，再为订单 {DELIVERED} 发起开票申请。", ("policy", "escalation"), "handoff", "invoice_request", "approve", (("invoice_status", "requested"),)),
    GoldCase("multi_status_cancel", f"核对订单 {LOW_VALUE} 是否还未发货；若可以，请取消它。", ("order_status", "escalation"), "execute", "cancel_order", projection=(("order_status", "canceled"),)),
    GoldCase("mixed_chinese_english", f"track order {DELAYED}, explain the refund policy and submit a refund request", ("order_status", "policy", "escalation"), "handoff", "refund_request", "approve", (("refund_status", "requested"),)),
    GoldCase("address_delivered", f"货已经收到了，订单 {DELIVERED} 还能修改收货地址吗？", ("escalation",), "reject", "change_address"),
    GoldCase("repeat_cancel", f"我改主意了，仍然请取消订单 {LOW_VALUE}。", ("escalation",), "execute", "cancel_order", projection=(("order_status", "canceled"),)),
    GoldCase("delay_complaint_with_status", f"查一下 {DELAYED} 为什么延迟，并帮我把问题升级给人工处理。", ("order_status", "escalation"), "handoff", "complaint_escalation", "reject"),
    GoldCase("policy_and_safe_action", f"取消订单有什么限制？另外请取消 {LOW_VALUE}。", ("policy", "escalation"), "execute", "cancel_order", projection=(("order_status", "canceled"),)),
    GoldCase("policy_and_rejected_action", f"请解释取消政策，然后取消已经送达的订单 {DELIVERED}。", ("policy", "escalation"), "reject", "cancel_order"),
)


class DirectDecision(BaseModel):
    """The output contract of a deliberately thin, single-turn baseline."""

    tasks: list[str] = Field(default_factory=list)
    action_type: str | None = ""
    disposition: Literal["answer", "execute", "handoff", "reject", "clarify"]
    rationale: str = ""


DIRECT_BASELINE_PROMPT = """You are a single-turn e-commerce support agent.

Given a customer message, optional order facts, and policy excerpts, select an
ordered list of intents from: small_talk, clarify, order_status, policy,
escalation.  For a write request choose one action_type from: refund_request,
cancel_order, change_address, invoice_request, complaint_escalation,
open_support_case.  Choose exactly one disposition:
- answer: provide information only;
- execute: execute a low-risk write immediately;
- handoff: submit the write for a human reviewer;
- reject: do not perform an impossible or prohibited write;
- clarify: ask for missing information.

You make the decision directly. Do not assume a hidden workflow will repair a
missing task, parameter, policy decision, or unsafe write.
"""


def _build_runtime(db_path: Path, client: LlmClient) -> tuple[SQLiteCaseService, OlistService]:
    case_service = SQLiteCaseService(db_path)
    olist_service = OlistService(projection_store=case_service)
    knowledge_base = MarkdownKnowledgeBase()
    support_retriever = HybridSupportRetriever()
    runtime_store = InMemoryRuntimeStore()
    tool_manager = build_business_tool_manager(
        olist_service=olist_service,
        knowledge_base=knowledge_base,
        support_retriever=support_retriever,
        case_service=case_service,
        cache_backend=runtime_store,
        runtime_store=runtime_store,
    )
    actions = AgentActions(
        intent_planner=IntentPlanner(client.chat_openai),
        qa_generator=QaResponseGenerator(client.chat_openai),
        policy_generator=PolicyResponseGenerator(client.chat_openai),
        task_extractor=OlistTaskExtractor(client.chat_openai),
        olist_service=olist_service,
        knowledge_base=knowledge_base,
        support_retriever=support_retriever,
        case_service=case_service,
        tool_manager=tool_manager,
        runtime_store=runtime_store,
    )
    checkpointer = InMemorySaver()
    conversation_graph = AgentGraph(actions).build(checkpointer)
    workflow = CaseWorkflowService(case_service, actions)
    workflow.configure(checkpointer, legacy_agent=None)
    actions.bind_case_workflow(workflow)
    main_module.agent = conversation_graph
    main_module.conversation_agent = conversation_graph
    main_module.case_workflow_service = workflow
    main_module.case_service = case_service
    main_module.olist_service = olist_service
    main_module.runtime_store = runtime_store
    main_module.guardrail = Guardrail(client.chat_openai)
    main_module.llm_client = client
    return case_service, olist_service


async def _direct_baseline(client: LlmClient, service: OlistService, kb: MarkdownKnowledgeBase, case: GoldCase) -> DirectDecision:
    order_id = _order_id(case.message)
    order = service.get_order_status(order_id) if order_id else None
    policy_hits = kb.search(case.message, k=3)
    context = {
        "customer_message": case.message,
        "order_facts": order.__dict__ if order else {},
        "policy_excerpts": [
            {"title": hit.section_title, "text": hit.text}
            for hit in policy_hits
        ],
    }
    messages = [
        SystemMessage(content=DIRECT_BASELINE_PROMPT),
        HumanMessage(content=json.dumps(context, ensure_ascii=False)),
    ]
    if native_structured_output_enabled(client.chat_openai):
        structured = client.chat_openai.with_structured_output(DirectDecision)
        return await structured.ainvoke(messages)
    raw = await client.chat_openai.ainvoke(add_json_instruction(messages, DirectDecision))
    return parse_json_model(raw, DirectDecision)


async def _run_full(client: AsyncClient, store: SQLiteCaseService, case: GoldCase) -> dict[str, object]:
    session_id = f"paired-full-{case.case_id}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    response = await client.post(
        "/customer/chat",
        json={"message": case.message, "session_id": session_id, "user_id": "demo-customer"},
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = response.json()
    trace_response = await client.get(f"/observability/traces/{session_id}", headers=REVIEW_HEADERS)
    planned = _plan_from_trace(trace_response.json() if trace_response.status_code == 200 else {})
    tasks = [str(task.get("intent", "")) for task in planned]
    record = store.get_by_session(session_id)
    pre_review_status = str(record.get("status")) if record else ""
    reviewer_ok = True
    if record and pre_review_status == "pending_review" and case.reviewer_action != "none":
        endpoint = "approve" if case.reviewer_action == "approve" else "reject"
        reviewed = await client.post(
            f"/review/sessions/{session_id}/{endpoint}",
            headers=REVIEW_HEADERS,
            json={"reviewer_id": "paired-eval", "role": "after_sales_operator", "auth_scopes": ["after_sales:write"]},
        )
        reviewer_ok = reviewed.status_code == 200
        record = store.get_by_session(session_id)
    order_id = _order_id(case.message)
    projection = store.get_order_projection(order_id) if order_id else {}
    event_count = store.count_order_events(order_id) if order_id else 0
    final_status = str(record.get("status")) if record else ""
    disposition = _full_disposition(
        pre_review_status,
        final_status,
        payload.get("answer", ""),
        event_count=event_count,
        tasks=tasks,
    )
    expected_projection = all(projection.get(key) == value for key, value in case.projection)
    task_match = tuple(tasks) == case.tasks
    planned_actions = [str(task.get("action_type", "")) for task in planned]
    actual_action = str(record.get("action_type")) if record else next((action for action in planned_actions if action and action != "none"), "")
    action_match = not case.action_type or actual_action == case.action_type
    passed = response.status_code == 200 and task_match and disposition == case.disposition and action_match and expected_projection and reviewer_ok
    return {
        "case_id": case.case_id,
        "tasks": tasks,
        "disposition": disposition,
        "action_type": actual_action,
        "latency_ms": latency_ms,
        "event_count": event_count,
        "task_match": task_match,
        "disposition_match": disposition == case.disposition,
        "action_match": action_match,
        "projection_match": expected_projection,
        "passed": passed,
    }


def _score_baseline(
    store: SQLiteCaseService,
    case: GoldCase,
    decision: DirectDecision,
    latency_ms: float,
) -> dict[str, object]:
    task_match = tuple(_canonical_tasks(decision.tasks)) == case.tasks
    action_match = not case.action_type or (decision.action_type or "") == case.action_type
    disposition_match = decision.disposition == case.disposition
    action_type = decision.action_type or ""
    order_id = _order_id(case.message)
    event_count = _apply_baseline_state(store, case, decision, order_id)
    projection = store.get_order_projection(order_id) if order_id else {}
    projection_match = all(projection.get(key) == value for key, value in case.projection)
    no_unexpected_write = bool(case.projection) or event_count == 0
    state_match = projection_match and no_unexpected_write
    unsafe_write = (
        case.disposition in {"handoff", "reject", "clarify"}
        and decision.disposition == "execute"
        and event_count > 0
    )
    return {
        "case_id": case.case_id,
        "tasks": _canonical_tasks(decision.tasks),
        "disposition": decision.disposition,
        "action_type": action_type,
        "rationale": decision.rationale,
        "latency_ms": latency_ms,
        "event_count": event_count,
        "task_match": task_match,
        "disposition_match": disposition_match,
        "action_match": action_match,
        "state_match": state_match,
        "unsafe_write": unsafe_write,
        "passed": task_match and disposition_match and action_match and state_match,
    }


def _summary(rows: list[dict[str, object]], name: str) -> dict[str, object]:
    def rate(field: str) -> float:
        return round(sum(bool(row.get(field)) for row in rows) / len(rows), 4) if rows else 0.0

    latency = [float(row["latency_ms"]) for row in rows]
    return {
        "agent": name,
        "cases": len(rows),
        "gold_action_path_success": rate("passed"),
        "task_plan_exact_match": rate("task_match"),
        "disposition_accuracy": rate("disposition_match"),
        "action_type_accuracy": rate("action_match"),
        "unsafe_write_rate": rate("unsafe_write"),
        "p50_latency_ms": round(statistics.median(latency), 2) if latency else 0.0,
        "p95_latency_ms": _percentile(latency, 0.95),
    }


def _apply_baseline_state(
    store: SQLiteCaseService,
    case: GoldCase,
    decision: DirectDecision,
    order_id: str,
) -> int:
    """Apply the baseline's selected action to the same durable state store.

    This keeps the paired score about final action-path correctness rather than
    only a JSON classification.  Reviewer decisions are fixed by the gold case
    for both variants; the model still decides whether to hand off at all.
    """

    action_type = decision.action_type or ""
    if not order_id or not action_type:
        return 0
    message_text = f"paired-baseline:{case.case_id}"
    if decision.disposition == "execute":
        store.execute_action(action_type, order_id, message_text)
    elif decision.disposition == "handoff":
        session_id = f"paired-baseline-{case.case_id}"
        store.submit_for_review(
            action_type=action_type,
            order_id=order_id,
            message_text=message_text,
            session_id=session_id,
            user_id="demo-customer",
            expires_at=None,
        )
        if case.reviewer_action == "approve":
            store.execute_action(action_type, order_id, message_text)
        elif case.reviewer_action == "reject":
            store.mark_review_result(session_id, "rejected", reviewer_id="paired-eval")
    return store.count_order_events(order_id)


async def run(
    limit: int | None,
    output: Path,
    reuse_baseline: Path | None = None,
) -> dict[str, object]:
    settings = resolve_llm_settings(default_model="deepseek-chat")
    if settings.provider == "offline":
        raise RuntimeError("Set OPENAI_API_KEY or AIHUBMIX_API_KEY before a paired live evaluation.")
    client = LlmClient(api_key=settings.api_key, model=settings.model, base_url=settings.base_url)
    if client.mode != "live_llm_agent":
        raise RuntimeError("The configured key selected offline mode; paired results require a live model.")
    cases = list(GOLD_CASES[:limit] if limit else GOLD_CASES)
    db_path = ROOT / "data" / f"paired_agent_eval_{uuid.uuid4().hex[:8]}.db"
    store, service = _build_runtime(db_path, client)
    kb = MarkdownKnowledgeBase()
    transport = ASGITransport(app=main_module.app)
    baseline_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    if reuse_baseline:
        payload = json.loads(reuse_baseline.read_text(encoding="utf-8"))
        previous_rows = payload.get("baseline_rows", [])
        by_case_id = {
            str(row.get("case_id")): dict(row)
            for row in previous_rows
            if isinstance(row, dict)
        }
        missing = [case.case_id for case in cases if case.case_id not in by_case_id]
        if missing:
            raise ValueError(f"Baseline file is missing cases: {', '.join(missing)}")
        baseline_rows = [by_case_id[case.case_id] for case in cases]
    async with AsyncClient(transport=transport, base_url="http://paired-eval") as http:
        for case in cases:
            store.reset()
            if not reuse_baseline:
                started = time.perf_counter()
                decision = await _direct_baseline(client, service, kb, case)
                baseline_rows.append(
                    _score_baseline(
                        store,
                        case,
                        decision,
                        round((time.perf_counter() - started) * 1000, 2),
                    )
                )
            store.reset()
            full_rows.append(await _run_full(http, store, case))
    result = {
        "scope": "Curated Olist after-sales gold suite; paired same-model live comparison.",
        "model": client.model,
        "case_count": len(cases),
        "baseline": _summary(baseline_rows, "single_turn_llm_decision"),
        "full_workflow": _summary(full_rows, "langgraph_after_sales_workflow"),
        "delta_percentage_points": {
            field: round(
                (
                    float(_summary(full_rows, "full")[field])
                    - float(_summary(baseline_rows, "base")[field])
                )
                * 100,
                2,
            )
            for field in (
                "gold_action_path_success",
                "task_plan_exact_match",
                "disposition_accuracy",
                "action_type_accuracy",
                "unsafe_write_rate",
            )
        },
        "baseline_rows": baseline_rows,
        "full_workflow_rows": full_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("rows")}, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return result


def _plan_from_trace(payload: dict[str, object]) -> list[dict[str, object]]:
    for trace in reversed(list(payload.get("traces", []))):
        if not isinstance(trace, dict):
            continue
        try:
            events = json.loads(str(trace.get("trajectory_json") or "[]"))
        except json.JSONDecodeError:
            continue
        for event in events:
            if isinstance(event, dict) and event.get("node") == "plan_tasks":
                tasks = event.get("details", {}).get("tasks", [])
                if isinstance(tasks, list):
                    return [dict(task) for task in tasks if isinstance(task, dict)]
    return []


def _order_id(message: str) -> str:
    match = re.search(r"[0-9a-f]{32}", message.lower())
    return match.group(0) if match else ""


def _full_disposition(
    pre_review_status: str,
    final_status: str,
    answer: object,
    *,
    event_count: int,
    tasks: list[str],
) -> str:
    if pre_review_status == "pending_review":
        return "handoff"
    if final_status == "executed" or event_count > 0:
        return "execute"
    if "clarify" in tasks:
        return "clarify"
    if "escalation" not in tasks:
        return "answer"
    text = str(answer)
    if "补充" in text or "请提供" in text or "缺少" in text:
        return "clarify"
    if "不能" in text or "无法" in text or "不支持" in text:
        return "reject"
    return "answer"


def _canonical_tasks(tasks: list[str]) -> list[str]:
    action_names = {
        "refund_request",
        "cancel_order",
        "change_address",
        "invoice_request",
        "complaint_escalation",
        "open_support_case",
    }
    return ["escalation" if task in action_names else task for task in tasks]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return round(values[max(0, min(len(values) - 1, int(p * len(values) + 0.999) - 1))], 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paired live after-sales Agent evaluation.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "paired_live_results.json")
    parser.add_argument(
        "--reuse-baseline",
        type=Path,
        default=None,
        help="Reuse matched baseline rows from a previous result and only rerun the full workflow.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.limit, args.output, args.reuse_baseline))
