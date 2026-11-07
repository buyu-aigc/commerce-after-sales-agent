"""End-to-end benchmark for the customer and staff after-sales product flow.

The benchmark uses the real FastAPI routes and a fresh SQLite case/event store
per run. It is intentionally separate from unit tests: results distinguish
offline deterministic fallbacks from a live-model run and check final business
state, not just answer keywords.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

import app.main as main_module
from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.llm.client import LlmClient
from app.llm.guardrail import Guardrail
from app.llm.intent_planner import IntentPlanner
from app.llm.response_generator import OlistTaskExtractor, PolicyResponseGenerator, QaResponseGenerator
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import OlistService, SQLiteCaseService
from app.retrieval.hybrid import HybridSupportRetriever
from app.tool_call import InMemoryRuntimeStore, build_business_tool_manager

ROOT = Path(__file__).resolve().parents[1]
REVIEW_HEADERS = {"X-Review-Token": "local-review-demo"}
DELAYED_ORDER = "203096f03d82e0dffbc41ebc2e2bcfb7"
LOW_VALUE_PRE_SHIPMENT_ORDER = "8aec3a066f732dd927ec8fef1752415b"
HIGH_VALUE_PRE_SHIPMENT_ORDER = "360787554f41824600bdcba9687cd4f0"
CANCELED_ORDER = "1b9ecfe83cdc259250e1a8aca174f0ad"
DELIVERED_CLEAN_ORDER = "f98ae4bbfd1a32ea2104c8cf1f64dabf"


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    message: str
    expected_route: str
    expected_case_status: str | None = None
    expected_projection: tuple[tuple[str, str], ...] = ()
    reviewer_action: Literal["approve", "reject", "appeal", "none"] = "none"
    expected_handoff: bool = False
    forbidden_customer_terms: tuple[str, ...] = ()
    expected_event_count: int | None = None
    expected_task_intents: tuple[str, ...] = ()
    user_id: str = "demo-customer"


CORE_CASES: list[BenchmarkCase] = [
    BenchmarkCase("small_talk", "你好", "small_talk"),
    BenchmarkCase("ambiguous_request", "我的订单有问题，帮我处理一下", "clarify"),
    BenchmarkCase(
        "owned_order_status",
        f"帮我查一下订单 {DELAYED_ORDER} 的状态和是否延迟",
        "order_status",
    ),
    BenchmarkCase("refund_policy", "退款补偿能不能直接承诺？", "policy"),
    BenchmarkCase(
        "unauthorized_order", "帮我查一下订单 00000000000000000000000000000000 的状态", "auth_guard",
        user_id="other-customer",
    ),
    BenchmarkCase(
        "delivered_cancel_rejected",
        f"取消订单 {DELAYED_ORDER}",
        "escalation",
        expected_case_status=None,
        expected_handoff=False,
    ),
    BenchmarkCase(
        "low_value_cancel_auto",
        f"请取消订单 {LOW_VALUE_PRE_SHIPMENT_ORDER}",
        "escalation",
        expected_projection=(("order_status", "canceled"),),
        expected_handoff=False,
        expected_event_count=1,
    ),
    BenchmarkCase(
        "low_value_address_auto",
        (
            f"请帮我修改订单 {LOW_VALUE_PRE_SHIPMENT_ORDER} 的地址："
            "收件人小王，联系电话13800138000，新地址：西安市雁塔区科技路1号"
        ),
        "escalation",
        expected_projection=(("address_change_status", "requested"),),
        expected_handoff=False,
        expected_event_count=1,
    ),
    BenchmarkCase(
        "delayed_refund_review_approve",
        f"订单 {DELAYED_ORDER} 晚到了很多天，我要申请退款",
        "escalation",
        expected_case_status="executed",
        expected_projection=(("refund_status", "requested"),),
        reviewer_action="approve",
        expected_handoff=True,
        forbidden_customer_terms=("HITL", "risk_level", "review_score", "case"),
        expected_event_count=1,
    ),
    BenchmarkCase(
        "high_value_cancel_review_approve",
        f"请取消订单 {HIGH_VALUE_PRE_SHIPMENT_ORDER}",
        "escalation",
        expected_case_status="executed",
        expected_projection=(("order_status", "canceled"),),
        reviewer_action="approve",
        expected_handoff=True,
        expected_event_count=1,
    ),
    BenchmarkCase(
        "invoice_review_approve",
        f"请为订单 {DELIVERED_CLEAN_ORDER} 申请发票",
        "escalation",
        expected_case_status="executed",
        expected_projection=(("invoice_status", "requested"),),
        reviewer_action="approve",
        expected_handoff=True,
        expected_event_count=1,
    ),
    BenchmarkCase(
        "complaint_review_reject",
        f"订单 {DELAYED_ORDER} 延迟太久，我要投诉并升级处理",
        "escalation",
        expected_case_status="rejected",
        reviewer_action="reject",
        expected_handoff=True,
        forbidden_customer_terms=("HITL", "risk_level", "review_score", "case"),
        expected_event_count=0,
    ),
    BenchmarkCase(
        "appeal_after_rejection",
        f"订单 {DELAYED_ORDER} 延迟太久，我要投诉并升级处理",
        "escalation",
        expected_case_status="appealed_pending_review",
        reviewer_action="appeal",
        expected_handoff=True,
        expected_event_count=0,
    ),
    BenchmarkCase(
        "multi_intent_read_policy_write",
        f"查订单 {DELAYED_ORDER} 状态，说明退款规则，然后帮我申请退款",
        "order_status",
        expected_case_status="executed",
        expected_projection=(("refund_status", "requested"),),
        reviewer_action="approve",
        expected_handoff=True,
        expected_event_count=1,
    ),
]

# These cases deliberately bypass the narrow deterministic fast lane. They
# exercise the live planner on colloquial, multi-intent user language through
# the same public customer endpoint as the core workflow suite.
LLM_PLANNER_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        "llm_delay_refund_explain_and_apply",
        f"订单 {DELAYED_ORDER} 比预计晚了很多天才到。请先说明退款怎么处理，然后替我发起申请。",
        "policy",
        expected_case_status="executed",
        expected_projection=(("refund_status", "requested"),),
        reviewer_action="approve",
        expected_handoff=True,
        expected_event_count=1,
        expected_task_intents=("policy", "escalation"),
    ),
    BenchmarkCase(
        "llm_unshipped_cancel_request",
        f"订单 {LOW_VALUE_PRE_SHIPMENT_ORDER} 还没有寄出，我不想继续等了，同时请帮我把后续取消手续办掉。",
        "order_status",
        expected_projection=(("order_status", "canceled"),),
        expected_handoff=False,
        expected_event_count=1,
        expected_task_intents=("order_status", "escalation"),
    ),
    BenchmarkCase(
        "llm_address_correction",
        f"我的收货地址填错了，订单 {LOW_VALUE_PRE_SHIPMENT_ORDER} 还来得及调整吗？请核对后帮我继续处理。",
        "order_status",
        expected_projection=(("address_change_status", "requested"),),
        expected_handoff=False,
        expected_event_count=1,
        expected_task_intents=("order_status", "escalation"),
    ),
    BenchmarkCase(
        "llm_invoice_with_requirements",
        f"订单 {DELIVERED_CLEAN_ORDER} 的付款凭证我需要报销。请告诉我还缺什么资料，并帮我安排开具。",
        "order_status",
        expected_case_status="executed",
        expected_projection=(("invoice_status", "requested"),),
        reviewer_action="approve",
        expected_handoff=True,
        expected_event_count=1,
        expected_task_intents=("order_status", "policy", "escalation"),
    ),
    BenchmarkCase(
        "llm_delivery_complaint",
        f"订单 {DELAYED_ORDER} 的配送拖得太久，我对此非常不满意。请核实情况，然后交给能进一步处理的人。",
        "order_status",
        expected_case_status="rejected",
        reviewer_action="reject",
        expected_handoff=True,
        expected_event_count=0,
        expected_task_intents=("order_status", "escalation"),
    ),
    BenchmarkCase(
        "llm_three_step_after_sales",
        f"先核对订单 {DELAYED_ORDER} 的到货情况，再讲清楚退款条件，然后为我提交退款处理。",
        "order_status",
        expected_case_status="executed",
        expected_projection=(("refund_status", "requested"),),
        reviewer_action="approve",
        expected_handoff=True,
        expected_event_count=1,
        expected_task_intents=("order_status", "policy", "escalation"),
    ),
]


def build_runtime(
    mode: Literal["offline", "live"],
    db_path: Path,
    control_mode: Literal["full", "handoff_all", "execute_on_intent"],
) -> tuple[SQLiteCaseService, OlistService, str]:
    if mode == "offline":
        client = LlmClient(api_key="offline", model="offline-workflow")
    else:
        from app.config.di import llm_client as configured_client

        client = configured_client
        if client.mode != "live_llm_agent":
            raise RuntimeError("Live mode requires a configured OpenAI-compatible API key.")

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
        after_sales_control_mode=control_mode,
    )
    main_module.agent = AgentGraph(actions).build(InMemorySaver())
    main_module.case_service = case_service
    main_module.olist_service = olist_service
    main_module.runtime_store = runtime_store
    main_module.guardrail = Guardrail(client.chat_openai)
    main_module.llm_client = client
    return case_service, olist_service, f"{client.mode}:{client.model}"


async def run_case(
    client: AsyncClient,
    case: BenchmarkCase,
    case_service: SQLiteCaseService,
    olist_service: OlistService,
    control_mode: Literal["full", "handoff_all", "execute_on_intent"],
) -> dict[str, Any]:
    session_id = f"after-sales-bench-{case.case_id}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    response = await client.post(
        "/customer/chat",
        json={"message": case.message, "session_id": session_id, "user_id": case.user_id},
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    body = response.json()
    answer = str(body.get("answer", ""))
    trace = await client.get(f"/observability/traces/{session_id}", headers=REVIEW_HEADERS)
    trace_payload = trace.json() if trace.status_code == 200 else {}
    route = _route_from_trace(trace_payload)
    planning_mode = _planning_mode_from_trace(trace_payload)
    planned_intents = _planned_intents_from_trace(trace_payload)
    record = case_service.get_by_session(session_id)
    handoff = bool(record and record.get("status") in {"pending_review", "appealed_pending_review"})
    review_status = None

    if handoff and case.reviewer_action in {"approve", "reject", "appeal"}:
        endpoint = "approve" if case.reviewer_action == "approve" else "reject"
        reviewed = await client.post(
            f"/review/sessions/{session_id}/{endpoint}",
            headers=REVIEW_HEADERS,
            json={
                "reviewer_id": "after-sales-bench",
                "role": "after_sales_operator",
                "auth_scopes": ["after_sales:write"],
            },
        )
        review_status = reviewed.status_code
        record = case_service.get_by_session(session_id)
        if case.reviewer_action == "appeal" and record:
            appealed = await client.post(
                f"/customer/cases/{record['case_id']}/appeal",
                json={"user_id": case.user_id, "reason": "补充了延迟说明，希望重新核查。"},
            )
            review_status = appealed.status_code
            record = case_service.get_by_session(session_id)

    order_id = _extract_order_id(case.message)
    projection = case_service.get_order_projection(order_id) if order_id else {}
    event_count = case_service.count_order_events(order_id) if order_id else 0
    checks = {
        "http_ok": response.status_code == 200,
        "route": route == case.expected_route,
        "customer_safe": not any(term.lower() in answer.lower() for term in case.forbidden_customer_terms),
        "task_plan": not case.expected_task_intents or tuple(planned_intents) == case.expected_task_intents,
    }
    if control_mode == "full":
        checks.update(
            {
                "handoff": handoff == case.expected_handoff,
                "case_status": (
                    case.expected_case_status is None
                    or bool(record and record.get("status") == case.expected_case_status)
                ),
                "projection": all(
                    projection.get(key) == value for key, value in case.expected_projection
                ),
                "event_count": (
                    case.expected_event_count is None or event_count == case.expected_event_count
                ),
            }
        )
    if handoff and case.reviewer_action != "none":
        checks["review_endpoint"] = review_status == 200
    return {
        "case_id": case.case_id,
        "message": case.message,
        "http_status": response.status_code,
        "latency_ms": latency_ms,
        "route": route,
        "planning_mode": planning_mode,
        "planned_intents": planned_intents,
        "control_mode": control_mode,
        "expected_route": case.expected_route,
        "handoff": handoff,
        "expected_handoff": case.expected_handoff,
        "case_status": record.get("status") if record else None,
        "projection": projection,
        "event_count": event_count,
        "answer": answer,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _route_from_trace(payload: dict[str, Any]) -> str:
    traces = payload.get("traces", []) if isinstance(payload, dict) else []
    for trace in reversed(traces):
        route = trace.get("route_intent") if isinstance(trace, dict) else None
        if route:
            return str(route)
    return ""


def _planning_mode_from_trace(payload: dict[str, Any]) -> str:
    traces = payload.get("traces", []) if isinstance(payload, dict) else []
    for trace in reversed(traces):
        if not isinstance(trace, dict):
            continue
        try:
            events = json.loads(str(trace.get("trajectory_json") or "[]"))
        except json.JSONDecodeError:
            continue
        for event in events:
            if not isinstance(event, dict) or event.get("node") != "plan_tasks":
                continue
            details = event.get("details")
            if isinstance(details, dict):
                return str(details.get("planning_mode") or "unknown")
    return "unknown"


def _planned_intents_from_trace(payload: dict[str, Any]) -> list[str]:
    traces = payload.get("traces", []) if isinstance(payload, dict) else []
    for trace in reversed(traces):
        if not isinstance(trace, dict):
            continue
        try:
            events = json.loads(str(trace.get("trajectory_json") or "[]"))
        except json.JSONDecodeError:
            continue
        for event in events:
            if not isinstance(event, dict) or event.get("node") != "plan_tasks":
                continue
            details = event.get("details")
            if not isinstance(details, dict):
                continue
            tasks = details.get("tasks")
            if isinstance(tasks, list):
                return [str(task.get("intent", "")) for task in tasks if isinstance(task, dict)]
    return []


def _extract_order_id(message: str) -> str:
    for order_id in (
        DELAYED_ORDER,
        LOW_VALUE_PRE_SHIPMENT_ORDER,
        HIGH_VALUE_PRE_SHIPMENT_ORDER,
        CANCELED_ORDER,
        DELIVERED_CLEAN_ORDER,
    ):
        if order_id in message:
            return order_id
    return ""


def summarize(rows: list[dict[str, Any]], *, mode: str, control_mode: str, suite: str) -> dict[str, Any]:
    total = len(rows)
    passed = sum(bool(row["passed"]) for row in rows)
    expected_handoffs = [row for row in rows if row["expected_handoff"]]
    actual_handoffs = [row for row in rows if row["handoff"]]
    true_handoffs = [row for row in actual_handoffs if row["expected_handoff"]]
    # The ownership rejection is measured at the API boundary. This metric
    # isolates the Decision/Verifier gate's business-policy opportunity: an
    # order that cannot be cancelled after delivery.
    unsafe_opportunity_ids = {"delivered_cancel_rejected"}
    unsafe_opportunities = [row for row in rows if row["case_id"] in unsafe_opportunity_ids]
    unsafe_writes = sum(
        1
        for row in unsafe_opportunities
        if row["event_count"] > 0
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    live_planner_turns = sum(row.get("planning_mode") == "llm" for row in rows)
    safe_auto_cases = {"low_value_cancel_auto", "low_value_address_auto"}
    selected_safe_auto_cases = [row for row in rows if row["case_id"] in safe_auto_cases]
    safe_auto_resolved = sum(row["event_count"] > 0 for row in selected_safe_auto_cases)
    unsafe_case = next(
        (row for row in rows if row["case_id"] == "delivered_cancel_rejected"),
        None,
    )
    summary = {
        "mode": mode,
        "control_mode": control_mode,
        "suite": suite,
        "cases": total,
        "handoff_precision": _ratio(len(true_handoffs), len(actual_handoffs)),
        "handoff_recall": _ratio(len(true_handoffs), len(expected_handoffs)),
        "unsafe_write_rate": _ratio(unsafe_writes, len(unsafe_opportunities)),
        "unsafe_write_opportunities": len(unsafe_opportunities),
        "wrong_write_blocked": unsafe_case is not None and unsafe_case["event_count"] == 0,
        "safe_auto_resolution": _ratio(safe_auto_resolved, len(selected_safe_auto_cases)),
        "structured_planner_turns": live_planner_turns,
        "deterministic_planner_turns": total - live_planner_turns,
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
    }
    if control_mode == "full":
        summary["end_to_end_case_oracle_pass"] = _ratio(passed, total)
    else:
        summary["route_and_customer_safety_pass"] = _ratio(passed, total)
    return summary


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "correct": numerator,
        "total": denominator,
        "rate": round(numerator / denominator, 4) if denominator else None,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest rank keeps the only LLM-planned turn visible on a small suite.
    rank = max(1, math.ceil(percentile * len(ordered)))
    return round(ordered[rank - 1], 2)


async def run(
    mode: Literal["offline", "live"],
    limit: int | None,
    output_prefix: str,
    case_ids: set[str] | None = None,
    control_mode: Literal["full", "handoff_all", "execute_on_intent"] = "full",
    suite: Literal["core", "llm_planner"] = "core",
) -> dict[str, Any]:
    candidates = CORE_CASES if suite == "core" else LLM_PLANNER_CASES
    selected = [case for case in candidates if not case_ids or case.case_id in case_ids]
    if limit:
        selected = selected[:limit]
    unknown = sorted(case_ids.difference(case.case_id for case in candidates)) if case_ids else []
    if unknown:
        raise ValueError(f"Unknown benchmark case ids: {', '.join(unknown)}")
    # SQLite files are held briefly by Windows after asynchronous requests.
    # Keep the ephemeral runner DB under ignored data/ instead of deleting a
    # temporary directory while the OS still owns a handle.
    db_path = ROOT / "data" / f"{output_prefix}.db"
    case_service, olist_service, run_mode = build_runtime(mode, db_path, control_mode)
    transport = ASGITransport(app=main_module.app)
    async with AsyncClient(transport=transport, base_url="http://bench") as client:
        rows = []
        for case in selected:
            # Each task begins with the same immutable historical facts.
            # This prevents a prior cancellation from contaminating the
            # next address-change task.
            case_service.reset()
            rows.append(await run_case(client, case, case_service, olist_service, control_mode))
    summary = summarize(rows, mode=run_mode, control_mode=control_mode, suite=suite)
    result = {"summary": summary, "rows": rows}
    json_path = ROOT / "evaluation" / f"{output_prefix}_results.jsonl"
    md_path = ROOT / "evaluation" / f"{output_prefix}_report.md"
    json_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_render_report(summary, rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return result


def _render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AfterSalesBench Report",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    lines.extend(
        [
            "| case | passed | route | case status | event count | latency ms |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['passed']} | {row['route']} | "
            f"{row['case_status'] or '-'} | {row['event_count']} | {row['latency_ms']} |"
        )
    lines.append("")
    lines.append(
        "The run mode is part of the result. Offline workflow results are regression evidence only; "
        "live-model runs are required for LLM quality claims."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AfterSalesBench through the FastAPI product boundary.")
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case", dest="case_ids", action="append", default=[])
    parser.add_argument(
        "--control-mode",
        choices=("full", "handoff_all", "execute_on_intent"),
        default="full",
    )
    parser.add_argument("--suite", choices=("core", "llm_planner"), default="core")
    parser.add_argument("--output-prefix", default="after_sales_bench")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run(
            args.mode,
            args.limit,
            args.output_prefix,
            set(args.case_ids) or None,
            args.control_mode,
            args.suite,
        )
    )
