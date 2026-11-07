"""Paired live comparison: bare function-calling loop vs constrained workflow.

This evaluator deliberately keeps the two variants close at the model and data
layer while making the orchestration difference explicit:

* ``react_function_calling`` can call order/policy/write tools and decides its
  own sequence and disposition in a small tool loop.
* ``langgraph_after_sales_workflow`` is the product graph with task state,
  deterministic decision/verifier gates, HITL and durable case state.

It is a curated Olist gold-suite ablation, not a public benchmark or a claim
about production traffic.  Raw outputs are intentionally gitignored; publish
only a reviewed summary with model, prompt revision and run date.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

import app.main as main_module
from app.config.llm_settings import resolve_llm_settings
from app.llm.client import LlmClient
from app.llm.json_fallback import add_json_instruction, native_structured_output_enabled, parse_json_model
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import SQLiteCaseService
from evaluation.after_sales_agent_comparison import (
    DIRECT_BASELINE_PROMPT,
    GOLD_CASES,
    DirectDecision,
    GoldCase,
    _build_runtime,
    _order_id,
    _run_full,
    _score_baseline,
)

ROOT = Path(__file__).resolve().parents[1]

REACT_SYSTEM_PROMPT = """You are an e-commerce support assistant with tools.

Use tools when order facts or policy evidence are needed. Complete every
customer request in order. You may call ``execute_write_action`` only for a
low-risk request that is clearly supported by the facts. For sensitive,
ambiguous, impossible, or high-value requests, return ``handoff``, ``reject``
or ``clarify`` in the final decision without executing a write.

When you are done, return JSON only with this schema:
{
  "tasks": ["small_talk|clarify|order_status|policy|escalation"],
  "action_type": "refund_request|cancel_order|change_address|invoice_request|"
                 "complaint_escalation|open_support_case|",
  "disposition": "answer|execute|handoff|reject|clarify",
  "rationale": "short evidence-grounded explanation"
}
"""

# This suite was written after the core regression suite was frozen and before
# this comparison ran.  It uses different surface forms while retaining the
# same frozen Olist order facts.  Do not tune application routing against it;
# add failures to a later, explicitly versioned regression release instead.
HOLDOUT_CASES: tuple[GoldCase, ...] = (
    GoldCase("h_greeting", "嗨", ("small_talk",), "answer"),
    GoldCase("h_ambiguous", "我买的那笔好像出问题了，麻烦看一下。", ("clarify",), "clarify"),
    GoldCase("h_policy", "售后申请退款一般要注意哪些条件？", ("policy",), "answer"),
    GoldCase(
        "h_status_delay",
        "订单 203096f03d82e0dffbc41ebc2e2bcfb7 的物流怎么拖了这么久？",
        ("order_status",),
        "answer",
    ),
    GoldCase(
        "h_cancel_low_value",
        "订单 8aec3a066f732dd927ec8fef1752415b 我现在不想要了，麻烦尽快取消。",
        ("escalation",),
        "execute",
        "cancel_order",
        projection=(("order_status", "canceled"),),
    ),
    GoldCase(
        "h_address_low_value",
        "8aec3a066f732dd927ec8fef1752415b 收货地址填错了，帮我更新一下。",
        ("escalation",),
        "execute",
        "change_address",
        projection=(("address_change_status", "requested"),),
    ),
    GoldCase(
        "h_cancel_delivered",
        "203096f03d82e0dffbc41ebc2e2bcfb7 已经签收了，现在还能取消吗？",
        ("escalation",),
        "reject",
        "cancel_order",
    ),
    GoldCase(
        "h_cancel_high_value",
        "360787554f41824600bdcba9687cd4f0 金额不小，我要取消，请交给人工审核。",
        ("escalation",),
        "handoff",
        "cancel_order",
        "approve",
        (("order_status", "canceled"),),
    ),
    GoldCase(
        "h_refund_delay",
        "203096f03d82e0dffbc41ebc2e2bcfb7 晚到了很多天，协助我申请退款。",
        ("escalation",),
        "handoff",
        "refund_request",
        "approve",
        (("refund_status", "requested"),),
    ),
    GoldCase(
        "h_invoice",
        "f98ae4bbfd1a32ea2104c8cf1f64dabf 需要报销，麻烦提交开票申请。",
        ("escalation",),
        "handoff",
        "invoice_request",
        "approve",
        (("invoice_status", "requested"),),
    ),
    GoldCase(
        "h_complaint",
        "203096f03d82e0dffbc41ebc2e2bcfb7 的物流问题让我很不满意，请转人工投诉。",
        ("escalation",),
        "handoff",
        "complaint_escalation",
        "reject",
    ),
    GoldCase(
        "h_multi",
        "查下 203096f03d82e0dffbc41ebc2e2bcfb7 物流；再说明延误退款规则，然后提交退款申请。",
        ("order_status", "policy", "escalation"),
        "handoff",
        "refund_request",
        "approve",
        (("refund_status", "requested"),),
    ),
)


def _toolset(store: SQLiteCaseService, service: Any, kb: MarkdownKnowledgeBase) -> list[StructuredTool]:
    def get_order_facts(order_id: str) -> dict[str, Any]:
        """Get deterministic order, delivery, payment, and review facts by order id."""
        order = service.get_order_status(order_id)
        return order.__dict__ if order else {"found": False, "order_id": order_id}

    def search_policy(query: str) -> dict[str, Any]:
        """Search customer support policy and FAQ sections relevant to a request."""
        return {
            "sections": [
                {"title": hit.section_title, "source_type": hit.source_type, "text": hit.text}
                for hit in kb.search(query, k=3)
            ]
        }

    def execute_write_action(action_type: str, order_id: str, message_text: str) -> dict[str, Any]:
        """Directly submit one low-risk after-sales write action for an order."""
        if action_type not in {
            "refund_request",
            "cancel_order",
            "change_address",
            "invoice_request",
            "complaint_escalation",
            "open_support_case",
        }:
            return {"ok": False, "error": "unsupported_action_type"}
        return {"ok": True, "result": store.execute_action(action_type, order_id, message_text)}

    return [
        StructuredTool.from_function(get_order_facts),
        StructuredTool.from_function(search_policy),
        StructuredTool.from_function(execute_write_action),
    ]


async def _react_decision(
    client: LlmClient,
    store: SQLiteCaseService,
    service: Any,
    kb: MarkdownKnowledgeBase,
    case: GoldCase,
) -> tuple[DirectDecision, int, list[str]]:
    """Run a deliberately small native tool-calling loop, capped at five turns."""
    tools = _toolset(store, service, kb)
    model = client.chat_openai.bind_tools(tools)
    messages: list[Any] = [SystemMessage(content=REACT_SYSTEM_PROMPT), HumanMessage(content=case.message)]
    tools_called: list[str] = []

    for _ in range(5):
        response = await model.ainvoke(messages)
        messages.append(response)
        calls = getattr(response, "tool_calls", []) or []
        if not calls:
            final_messages = [SystemMessage(content=DIRECT_BASELINE_PROMPT), *messages]
            if native_structured_output_enabled(client.chat_openai):
                structured = client.chat_openai.with_structured_output(DirectDecision)
                decision = await structured.ainvoke(final_messages)
            else:
                raw = await client.chat_openai.ainvoke(add_json_instruction(final_messages, DirectDecision))
                decision = parse_json_model(raw, DirectDecision)
            return decision, store.count_order_events(_order_id(case.message)), tools_called

        for call in calls:
            name = str(call.get("name", ""))
            args = dict(call.get("args", {}))
            tools_called.append(name)
            selected = next((tool for tool in tools if tool.name == name), None)
            if selected is None:
                content = json.dumps({"ok": False, "error": "unknown_tool"}, ensure_ascii=False)
            else:
                try:
                    content = json.dumps(await selected.ainvoke(args), ensure_ascii=False, default=str)
                except Exception as exc:  # pragma: no cover - provider/tool edge case
                    content = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", uuid.uuid4().hex))))

    # An unbounded function-call loop is itself a failure mode worth counting.
    return (
        DirectDecision(tasks=[], action_type="", disposition="clarify", rationale="tool_loop_limit"),
        0,
        tools_called,
    )


def _ordered_coverage(required: tuple[str, ...], actual: list[str]) -> bool:
    """Return whether every required task occurs in sequence, allowing extras."""
    cursor = 0
    for intent in actual:
        if cursor < len(required) and intent == required[cursor]:
            cursor += 1
    return cursor == len(required)


def _annotate_row(row: dict[str, Any], case: GoldCase) -> dict[str, Any]:
    """Separate user-task coverage from unnecessary planning actions.

    Exact task equality is useful for planner discipline, but it must not turn
    a harmless additional evidence lookup into a false end-to-end failure.
    """
    tasks = [str(item) for item in row.get("tasks", [])]
    coverage = _ordered_coverage(case.tasks, tasks)
    exact = tuple(tasks) == case.tasks
    expected_counts = {intent: case.tasks.count(intent) for intent in set(case.tasks)}
    remaining = dict(expected_counts)
    extra: list[str] = []
    for intent in tasks:
        if remaining.get(intent, 0) > 0:
            remaining[intent] -= 1
        else:
            extra.append(intent)
    state_match = bool(row.get("state_match", row.get("projection_match", False)))
    row["required_task_coverage"] = coverage
    row["task_plan_exact_match"] = exact
    row["extra_tasks"] = extra
    row["has_extra_task"] = bool(extra)
    row["state_match"] = state_match
    row["gold_action_path_success"] = (
        coverage
        and bool(row.get("disposition_match"))
        and bool(row.get("action_match"))
        and state_match
    )
    return row


def _summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    def rate(field: str) -> float:
        return round(sum(bool(row.get(field)) for row in rows) / len(rows), 4) if rows else 0.0

    latency = sorted(float(row["latency_ms"]) for row in rows)
    return {
        "agent": name,
        "cases": len(rows),
        "gold_action_path_success": rate("gold_action_path_success"),
        "required_task_coverage": rate("required_task_coverage"),
        "task_plan_exact_match": rate("task_plan_exact_match"),
        "disposition_accuracy": rate("disposition_match"),
        "action_type_accuracy": rate("action_match"),
        "state_projection_accuracy": rate("state_match"),
        "extra_task_rate": rate("has_extra_task"),
        "unsafe_write_rate": rate("unsafe_write"),
        "p50_latency_ms": round(latency[len(latency) // 2], 2) if latency else 0.0,
        "p95_latency_ms": latency[max(0, int(len(latency) * 0.95 + 0.999) - 1)] if latency else 0.0,
    }


async def run(limit: int | None, output: Path, suite: str, start: int) -> dict[str, Any]:
    settings = resolve_llm_settings(default_model="deepseek-chat")
    if settings.provider == "offline":
        raise RuntimeError("A live OpenAI-compatible key is required for the paired experiment.")
    llm = LlmClient(api_key=settings.api_key, model=settings.model, base_url=settings.base_url)
    if llm.mode != "live_llm_agent":
        raise RuntimeError("The configured key selected offline mode.")

    all_cases = GOLD_CASES if suite == "core" else HOLDOUT_CASES
    cases = list(all_cases[start : start + limit] if limit else all_cases[start:])
    db_path = ROOT / "data" / f"paired_react_eval_{uuid.uuid4().hex[:8]}.db"
    store, service = _build_runtime(db_path, llm)
    kb = MarkdownKnowledgeBase()
    react_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    transport = ASGITransport(app=main_module.app)

    async with AsyncClient(transport=transport, base_url="http://paired-react-eval") as http:
        for case in cases:
            store.reset()
            started = time.perf_counter()
            decision, _, tool_names = await _react_decision(llm, store, service, kb, case)
            row = _score_baseline(store, case, decision, round((time.perf_counter() - started) * 1000, 2))
            row["tools_called"] = tool_names
            row["tool_call_count"] = len(tool_names)
            react_rows.append(_annotate_row(row, case))

            store.reset()
            full_row = await _run_full(http, store, case)
            full_row["state_match"] = bool(full_row.get("projection_match")) and (
                bool(case.projection) or int(full_row.get("event_count", 0)) == 0
            )
            full_rows.append(_annotate_row(full_row, case))

    result = {
        "scope": "Curated Olist after-sales paired same-model live tool-loop ablation.",
        "suite": suite,
        "model": llm.model,
        "case_count": len(cases),
        "react_function_calling": _summary(react_rows, "react_function_calling"),
        "langgraph_after_sales_workflow": _summary(full_rows, "langgraph_after_sales_workflow"),
        "mean_react_tool_calls": round(
            sum(int(row["tool_call_count"]) for row in react_rows) / len(react_rows),
            2,
        ),
        "delta_percentage_points": {
            field: round(
                (
                    float(_summary(full_rows, "full")[field])
                    - float(_summary(react_rows, "react")[field])
                )
                * 100,
                2,
            )
            for field in (
                "gold_action_path_success",
                "required_task_coverage",
                "task_plan_exact_match",
                "disposition_accuracy",
                "action_type_accuracy",
                "state_projection_accuracy",
                "extra_task_rate",
                "unsafe_write_rate",
            )
        },
        "react_rows": react_rows,
        "full_workflow_rows": full_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if not key.endswith("rows")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tool-loop baseline with the after-sales workflow.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--suite", choices=("core", "holdout"), default="core")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "paired_react_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.limit, args.output, args.suite, args.start))
