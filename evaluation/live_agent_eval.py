from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.agent.state import AgentState
from app.config.llm_settings import resolve_llm_settings
from app.llm.client import LlmClient
from app.llm.guardrail import Guardrail
from app.llm.intent_planner import IntentPlanner
from app.llm.response_generator import OlistTaskExtractor, PolicyResponseGenerator, QaResponseGenerator
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import InMemoryCaseService, OlistService
from app.retrieval.hybrid import HybridSupportRetriever

ROOT = Path(__file__).resolve().parents[1]
OUT_JSONL = ROOT / "evaluation" / "live_agent_eval_results.jsonl"
OUT_MD = ROOT / "evaluation" / "live_agent_eval_report.md"


@dataclass(frozen=True)
class LiveCase:
    name: str
    message: str
    expected_tasks: list[str]
    expected_tools: list[str]
    requires_hitl: bool
    answer_keyword_groups: tuple[tuple[str, ...], ...]


CASES = [
    LiveCase(
        name="order_status",
        message="帮我查一下订单 203096f03d82e0dffbc41ebc2e2bcfb7 的状态",
        expected_tasks=["order_status"],
        expected_tools=["get_order_status"],
        requires_hitl=False,
        answer_keyword_groups=(("203096f03d82e0dffbc41ebc2e2bcfb7",), ("delivered", "已送达")),
    ),
    LiveCase(
        name="category_risk",
        message="health beauty 类目有什么运营风险？",
        expected_tasks=["qa"],
        expected_tools=["search_category_risk"],
        requires_hitl=False,
        answer_keyword_groups=(
            ("health_beauty", "健康美妆", "健康美容"),
            ("延迟", "delay"),
            ("评分", "评价", "review"),
        ),
    ),
    LiveCase(
        name="policy_boundary",
        message="退款补偿能不能直接承诺？",
        expected_tasks=["policy"],
        expected_tools=["search_policy_knowledge"],
        requires_hitl=False,
        answer_keyword_groups=(("退款",), ("补偿",), ("人工", "确认", "审核")),
    ),
    LiveCase(
        name="ops_decision",
        message="生成审核台优先处理队列，列出最需要人工跟进的类目和订单",
        expected_tasks=["ops_decision"],
        expected_tools=["generate_after_sales_priority_report"],
        requires_hitl=False,
        answer_keyword_groups=(("高风险类目",), ("优先跟进订单",), ("HITL", "人工确认")),
    ),
    LiveCase(
        name="multi_intent_hitl",
        message=(
            "查订单 203096f03d82e0dffbc41ebc2e2bcfb7 状态，并且说明退款政策，"
            "然后提交退款申请"
        ),
        expected_tasks=["order_status", "policy", "escalation"],
        expected_tools=["get_order_status", "search_policy_knowledge", "prepare_side_effect"],
        requires_hitl=True,
        answer_keyword_groups=(("订单查询",), ("政策问答",), ("售后升级",), ("是否确认执行",)),
    ),
    LiveCase(
        name="order_status_cn_now",
        message="订单 203096f03d82e0dffbc41ebc2e2bcfb7 现在送到了吗？",
        expected_tasks=["order_status"],
        expected_tools=["get_order_status"],
        requires_hitl=False,
        answer_keyword_groups=(("203096f03d82e0dffbc41ebc2e2bcfb7",), ("delivered", "已送达")),
    ),
    LiveCase(
        name="order_status_delivery_delay",
        message="查 203096f03d82e0dffbc41ebc2e2bcfb7 的物流和延迟天数",
        expected_tasks=["order_status"],
        expected_tools=["get_order_status"],
        requires_hitl=False,
        answer_keyword_groups=(("203096f03d82e0dffbc41ebc2e2bcfb7",), ("延迟", "delay")),
    ),
    LiveCase(
        name="order_status_review",
        message="帮客服看下订单 203096f03d82e0dffbc41ebc2e2bcfb7 的支付金额和评价分",
        expected_tasks=["order_status"],
        expected_tools=["get_order_status"],
        requires_hitl=False,
        answer_keyword_groups=(("支付",), ("评价", "评分")),
    ),
    LiveCase(
        name="order_status_mixed_en",
        message="track order 203096f03d82e0dffbc41ebc2e2bcfb7 and show delivery status",
        expected_tasks=["order_status"],
        expected_tools=["get_order_status"],
        requires_hitl=False,
        answer_keyword_groups=(("203096f03d82e0dffbc41ebc2e2bcfb7",), ("delivered", "已送达")),
    ),
    LiveCase(
        name="order_status_compact_request",
        message="订单203096f03d82e0dffbc41ebc2e2bcfb7状态",
        expected_tasks=["order_status"],
        expected_tools=["get_order_status"],
        requires_hitl=False,
        answer_keyword_groups=(("203096f03d82e0dffbc41ebc2e2bcfb7",), ("状态", "delivered")),
    ),
    LiveCase(
        name="category_risk_space_alias",
        message="health beauty 的延迟率和低分率怎么样？",
        expected_tasks=["qa"],
        expected_tools=["search_category_risk"],
        requires_hitl=False,
        answer_keyword_groups=(("health_beauty", "健康美妆"), ("延迟",), ("低分", "评分", "评价")),
    ),
    LiveCase(
        name="category_risk_hyphen_alias",
        message="health-beauty category risk summary",
        expected_tasks=["qa"],
        expected_tools=["search_category_risk"],
        requires_hitl=False,
        answer_keyword_groups=(("health_beauty", "health-beauty", "health beauty"), ("risk", "风险")),
    ),
    LiveCase(
        name="category_risk_bed_bath",
        message="bed bath table 类目的售后风险高吗？",
        expected_tasks=["qa"],
        expected_tools=["search_category_risk"],
        requires_hitl=False,
        answer_keyword_groups=(("bed_bath_table", "bed bath table"), ("风险", "延迟", "低分")),
    ),
    LiveCase(
        name="category_risk_furniture",
        message="furniture decor 类目的取消率和评价风险",
        expected_tasks=["qa"],
        expected_tools=["search_category_risk"],
        requires_hitl=False,
        answer_keyword_groups=(("furniture_decor", "furniture decor"), ("取消",), ("评价", "低分")),
    ),
    LiveCase(
        name="category_risk_operations",
        message="从运营角度看 healthbeauty 类目最该关注什么？",
        expected_tasks=["qa"],
        expected_tools=["search_category_risk"],
        requires_hitl=False,
        answer_keyword_groups=(("health_beauty", "healthbeauty", "健康美妆"), ("关注", "建议", "风险")),
    ),
    LiveCase(
        name="policy_refund_direct_promise",
        message="客服能不能直接答应给客户退款？",
        expected_tasks=["policy"],
        expected_tools=["search_policy_knowledge"],
        requires_hitl=False,
        answer_keyword_groups=(("退款",), ("不能", "不可", "不可以"), ("人工", "确认", "审核")),
    ),
    LiveCase(
        name="policy_coupon_boundary",
        message="延迟配送时可以直接发优惠券补偿吗？",
        expected_tasks=["policy"],
        expected_tools=["search_policy_knowledge"],
        requires_hitl=False,
        answer_keyword_groups=(("优惠券", "补偿"), ("确认", "审核", "人工")),
    ),
    LiveCase(
        name="policy_invoice",
        message="发票申请需要哪些信息？",
        expected_tasks=["policy"],
        expected_tools=["search_policy_knowledge"],
        requires_hitl=False,
        answer_keyword_groups=(("发票",), ("信息", "订单")),
    ),
    LiveCase(
        name="policy_cancel_order",
        message="取消订单有什么规则和限制？",
        expected_tasks=["policy"],
        expected_tools=["search_policy_knowledge"],
        requires_hitl=False,
        answer_keyword_groups=(("取消",), ("规则", "限制", "政策")),
    ),
    LiveCase(
        name="policy_address_change",
        message="客户要修改收货地址，客服应该怎么处理？",
        expected_tasks=["policy"],
        expected_tools=["search_policy_knowledge"],
        requires_hitl=False,
        answer_keyword_groups=(("地址",), ("确认", "订单", "处理")),
    ),
    LiveCase(
        name="ops_decision_daily_report",
        message="输出审核台今天的优先处理队列",
        expected_tasks=["ops_decision"],
        expected_tools=["generate_after_sales_priority_report"],
        requires_hitl=False,
        answer_keyword_groups=(("审核台优先处理", "优先处理"), ("高风险类目",), ("优先跟进订单",)),
    ),
    LiveCase(
        name="ops_decision_priority_orders",
        message="帮售后主管列一下高风险订单优先级",
        expected_tasks=["ops_decision"],
        expected_tools=["generate_after_sales_priority_report"],
        requires_hitl=False,
        answer_keyword_groups=(("优先",), ("订单",), ("HITL", "人工确认")),
    ),
    LiveCase(
        name="ops_decision_category_actions",
        message="哪些类目需要重点跟进，并给出行动建议",
        expected_tasks=["ops_decision"],
        expected_tools=["generate_after_sales_priority_report"],
        requires_hitl=False,
        answer_keyword_groups=(("类目",), ("行动", "建议"), ("高风险", "重点")),
    ),
    LiveCase(
        name="ops_decision_after_sales_queue",
        message="生成售后优先处理队列",
        expected_tasks=["ops_decision"],
        expected_tools=["generate_after_sales_priority_report"],
        requires_hitl=False,
        answer_keyword_groups=(("优先",), ("订单",), ("售后",)),
    ),
    LiveCase(
        name="ops_decision_risk_review",
        message="给我一份客服主管看的风险复盘",
        expected_tasks=["ops_decision"],
        expected_tools=["generate_after_sales_priority_report"],
        requires_hitl=False,
        answer_keyword_groups=(("风险",), ("类目",), ("订单",)),
    ),
    LiveCase(
        name="multi_order_policy_refund",
        message="查订单 203096f03d82e0dffbc41ebc2e2bcfb7，并说明退款政策，再申请退款",
        expected_tasks=["order_status", "policy", "escalation"],
        expected_tools=["get_order_status", "search_policy_knowledge", "prepare_side_effect"],
        requires_hitl=True,
        answer_keyword_groups=(("订单查询",), ("政策问答",), ("售后升级",), ("是否确认执行",)),
    ),
    LiveCase(
        name="multi_policy_invoice",
        message="先说明发票政策，然后为订单 203096f03d82e0dffbc41ebc2e2bcfb7 申请发票",
        expected_tasks=["policy", "escalation"],
        expected_tools=["search_policy_knowledge", "prepare_side_effect"],
        requires_hitl=True,
        answer_keyword_groups=(("政策问答",), ("售后升级",), ("发票",), ("是否确认执行",)),
    ),
    LiveCase(
        name="multi_status_cancel",
        message="查订单 203096f03d82e0dffbc41ebc2e2bcfb7 状态，然后申请取消订单",
        expected_tasks=["order_status", "escalation"],
        expected_tools=["get_order_status", "prepare_side_effect"],
        requires_hitl=True,
        answer_keyword_groups=(("订单查询",), ("售后升级",), ("取消",), ("是否确认执行",)),
    ),
    LiveCase(
        name="multi_ops_policy",
        message="生成审核台优先处理队列，同时说明退款补偿边界",
        expected_tasks=["ops_decision", "policy"],
        expected_tools=["generate_after_sales_priority_report", "search_policy_knowledge"],
        requires_hitl=False,
        answer_keyword_groups=(("审核台优先处理",), ("政策问答",), ("补偿", "退款")),
    ),
    LiveCase(
        name="multi_qa_escalation",
        message="分析 health beauty 风险，并为订单 203096f03d82e0dffbc41ebc2e2bcfb7 生成升级话术",
        expected_tasks=["qa", "escalation"],
        expected_tools=["search_category_risk", "prepare_side_effect"],
        requires_hitl=True,
        answer_keyword_groups=(("运营分析",), ("售后升级",), ("是否确认执行",)),
    ),
]


async def main() -> None:
    load_dotenv(ROOT / ".env")
    limit = int(os.environ.get("LIVE_AGENT_EVAL_LIMIT", "30"))
    offset = int(os.environ.get("LIVE_AGENT_EVAL_OFFSET", "0"))
    cases = CASES[offset : offset + limit]
    settings = resolve_llm_settings(default_model="deepseek-v4-flash")
    if settings.provider == "offline":
        raise SystemExit(
            "OPENAI_API_KEY or AIHUBMIX_API_KEY is not set. Set one in the current "
            "shell before running live Agent evaluation."
        )
    llm_client = LlmClient(
        api_key=settings.api_key,
        model=settings.model,
        base_url=settings.base_url or "https://api.deepseek.com",
    )
    graph = _build_graph(llm_client)
    guardrail = Guardrail(llm_client.chat_openai)

    rows = []
    for case in cases:
        t0 = time.perf_counter()
        result = await _run_case(graph, guardrail, case)
        latency_ms = (time.perf_counter() - t0) * 1000
        row = _score_case(case, result, latency_ms, llm_client)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    OUT_JSONL.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(_render_report(rows, llm_client), encoding="utf-8")
    print("")
    print(_summary_line(rows))
    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_MD}")


def _build_graph(llm_client: LlmClient):
    actions = AgentActions(
        intent_planner=IntentPlanner(llm_client.chat_openai),
        qa_generator=QaResponseGenerator(llm_client.chat_openai),
        policy_generator=PolicyResponseGenerator(llm_client.chat_openai),
        task_extractor=OlistTaskExtractor(llm_client.chat_openai),
        olist_service=OlistService(),
        knowledge_base=MarkdownKnowledgeBase(),
        support_retriever=HybridSupportRetriever(),
        case_service=InMemoryCaseService(),
    )
    return AgentGraph(actions).build(InMemorySaver())


async def _run_case(graph, guardrail: Guardrail, case: LiveCase) -> dict:
    input_check = await guardrail.check_input(case.message)
    if not input_check.on_topic:
        return {
            "route_intent": "input_guard",
            "task_plan": [],
            "completed_tasks": [],
            "trajectory_events": [],
            "final_answer": f"Rejected by guard: {input_check.reason}",
            "output_valid": False,
        }
    session_id = f"live-agent-eval-{case.name}-{uuid.uuid4().hex[:8]}"
    state: AgentState = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": case.message}],
        "route_intent": "",
        "task_plan": [],
        "completed_tasks": [],
        "trajectory_events": [],
        "artifacts": {},
        "retrieved_insights": [],
        "retrieved_policy": [],
        "retrieved_support_docs": [],
        "final_answer": "",
    }
    result = await graph.ainvoke(state, {"configurable": {"thread_id": session_id}})
    output_check = await guardrail.check_output(str(result.get("final_answer", "")))
    return {**result, "output_valid": output_check.valid, "output_guard_reason": output_check.reason}


def _score_case(case: LiveCase, result: dict, latency_ms: float, llm_client: LlmClient) -> dict:
    tasks = [str(task.get("intent")) for task in result.get("task_plan", [])]
    tools = [str(event.get("details", {}).get("tool", "")) for event in result.get("trajectory_events", [])]
    statuses = [str(event.get("status")) for event in result.get("trajectory_events", [])]
    answer = str(result.get("final_answer", ""))
    checks = {
        "task_exact": tasks == case.expected_tasks,
        "tools_used": set(case.expected_tools).issubset(set(tools)),
        "hitl_correct": bool(result.get("pending_side_effect", {}).get("requires_confirmation"))
        == case.requires_hitl,
        "no_failed_event": "failed" not in statuses and "blocked" not in statuses,
        "output_valid": bool(result.get("output_valid")),
        "answer_keywords": all(
            any(keyword.lower() in answer.lower() for keyword in group)
            for group in case.answer_keyword_groups
        ),
    }
    return {
        "case": case.name,
        "provider_mode": llm_client.mode,
        "model": llm_client.model,
        "prompt_version": os.environ.get("PROMPT_VERSION", "local-prompts-v1"),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "latency_ms": round(latency_ms, 2),
        "input_chars": len(case.message),
        "answer_chars": len(answer),
        "approx_turn_tokens": _approx_tokens(case.message + "\n" + answer),
        "expected_tasks": case.expected_tasks,
        "actual_tasks": tasks,
        "expected_tools": case.expected_tools,
        "actual_tools": [tool for tool in tools if tool],
        "requires_hitl": case.requires_hitl,
        "answer_preview": answer.replace("\n", " ")[:260],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _render_report(rows: list[dict], llm_client: LlmClient) -> str:
    lines = [
        "# Live Agent Eval Report",
        "",
        (
            "This report is generated by `python -m evaluation.live_agent_eval` "
            "with a real LLM in the Agent loop."
        ),
        f"mode: `{llm_client.mode}`; model: `{llm_client.model}`; cases: `{len(rows)}`.",
        "",
        "| case | passed | tasks | tools | hitl | latency_ms | failed_checks |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        failed = [name for name, ok in row["checks"].items() if not ok]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case"],
                    str(row["passed"]),
                    ",".join(row["actual_tasks"]),
                    ",".join(row["actual_tools"]),
                    str(row["requires_hitl"]),
                    str(row["latency_ms"]),
                    ",".join(failed) or "-",
                ]
            )
            + " |"
        )
    lines.extend(["", _summary_line(rows)])
    return "\n".join(lines)


def _summary_line(rows: list[dict]) -> str:
    if not rows:
        return "No live eval rows."
    check_names = list(rows[0]["checks"])
    parts = [f"case_pass_rate={sum(row['passed'] for row in rows) / len(rows):.2%}"]
    for name in check_names:
        parts.append(f"{name}={sum(row['checks'][name] for row in rows) / len(rows):.2%}")
    return "; ".join(parts)


def _approx_tokens(text: str) -> int:
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return round(ascii_chars / 4 + non_ascii_chars * 1.5)


if __name__ == "__main__":
    asyncio.run(main())
