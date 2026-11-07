#!/usr/bin/env python3
"""Run 20 synthetic orders through the real V2 LangGraph, with/without policy_check."""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

import app.config.di as di
from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.olist.service import OlistService, OrderStatusView, SQLiteCaseService
from app.policy.engine import PolicyEngine
from app.policy.guard import PolicyGuard
from app.tool_call import InMemoryRuntimeStore, build_business_tool_manager

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "policy_workflow_graph_report.json"


class SyntheticOlist(OlistService):
    def __init__(self, views: dict[str, OrderStatusView]) -> None:
        super().__init__()
        self._views = views

    def get_order_status(self, order_id: str) -> OrderStatusView | None:
        return self._views.get(order_id)


def make_orders() -> list[dict[str, Any]]:
    specs = []
    # 6 eligible auto cases
    for index in range(6):
        specs.append({"kind": "auto", "amount": 43.0 + index, "delay": 11, "status": "delivered"})
    # 6 high-value cases
    for index in range(6):
        specs.append({"kind": "review", "amount": 118.86 + index, "delay": 11, "status": "delivered"})
    # 4 no-delay cases
    for index in range(4):
        specs.append({"kind": "review", "amount": 43.0 + index, "delay": 0, "status": "delivered"})
    # 2 invoiced refund cases
    for index in range(2):
        specs.append({"kind": "review", "amount": 32.0 + index, "delay": 0, "status": "invoiced"})
    # 2 canceled cases
    for index in range(2):
        specs.append({"kind": "review", "amount": 30.0 + index, "delay": 0, "status": "canceled"})
    return specs


def make_view(order_id: str, spec: dict[str, Any]) -> OrderStatusView:
    return OrderStatusView(
        order_id=order_id,
        status=spec["status"],
        customer_state="SP",
        purchase_timestamp="2017-09-18 14:31:30",
        estimated_delivery_date="2017-09-28 00:00:00",
        delivered_customer_date="2017-10-09 22:23:46" if spec["status"] == "delivered" else "",
        delay_days=spec["delay"] or None,
        payment_value=spec["amount"],
        review_score=2,
        category_summary="health_beauty",
    )


def make_state(
    session_id: str,
    order_id: str,
    allowed: list[str],
    *,
    risk_level: str = "",
    repeat_claim: bool = False,
    refund_failure: bool = False,
) -> dict[str, Any]:
    message = f"订单 {order_id} 晚到了很多天，我要申请退款"
    return {
        "session_id": session_id,
        "conversation_id": session_id,
        "decouple_cases": False,
        "tenant_id": "olist-demo",
        "user_id": "demo-customer",
        "role": "customer",
        "auth_scopes": ["orders:read", "after_sales:assess", "after_sales:draft"],
        "channel": "customer_self_service",
        "page_context": {"surface": "assistant", "selected_order_id": order_id},
        "requested_action": "refund_request",
        "active_order_id": order_id,
        "allowed_order_ids": allowed,
        "messages": [{"role": "user", "content": message}],
        "route_intent": "",
        "task_plan": [],
        "completed_tasks": [],
        "current_task_index": 0,
        "current_task": {},
        "current_task_text": "",
        "current_slots": {},
        "current_context": {},
        "dialog_state": {"active_order_id": order_id, "filled_slots": {"order_id": order_id}, "turn_count": 1},
        "filled_slots": {"order_id": order_id},
        "missing_slots": [],
        "handoff_reasons": [],
        "turn_count": 1,
        "failure_count": 0,
        "answer_parts": [],
        "workflow_complete": False,
        "trajectory_events": [],
        "artifacts": {},
        "retrieved_insights": [],
        "retrieved_policy": [],
        "retrieved_support_docs": [],
        "after_sales_cases": [],
        "final_answer": "",
        "risk_level": risk_level,
        "repeat_claim": repeat_claim,
        "refund_failure": refund_failure,
    }


def build_actions(views: dict[str, OrderStatusView], case_service: SQLiteCaseService, use_policy: bool):
    store = InMemoryRuntimeStore()
    olist = SyntheticOlist(views)
    manager = build_business_tool_manager(
        olist_service=olist,
        knowledge_base=di.knowledge_base,
        support_retriever=di.support_retriever,
        case_service=case_service,
        cache_backend=store,
        runtime_store=store,
    )
    engine = PolicyEngine() if use_policy else None
    guard = PolicyGuard(engine, olist, case_service) if engine else None
    actions = AgentActions(
        intent_planner=di.intent_planner,
        qa_generator=di.qa_generator,
        policy_generator=di.policy_generator,
        task_extractor=di.task_extractor,
        olist_service=olist,
        knowledge_base=di.knowledge_base,
        support_retriever=di.support_retriever,
        case_service=case_service,
        tool_manager=manager,
        runtime_store=store,
        policy_engine=engine,
        policy_guard=guard,
    )
    return actions


async def run_scenario(use_policy: bool, specs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        case_service = SQLiteCaseService(Path(tmp) / "cases.db")
        views = {}
        ids = []
        for index, spec in enumerate(specs):
            order_id = f"{index + 1:032x}"
            ids.append(order_id)
            views[order_id] = make_view(order_id, spec)
        actions = build_actions(views, case_service, use_policy)
        graph = AgentGraph(actions).build(InMemorySaver())
        for index, spec in enumerate(specs):
            order_id = ids[index]
            state = make_state(f"policy-graph-{use_policy}-{index}", order_id, ids)
            result = await graph.ainvoke(state, {"configurable": {"thread_id": state["session_id"]}})
            pending = result.get("pending_side_effect") or {}
            auto = not bool(pending.get("requires_confirmation"))
            rows.append(
                {
                    "index": index,
                    "expected_group": spec["kind"],
                    "auto": auto,
                    "status": result.get("policy_decision", {}).get("decision") if isinstance(result.get("policy_decision"), dict) else None,
                    "case_id": result.get("case_id"),
                    "final_answer": str(result.get("final_answer", ""))[:120],
                }
            )
    auto_count = sum(row["auto"] for row in rows)
    return {
        "use_policy": use_policy,
        "total": len(rows),
        "auto": auto_count,
        "review": len(rows) - auto_count,
        "auto_rate": round(auto_count / len(rows), 4) if rows else 0,
        "review_rate": round((len(rows) - auto_count) / len(rows), 4) if rows else 0,
        "rows": rows,
    }


async def main_async(output: Path) -> dict[str, Any]:
    specs = make_orders()
    without = await run_scenario(False, specs)
    with_policy = await run_scenario(True, specs)
    report = {
        "demo_only": True,
        "scope": "20 synthetic orders through the real LangGraph; not live conversations.",
        "total": len(specs),
        "without_policy": without,
        "with_policy": with_policy,
        "review_reduction_pp": round((without["review_rate"] - with_policy["review_rate"]) * 100, 2),
        "false_auto": sum(
            1
            for spec, row in zip(specs, with_policy["rows"])
            if spec["kind"] == "review" and row["auto"]
        ),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Policy Workflow Graph Bench",
        "",
        "> 20 synthetic orders through the real V2 LangGraph, with/without policy_check.",
        f"- without policy: auto {without['auto']} / review {without['review']} ({without['review_rate']:.2%} review)",
        f"- with policy: auto {with_policy['auto']} / review {with_policy['review']} ({with_policy['review_rate']:.2%} review)",
        f"- review reduction: {report['review_reduction_pp']:.2f} pp",
        f"- false auto: {report['false_auto']}",
        "",
        "| # | expected | auto | policy decision |",
        "|---|---|---|---|",
    ]
    for row in with_policy["rows"]:
        md.append(f"| {row['index']} | {row['expected_group']} | {row['auto']} | {row['status']} |")
    output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = asyncio.run(main_async(args.output))
    print(
        json.dumps(
            {
                "without_review": report["without_policy"]["review_rate"],
                "with_review": report["with_policy"]["review_rate"],
                "reduction_pp": report["review_reduction_pp"],
                "false_auto": report["false_auto"],
            },
            ensure_ascii=False,
        )
    )
    if report["false_auto"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
