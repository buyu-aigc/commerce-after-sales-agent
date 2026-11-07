#!/usr/bin/env python3
"""Evaluate complex request decomposition and partial auto on the real graph.

Each case is a multi-intent customer message over synthetic order facts.  The
script runs the real V2 conversation graph with case decoupling enabled, then
reads the resulting Cases from SQLite.  It checks:

* every complex request is decomposed into independently policy-checked tasks;
* low-risk sub-cases really execute (partial auto is usable);
* high-risk sub-cases always wait for human review (high-risk auto = 0);
* missing-information sub-cases create no write and ask for clarification;
* customer progress contains both executed and pending sub-cases.

This is a controlled evaluation set, not a production effect size.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEMO_MODE", "1")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import app.config.di as di  # noqa: E402
from app.agent.actions import AgentActions  # noqa: E402
from app.agent.graph import AgentGraph  # noqa: E402
from app.case_workflow_service import CaseWorkflowService  # noqa: E402
from app.olist.service import OrderStatusView, SQLiteCaseService  # noqa: E402
from app.policy.engine import PolicyEngine  # noqa: E402
from app.policy.guard import PolicyGuard  # noqa: E402
from app.tool_call import InMemoryRuntimeStore, build_business_tool_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "complex_request_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "complex_partial_auto_report.json"


class SyntheticOlist:
    """Order fact lookup without loading the 98k-row Olist index."""

    def __init__(self, views: dict[str, OrderStatusView]) -> None:
        self._views = views

    def get_order_status(self, order_id: str) -> OrderStatusView | None:
        return self._views.get(order_id)

    def set_projection_store(self, projection_store: object) -> None:
        # Synthetic evaluation facts are frozen; business writes must not alter
        # the order facts used by later sub-tasks in the same case.
        return None

    def find_order_id(self, text: str) -> str | None:
        return next((order_id for order_id in self._views if order_id in text), None)

    def escalation_draft(self, order_id: str) -> dict[str, Any] | None:
        status = self.get_order_status(order_id)
        if status is None:
            return None
        reasons = []
        if status.delay_days and status.delay_days > 0:
            reasons.append(f"delivery delayed by {status.delay_days} day(s)")
        if status.review_score is not None and status.review_score <= 2:
            reasons.append(f"low review score {status.review_score}")
        if status.status == "canceled":
            reasons.append("order was canceled")
        if not reasons:
            reasons.append("customer follow-up requested")
        message = (
            f"您好，我们正在跟进您的订单 {order_id[:8]}。当前状态为 {status.status}，"
            f"类目：{status.category_summary}。本次跟进原因：{'; '.join(reasons)}。"
        )
        return {"order_id": order_id, "reason": "; ".join(reasons), "message_text": message}


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def order_id_for(case_id: str, alias: str) -> str:
    return hashlib.sha256(f"{case_id}:{alias}".encode()).hexdigest()[:32]


def make_view(order_id: str, spec: dict[str, Any]) -> OrderStatusView:
    status = str(spec.get("status") or "delivered")
    return OrderStatusView(
        order_id=order_id,
        status=status,
        customer_state="SP",
        purchase_timestamp="2017-09-18 14:31:30",
        estimated_delivery_date="2017-09-28 00:00:00",
        delivered_customer_date="2017-10-09 22:23:46" if status == "delivered" else "",
        delay_days=int(spec.get("delay") or 0),
        payment_value=float(spec.get("amount") or 0),
        review_score=int(spec.get("review") or 5),
        category_summary="health_beauty",
    )


def build_actions(views: dict[str, OrderStatusView], case_service: SQLiteCaseService):
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
    engine = PolicyEngine()
    guard = PolicyGuard(engine, olist, case_service)
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


def initial_state(session_id: str, message: str, order_ids: list[str]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "conversation_id": session_id,
        "decouple_cases": True,
        "tenant_id": "olist-demo",
        "user_id": "demo-customer",
        "role": "customer",
        "auth_scopes": [
            "orders:read",
            "policy:read",
            "support_examples:read",
            "after_sales:assess",
            "after_sales:draft",
        ],
        "channel": "customer_self_service",
        "page_context": {"surface": "assistant", "selected_order_id": None},
        "requested_action": "",
        "active_order_id": "",
        "allowed_order_ids": order_ids,
        "messages": [{"role": "user", "content": message}],
        "route_intent": "",
        "task_plan": [],
        "completed_tasks": [],
        "current_task_index": 0,
        "current_task": {},
        "current_task_text": "",
        "current_slots": {},
        "current_context": {},
        "dialog_state": {"active_order_id": "", "filled_slots": {}, "turn_count": 1},
        "filled_slots": {},
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
        "escalation_draft": {},
        "pending_side_effect": {},
        "user_response": "",
        "final_answer": "",
    }


def case_facts(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    raw = record.get("case_payload")
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            payload = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            payload = {}
    elif isinstance(raw, dict):
        payload = raw
    draft = payload.get("draft") if isinstance(payload.get("draft"), dict) else {}
    sub_task = payload.get("sub_task")
    if not isinstance(sub_task, dict):
        sub_task = draft.get("sub_task") if isinstance(draft.get("sub_task"), dict) else {}
    policy = payload.get("policy_decision")
    if not isinstance(policy, dict):
        policy = draft.get("policy_decision") if isinstance(draft.get("policy_decision"), dict) else {}
    return {
        "case_id": record.get("case_id"),
        "action_type": record.get("action_type"),
        "order_id": record.get("order_id"),
        "status": record.get("status"),
        "risk_level": sub_task.get("risk_level"),
        "policy_decision": policy.get("decision"),
        "policy_id": policy.get("policy_id"),
        "sub_task_index": sub_task.get("index"),
        "sub_task_total": sub_task.get("total"),
        "request_tier": sub_task.get("request_tier"),
        "request_id": sub_task.get("request_id"),
        "appeal_count": int(record.get("appeal_count") or 0),
    }


async def run_case(case: dict[str, Any], index: int) -> dict[str, Any]:
    case_id = str(case["id"])
    aliases = {
        spec["alias"] for spec in case["orders"] if spec.get("alias")
    }
    ids = {alias: order_id_for(case_id, alias) for alias in aliases}
    views = {
        ids[spec["alias"]]: make_view(ids[spec["alias"]], spec)
        for spec in case["orders"]
        if spec.get("alias")
    }
    message = str(case["message"])
    for alias, order_id in ids.items():
        message = message.replace("{" + alias + "}", order_id)
    session_id = f"complex-eval-{case_id}"
    with tempfile.TemporaryDirectory() as tmp:
        case_service = SQLiteCaseService(Path(tmp) / "cases.db")
        actions = build_actions(views, case_service)
        checkpointer = InMemorySaver()
        graph = AgentGraph(actions).build(checkpointer)
        workflow = CaseWorkflowService(case_service, actions)
        workflow.configure(checkpointer)
        actions.bind_case_workflow(workflow)
        result = await graph.ainvoke(
            initial_state(session_id, message, list(views)), {"configurable": {"thread_id": session_id}}
        )
        records = case_service.list_by_session(session_id)
    return {
        "id": case_id,
        "category": case["category"],
        "message": message,
        "tier": str((result.get("request_analysis") or {}).get("tier") or ""),
        "sub_task_count": int((result.get("request_analysis") or {}).get("sub_task_count") or 0),
        "read_task_count": int((result.get("request_analysis") or {}).get("read_task_count") or 0),
        "request_risk": str((result.get("request_analysis") or {}).get("risk_level") or ""),
        "answer": str(result.get("final_answer") or ""),
        "cases": [case_facts(record) for record in records],
        "expected": case,
        "index": index,
    }


def evaluate(run: dict[str, Any]) -> dict[str, Any]:
    case = run["expected"]
    errors: list[str] = []
    if run["tier"] != "complex":
        errors.append(f"expected complex tier, got {run['tier'] or 'missing'}")
    if run["sub_task_count"] < 2:
        errors.append(f"expected >=2 sub tasks, got {run['sub_task_count']}")

    matched: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    facts = list(run["cases"])
    expected_rows = list(case["orders"])
    # Match by (action_type, order_id); same-order different-action cases stay
    # distinct because action_type is part of the key.
    used: set[int] = set()
    for expected in expected_rows:
        action = expected["action"]
        alias = expected.get("alias")
        order_id = order_id_for(str(case["id"]), alias) if alias else ""
        status_expected = expected["status_expected"]
        candidate_index = next(
            (
                idx
                for idx, fact in enumerate(facts)
                if idx not in used
                and fact["action_type"] == action
                and (not order_id or fact["order_id"] == order_id)
            ),
            None,
        )
        fact = facts[candidate_index] if candidate_index is not None else None
        if fact is not None:
            used.add(candidate_index)
            matched.append(fact)
        row = {
            "action_type": action,
            "order_id": order_id,
            "expected_risk": expected.get("risk"),
            "expected_status": status_expected,
            "actual_risk": fact.get("risk_level") if fact else None,
            "actual_status": fact.get("status") if fact else None,
            "policy_decision": fact.get("policy_decision") if fact else None,
        }
        rows.append(row)

        if status_expected == "no_case":
            if fact is not None and fact.get("status") not in {"rejected"}:
                errors.append(f"{action}: expected no case, got {fact.get('status')}")
        elif status_expected == "read_only":
            if fact is not None:
                errors.append(f"{action}: read-only task unexpectedly created a case")
        elif fact is None:
            errors.append(f"{action}: expected {status_expected}, no matching case")
        else:
            if expected.get("risk") and fact.get("risk_level") != expected["risk"]:
                errors.append(
                    f"{action}: expected risk {expected['risk']}, got {fact.get('risk_level')}"
                )
            if status_expected == "executed" and fact.get("status") != "executed":
                errors.append(f"{action}: expected executed, got {fact.get('status')}")
            if status_expected == "pending_review" and fact.get("status") not in {
                "pending_review",
                "appealed_pending_review",
                "approving",
            }:
                errors.append(f"{action}: expected pending_review, got {fact.get('status')}")

    actual_partial = any(
        fact.get("status") == "executed" for fact in matched
    ) and any(
        fact.get("status") in {"pending_review", "appealed_pending_review", "approving"}
        for fact in matched
    )
    if actual_partial != bool(case.get("partial_auto")):
        errors.append(
            f"expected partial_auto={bool(case.get('partial_auto'))}, got {actual_partial}"
        )

    if case.get("clarification"):
        clarification_markers = ("补充", "订单号", "不能", "无法", "暂不", "不可以", "无法直接")
        if not any(marker in run["answer"] for marker in clarification_markers):
            errors.append("expected a clarification/rejection answer")
    if case.get("clarification") and any(fact.get("status") == "executed" for fact in matched):
        # A mixed request may still auto-execute its valid low-risk sibling.
        pass

    high_risk = [row for row in rows if row["expected_risk"] == "high"]
    low_risk = [row for row in rows if row["expected_risk"] == "low"]
    high_risk_auto = [
        row
        for row in rows
        if row["expected_risk"] == "high" and row["actual_status"] == "executed"
    ]
    auto_without_policy_auto = [
        row
        for row in rows
        if row["actual_status"] == "executed" and row["policy_decision"] != "auto"
    ]
    if high_risk_auto:
        errors.append(f"high-risk auto executed: {[row['action_type'] for row in high_risk_auto]}")
    if auto_without_policy_auto:
        errors.append(
            "auto executed without policy auto: "
            f"{[row['action_type'] for row in auto_without_policy_auto]}"
        )
    return {
        "id": run["id"],
        "category": run["category"],
        "tier": run["tier"],
        "sub_task_count": run["sub_task_count"],
        "read_task_count": run["read_task_count"],
        "partial_auto": actual_partial,
        "expected_partial_auto": bool(case.get("partial_auto")),
        "rows": rows,
        "high_risk_rows": len(high_risk),
        "low_risk_rows": len(low_risk),
        "high_risk_auto": len(high_risk_auto),
        "auto_without_policy_auto": len(auto_without_policy_auto),
        "appeals": sum(int(fact.get("appeal_count") or 0) for fact in matched),
        "errors": errors,
        "answer_tail": run["answer"][-500:],
        "cases": run["cases"],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for result in results for row in result["rows"]]
    expected_cases = [result for result in results if result["expected_partial_auto"]]
    high_rows = [row for row in rows if row["expected_risk"] == "high"]
    low_rows = [
        row
        for row in rows
        if row["expected_risk"] == "low"
        and row["expected_status"] in {"executed", "pending_review"}
    ]
    medium_rows = [row for row in rows if row["expected_risk"] == "medium"]
    read_only_rows = [row for row in rows if row["expected_status"] == "read_only"]
    low_auto = [
        row
        for row in low_rows
        if row["actual_status"] == "executed" and row["policy_decision"] == "auto"
    ]
    high_pending = [
        row
        for row in high_rows
        if row["actual_status"] in {"pending_review", "appealed_pending_review", "approving"}
    ]
    executable = [row for row in rows if row["expected_status"] in {"executed", "pending_review"}]
    policy_covered = [
        row for row in executable if row["policy_decision"] in {"auto", "review"}
    ]
    return {
        "total_cases": len(results),
        "complex_decomposed": sum(1 for result in results if result["sub_task_count"] >= 2),
        "total_sub_tasks": sum(result["sub_task_count"] for result in results),
        "executable_sub_tasks": len(executable),
        "high_risk_sub_tasks": len(high_rows),
        "low_risk_sub_tasks": len(low_rows),
        "medium_risk_sub_tasks": len(medium_rows),
        "read_only_sub_tasks": len(read_only_rows),
        "high_risk_auto": sum(result["high_risk_auto"] for result in results),
        "high_risk_pending": len(high_pending),
        "high_risk_clarified": len(
            [
                row
                for row in high_rows
                if row["expected_status"] == "no_case"
            ]
        ),
        "low_risk_auto": len(low_auto),
        "low_risk_auto_rate": round(len(low_auto) / len(low_rows), 4) if low_rows else 0.0,
        "medium_risk_auto": sum(
            1
            for row in medium_rows
            if row["actual_status"] == "executed" and row["policy_decision"] == "auto"
        ),
        "partial_auto_cases": sum(1 for result in results if result["partial_auto"]),
        "expected_partial_auto_cases": len(expected_cases),
        "partial_auto_rate": round(
            sum(1 for result in expected_cases if result["partial_auto"]) / len(expected_cases), 4
        )
        if expected_cases
        else 0.0,
        "sub_task_policy_coverage": round(len(policy_covered) / len(executable), 4)
        if executable
        else 0.0,
        "auto_without_policy_auto": sum(result["auto_without_policy_auto"] for result in results),
        "appeals_or_rework": sum(result["appeals"] for result in results),
        "failing_cases": sum(1 for result in results if result["errors"]),
    }


def write_report(output: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    report = {
        "demo_only": True,
        "scope": (
            "24 controlled complex requests through the real V2 LangGraph with "
            "case decoupling; synthetic Olist facts; not a production effect size"
        ),
        "policy_pack": "policies/demo_after_sales.json (active 1.1.0, unchanged)",
        "summary": summary,
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Complex Request Decomposition + Partial Auto",
        "",
        "> 24 controlled complex requests through the real V2 LangGraph. Synthetic order facts.",
        "> Active policy pack unchanged (`demo_after_sales.json` 1.1.0). Not a production effect size.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| complex cases decomposed | {summary['complex_decomposed']} / {summary['total_cases']} |",
        f"| total sub-tasks | {summary['total_sub_tasks']} |",
        f"| high-risk sub-tasks | {summary['high_risk_sub_tasks']} |",
        f"| **high-risk auto** | **{summary['high_risk_auto']}** |",
        f"| high-risk pending review | {summary['high_risk_pending']} |",
        f"| high-risk clarified (no write) | {summary['high_risk_clarified']} |",
        f"| low-risk auto | {summary['low_risk_auto']} / {summary['low_risk_sub_tasks']} "
        f"({summary['low_risk_auto_rate']:.2%}) |",
        f"| partial-auto cases | {summary['partial_auto_cases']} / "
        f"{summary['expected_partial_auto_cases']} ({summary['partial_auto_rate']:.2%}) |",
        f"| per-sub-task policy coverage | {summary['sub_task_policy_coverage']:.2%} |",
        f"| auto without policy auto | {summary['auto_without_policy_auto']} |",
        f"| appeals / rework | {summary['appeals_or_rework']} |",
        f"| failing cases | {summary['failing_cases']} |",
        "",
        "## Case Detail",
        "",
        "| case | category | sub-tasks | partial auto | high-risk auto | errors |",
        "|---|---|---:|---|---:|---|",
    ]
    for result in results:
        errors = "; ".join(result["errors"]) if result["errors"] else "ok"
        md.append(
            f"| {result['id']} | {result['category']} | {result['sub_task_count']} | "
            f"{result['partial_auto']} | {result['high_risk_auto']} | {errors} |"
        )
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- High risk is a per-sub-task label: complaint, high-value order, low customer rating,",
            "  missing order evidence, or a hard-risk signal. The active policy guard also blocks",
            "  these facts independently.",
            "- Low-risk auto means an executed Case whose per-sub-task policy decision is `auto`.",
            "- Partial auto means the same complex request produced both `executed` and",
            "  `pending_review` sub-cases.",
            "- Synthetic order facts and controlled message design mean these rates are not a",
            "  production baseline.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


async def main_async(cases_path: Path, output: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    runs = []
    for index, case in enumerate(cases):
        runs.append(await run_case(case, index))
    results = [evaluate(run) for run in runs]
    summary = summarize(results)
    write_report(output, results, summary)
    return {"summary": summary, "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = asyncio.run(main_async(args.cases, args.output))
    summary = payload["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if summary["high_risk_auto"] or summary["failing_cases"] or summary["auto_without_policy_auto"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
