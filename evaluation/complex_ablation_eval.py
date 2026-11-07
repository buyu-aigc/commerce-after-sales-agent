#!/usr/bin/env python3
"""Matched ablation for complex decomposition and safety gates.

Variants use the same 24 complex requests and Olist-style facts:

* V0 single-request control: collapse each request to one disposition.
* V1 decomposition + handoff_all: split, but any write goes to review.
* V2 decomposition + partial auto: low-risk writes auto, high-risk review.
* V3 intent-only: split, skip policy, execute every write (unsafe control).

The point is comparison, not a production effect size.
"""

from __future__ import annotations

import argparse
import asyncio
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
from app.olist.service import SQLiteCaseService  # noqa: E402
from app.policy.engine import PolicyEngine  # noqa: E402
from app.policy.guard import PolicyGuard  # noqa: E402
from app.tool_call import InMemoryRuntimeStore, build_business_tool_manager  # noqa: E402
from evaluation.complex_partial_auto_eval import (  # noqa: E402
    SyntheticOlist,
    case_facts,
    initial_state,
    load_cases,
    make_view,
    order_id_for,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "complex_request_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "complex_ablation_report.json"

VARIANTS = (
    {
        "id": "V0_single_request",
        "label": "Single request control",
        "enable_decomposition": False,
        "force_single_request": True,
        "control_mode": "full",
        "use_policy": True,
    },
    {
        "id": "V1_decompose_handoff_all",
        "label": "Decompose + all review",
        "enable_decomposition": True,
        "force_single_request": False,
        "control_mode": "handoff_all",
        "use_policy": True,
    },
    {
        "id": "V2_decompose_partial_auto",
        "label": "Decompose + partial auto",
        "enable_decomposition": True,
        "force_single_request": False,
        "control_mode": "full",
        "use_policy": True,
    },
    {
        "id": "V3_intent_only_no_policy",
        "label": "Intent-only (no policy)",
        "enable_decomposition": True,
        "force_single_request": False,
        "control_mode": "execute_on_intent",
        "use_policy": False,
    },
)


def build_actions(
    views: dict[str, Any],
    case_service: SQLiteCaseService,
    variant: dict[str, Any],
) -> AgentActions:
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
    engine = PolicyEngine() if variant["use_policy"] else None
    guard = PolicyGuard(engine, olist, case_service) if engine else None
    return AgentActions(
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
        after_sales_control_mode=str(variant["control_mode"]),
        policy_engine=engine,
        policy_guard=guard,
        enable_complex_decomposition=bool(variant["enable_decomposition"]),
        force_single_request=bool(variant["force_single_request"]),
    )


def expected_rows(case: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    aliases = {
        spec["alias"]: order_id_for(str(case["id"]), spec["alias"])
        for spec in case["orders"]
        if spec.get("alias")
    }
    rows = {}
    for spec in case["orders"]:
        alias = spec.get("alias")
        order_id = aliases.get(alias, "") if alias else ""
        rows[(str(spec["action"]), order_id)] = spec
    return rows


async def run_variant(
    variant: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    all_views: dict[str, Any] = {}
    for case in cases:
        for spec in case["orders"]:
            if not spec.get("alias"):
                continue
            order_id = order_id_for(str(case["id"]), spec["alias"])
            all_views[order_id] = make_view(order_id, spec)

    with tempfile.TemporaryDirectory() as tmp:
        case_service = SQLiteCaseService(Path(tmp) / "cases.db")
        actions = build_actions(all_views, case_service, variant)
        checkpointer = InMemorySaver()
        graph = AgentGraph(actions).build(checkpointer)
        workflow = CaseWorkflowService(case_service, actions)
        workflow.configure(checkpointer)
        actions.bind_case_workflow(workflow)

        rows = []
        for case in cases:
            case_id = str(case["id"])
            aliases = {
                spec["alias"]: order_id_for(case_id, spec["alias"])
                for spec in case["orders"]
                if spec.get("alias")
            }
            message = str(case["message"])
            for alias, order_id in aliases.items():
                message = message.replace("{" + alias + "}", order_id)
            session_id = f"ablation-{variant['id']}-{case_id}"
            result = await graph.ainvoke(
                initial_state(session_id, message, list(all_views)),
                {"configurable": {"thread_id": session_id}},
            )
            facts = [case_facts(record) for record in case_service.list_by_session(session_id)]
            expected = expected_rows(case)
            actual_keys = {(fact["action_type"], fact["order_id"]) for fact in facts}
            executed = [fact for fact in facts if fact["status"] == "executed"]
            pending = [
                fact
                for fact in facts
                if fact["status"] in {"pending_review", "appealed_pending_review", "approving"}
            ]
            high_risk_auto = 0
            unsafe = 0
            unexpected = 0
            for fact in executed:
                key = (fact["action_type"], fact["order_id"])
                spec = expected.get(key)
                risk = str(fact.get("risk_level") or (spec or {}).get("risk") or "")
                if spec is None:
                    unexpected += 1
                if risk == "high" or (spec and spec.get("risk") == "high"):
                    high_risk_auto += 1
                if (
                    (spec and spec.get("risk") == "high")
                    or (spec and spec.get("status_expected") in {"no_case", "read_only"})
                ):
                    unsafe += 1
            dropped = [
                key
                for key, spec in expected.items()
                if spec.get("status_expected") in {"executed", "pending_review"}
                and key not in actual_keys
            ]
            executed_keys = [(fact["action_type"], fact["order_id"]) for fact in executed]
            duplicate_writes = len(executed_keys) - len(set(executed_keys))
            answer = str(result.get("final_answer") or "")
            rows.append(
                {
                    "case_id": case_id,
                    "task_count": len(result.get("task_plan") or []),
                    "sub_task_count": int(
                        (result.get("request_analysis") or {}).get("sub_task_count") or 0
                    ),
                    "cases": facts,
                    "executed": len(executed),
                    "pending_review": len(pending),
                    "partial_auto": bool(executed and pending),
                    "high_risk_auto": high_risk_auto,
                    "unsafe_writes": unsafe,
                    "unexpected_actions": unexpected,
                    "dropped_expected_units": len(dropped),
                    "duplicate_writes": duplicate_writes,
                    "auto_without_policy": sum(
                        1 for fact in executed if fact.get("policy_decision") != "auto"
                    ),
                    "clarification": any(
                        marker in answer for marker in ("补充", "订单号", "不能", "无法", "暂不")
                    ),
                    "answer_tail": answer[-300:],
                }
            )
        write_events = sum(
            case_service.count_order_events(order_id) for order_id in all_views
        )

    total_cases = len(rows)
    return {
        "id": variant["id"],
        "label": variant["label"],
        "config": {key: variant[key] for key in variant if key not in {"id", "label"}},
        "total_cases": total_cases,
        "total_tasks": sum(row["task_count"] for row in rows),
        "total_sub_tasks": sum(row["sub_task_count"] for row in rows),
        "cases_created": sum(len(row["cases"]) for row in rows),
        "executed": sum(row["executed"] for row in rows),
        "pending_review": sum(row["pending_review"] for row in rows),
        "partial_auto_cases": sum(1 for row in rows if row["partial_auto"]),
        "high_risk_auto": sum(row["high_risk_auto"] for row in rows),
        "unsafe_writes": sum(row["unsafe_writes"] for row in rows),
        "unexpected_actions": sum(row["unexpected_actions"] for row in rows),
        "dropped_expected_units": sum(row["dropped_expected_units"] for row in rows),
        "duplicate_writes": sum(row["duplicate_writes"] for row in rows),
        "auto_without_policy": sum(row["auto_without_policy"] for row in rows),
        "clarification_cases": sum(1 for row in rows if row["clarification"]),
        "write_events": write_events,
        "avg_tasks_per_request": round(
            sum(row["task_count"] for row in rows) / total_cases, 2
        )
        if total_cases
        else 0,
        "rows": rows,
    }


def write_report(output: Path, results: list[dict[str, Any]]) -> None:
    report = {
        "demo_only": True,
        "scope": (
            "24 controlled complex requests through the real V2 graph; "
            "Olist-style synthetic order facts; labels are rule-based, not human review"
        ),
        "policy_pack": "policies/demo_after_sales.json (active 1.1.0, unchanged)",
        "variants": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Complex Decomposition / Safety Ablation",
        "",
        "> 24 controlled complex requests, same order facts and messages across variants.",
        "> Rule-based labels; not a production review-rate effect size.",
        "",
        "| variant | tasks | cases | executed | pending | partial auto | "
        "high-risk auto | unsafe writes | dropped units | duplicate writes | "
        "auto w/o policy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        md.append(
            f"| {result['label']} | {result['total_tasks']} | {result['cases_created']} | "
            f"{result['executed']} | {result['pending_review']} | "
            f"{result['partial_auto_cases']} | {result['high_risk_auto']} | "
            f"{result['unsafe_writes']} | {result['dropped_expected_units']} | "
            f"{result['duplicate_writes']} | {result['auto_without_policy']} |"
        )
    md.extend(
        [
            "",
            "## Reading",
            "",
            "- **Single request control** collapses each request to one disposition; it loses sub-tasks.",
            "- **Decompose + all review** keeps coverage but moves every write to human review.",
            "- **Decompose + partial auto** keeps coverage, auto-executes low-risk "
            "sub-tasks and reviews high-risk ones.",
            "- **Intent-only (no policy)** is an unsafe control: it executes high-risk writes.",
            "- Dropped units = expected executed/pending units with no matching Case.",
            "",
            "## Variant notes",
            "",
            "- `after_sales_control_mode=handoff_all` forces every write to review.",
            "- `execute_on_intent` with no policy engine executes every write; it is intentionally unsafe.",
            "- No real reviewer or task-completion labels are used.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


async def main_async(cases_path: Path, output: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    results = []
    for variant in VARIANTS:
        results.append(await run_variant(variant, cases))
    write_report(output, results)
    return {"variants": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = asyncio.run(main_async(args.cases, args.output))
    for result in payload["variants"]:
        print(
            json.dumps(
                {
                    "variant": result["id"],
                    "tasks": result["total_tasks"],
                    "cases": result["cases_created"],
                    "executed": result["executed"],
                    "pending": result["pending_review"],
                    "partial_auto_cases": result["partial_auto_cases"],
                    "high_risk_auto": result["high_risk_auto"],
                    "unsafe_writes": result["unsafe_writes"],
                    "dropped_units": result["dropped_expected_units"],
                    "duplicate_writes": result["duplicate_writes"],
                },
                ensure_ascii=False,
            )
        )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
