from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterable
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.agent.state import AgentState
from app.llm.intent_planner import IntentPlanner
from app.llm.offline import OfflineChatModel
from app.llm.response_generator import OlistTaskExtractor, PolicyResponseGenerator, QaResponseGenerator
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import InMemoryCaseService, OlistService
from app.retrieval.hybrid import HybridSupportRetriever

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "olist_derived" / "eval_cases.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_trajectory(events: Iterable[dict], expected_intent: str) -> dict[str, bool]:
    event_list = list(events)
    intents = [str(event.get("intent")) for event in event_list]
    tools = [str(event.get("details", {}).get("tool", "")) for event in event_list]
    statuses = [str(event.get("status")) for event in event_list]
    side_effect_terminal_statuses = {
        "awaiting_confirmation",
        "completed",
        "execute",
        "hitl",
        "reject",
        "ask_clarification",
        "needs_human_review",
    }
    return {
        "has_plan": any(event.get("node") == "plan_tasks" for event in event_list),
        "intent_covered": expected_intent in intents,
        "no_failed_event": "failed" not in statuses and "blocked" not in statuses,
        "expected_tool_used": _expected_tool(expected_intent) in tools
        if _expected_tool(expected_intent)
        else True,
        "controlled_side_effect_terminal": (
            any(status in side_effect_terminal_statuses for status in statuses)
            if expected_intent == "escalation"
            else True
        ),
    }


def _expected_tool(intent: str) -> str:
    return {
        "order_status": "get_order_status",
        "qa": "search_category_risk",
        "policy": "search_policy_knowledge",
        "ops_decision": "generate_after_sales_priority_report",
        "escalation": "prepare_side_effect",
    }.get(intent, "")


async def evaluate() -> None:
    cases = load_jsonl(CASES)
    graph = _build_offline_graph()
    metric_totals = {
        "has_plan": 0,
        "intent_covered": 0,
        "expected_tool_used": 0,
        "controlled_side_effect_terminal": 0,
        "no_failed_event": 0,
    }
    failures: list[str] = []
    for case in cases:
        expected_intent = case.get("expected_intent", case.get("expected_skill"))
        result = await _run_case(graph, case)
        scores = score_trajectory(result.get("trajectory_events", []), expected_intent)
        for metric, ok in scores.items():
            metric_totals[metric] += int(ok)
        if not all(scores.values()):
            failures.append(f"{case['id']} expected={expected_intent} scores={scores}")

    n = len(cases)
    print(f"Trajectory eval cases: {n} (real LangGraph offline runs)")
    print("metric,score")
    for metric, total in metric_totals.items():
        print(f"{metric},{total / n:.2%}")
    if failures:
        print("Sample failures:")
        for failure in failures[:10]:
            print(f"  {failure}")


def _build_offline_graph():
    llm = OfflineChatModel()
    actions = AgentActions(
        intent_planner=IntentPlanner(llm),
        qa_generator=QaResponseGenerator(llm),
        policy_generator=PolicyResponseGenerator(llm),
        task_extractor=OlistTaskExtractor(llm),
        olist_service=OlistService(),
        knowledge_base=MarkdownKnowledgeBase(),
        support_retriever=HybridSupportRetriever(),
        case_service=InMemoryCaseService(),
    )
    return AgentGraph(actions).build(InMemorySaver())


async def _run_case(graph, case: dict) -> dict:
    session_id = f"trajectory-eval-{case['id']}-{uuid.uuid4()}"
    state: AgentState = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": case["message"]}],
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
    return await graph.ainvoke(state, {"configurable": {"thread_id": session_id}})


def main() -> None:
    asyncio.run(evaluate())


if __name__ == "__main__":
    main()
