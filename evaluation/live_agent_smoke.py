from __future__ import annotations

import asyncio
import os
import uuid

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

MESSAGES = [
    ("qa", "health beauty 类目有什么运营风险？"),
    ("policy", "退款补偿能不能直接承诺？"),
    (
        "escalation",
        "查订单 203096f03d82e0dffbc41ebc2e2bcfb7 状态，并且说明退款政策，然后提交退款申请",
    ),
]


async def main() -> None:
    settings = resolve_llm_settings(default_model="deepseek-v4-flash")
    if settings.provider == "offline":
        raise SystemExit(
            "OPENAI_API_KEY or AIHUBMIX_API_KEY is not set. Set one in the current "
            "shell before running live smoke evaluation."
        )
    llm_client = LlmClient(
        api_key=settings.api_key,
        model=settings.model,
        base_url=settings.base_url or "https://api.deepseek.com",
    )
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
    graph = AgentGraph(actions).build(InMemorySaver())
    guardrail = Guardrail(llm_client.chat_openai)
    offset = int(os.environ.get("LIVE_SMOKE_OFFSET", "0"))
    limit = int(os.environ.get("LIVE_SMOKE_LIMIT", str(len(MESSAGES))))
    cases = MESSAGES[offset : offset + limit]
    print(f"model={llm_client.model} mode={llm_client.mode} cases={len(cases)}")
    for expected_intent, message in cases:
        input_check = await guardrail.check_input(message)
        if not input_check.on_topic:
            print(f"Case expected={expected_intent} REJECTED by guard: {input_check.reason}")
            continue
        session_id = f"live-smoke-{uuid.uuid4().hex[:8]}"
        state: AgentState = {
            "session_id": session_id,
            "messages": [{"role": "user", "content": message}],
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
        answer = str(result.get("final_answer", "")).replace("\n", " | ")
        output_check = await guardrail.check_output(str(result.get("final_answer", "")))
        task_intents = [task.get("intent") for task in result.get("task_plan", [])]
        completed_statuses = [task.get("status") for task in result.get("completed_tasks", [])]
        hitl = bool(result.get("pending_side_effect", {}).get("requires_confirmation"))
        print(
            f"Case expected={expected_intent} route={result.get('route_intent')} "
            f"tasks={task_intents} statuses={completed_statuses} hitl={hitl} "
            f"output_valid={output_check.valid} answer={answer[:500]}"
        )


if __name__ == "__main__":
    asyncio.run(main())
