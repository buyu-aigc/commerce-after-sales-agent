"""Dependency injection — module-level singletons wired together."""

import os

from dotenv import load_dotenv

from app.after_sales.shadow import build_shadow_service_from_env
from app.agent.actions import AgentActions
from app.agent.customer_order_agent import CustomerOrderAgent
from app.agent.graph import AgentGraph
from app.case_workflow_service import CaseWorkflowService
from app.config.llm_settings import resolve_llm_settings
from app.llm.client import LlmClient
from app.llm.guardrail import Guardrail
from app.llm.intent_planner import IntentPlanner
from app.llm.response_generator import OlistTaskExtractor, PolicyResponseGenerator, QaResponseGenerator
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import OlistService, SQLiteCaseService
from app.policy.engine import PolicyEngine
from app.policy.guard import PolicyGuard
from app.retrieval.hybrid import HybridSupportRetriever
from app.tool_call import build_business_tool_manager, build_runtime_store_from_env

load_dotenv()

knowledge_base = MarkdownKnowledgeBase()
support_retriever = HybridSupportRetriever()
case_service = SQLiteCaseService()
olist_service = OlistService(projection_store=case_service)
runtime_store = build_runtime_store_from_env()
shadow_service = build_shadow_service_from_env()
tool_manager = build_business_tool_manager(
    olist_service=olist_service,
    knowledge_base=knowledge_base,
    support_retriever=support_retriever,
    case_service=case_service,
    cache_backend=runtime_store,
    runtime_store=runtime_store,
)

llm_settings = resolve_llm_settings(default_model="gpt-4o-mini")
llm_client = LlmClient(
    api_key=llm_settings.api_key,
    model=llm_settings.model,
    base_url=llm_settings.base_url,
)
runtime_backend = (
    os.environ.get("RUNTIME_STORE_BACKEND")
    or os.environ.get("TOOL_CACHE_BACKEND")
    or "memory"
).strip().lower() or "memory"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


runtime_status = {
    "mode": llm_client.mode,
    "model": llm_client.model,
    "base_url": llm_client.base_url,
    "provider": llm_settings.provider,
    "runtime_backend": runtime_backend,
    "rate_limit_per_minute": _env_int("AGENT_RATE_LIMIT_PER_MINUTE", 1000),
    "shadow_mode": shadow_service.enabled,
    "shadow_sample_rate": shadow_service.sample_rate if shadow_service.enabled else 0.0,
}

intent_planner = IntentPlanner(llm_client.chat_openai)
qa_generator = QaResponseGenerator(llm_client.chat_openai)
policy_generator = PolicyResponseGenerator(llm_client.chat_openai)
task_extractor = OlistTaskExtractor(llm_client.chat_openai)
guardrail = Guardrail(llm_client.chat_openai)
policy_engine = PolicyEngine()
policy_guard = PolicyGuard(policy_engine, olist_service, case_service)
customer_order_agent = CustomerOrderAgent(llm_client.chat_openai, olist_service, policy_guard)

actions = AgentActions(
    intent_planner=intent_planner,
    qa_generator=qa_generator,
    policy_generator=policy_generator,
    task_extractor=task_extractor,
    olist_service=olist_service,
    knowledge_base=knowledge_base,
    support_retriever=support_retriever,
    case_service=case_service,
    tool_manager=tool_manager,
    runtime_store=runtime_store,
    policy_engine=policy_engine,
    policy_guard=policy_guard,
    shadow_service=shadow_service,
)

agent_graph_builder = AgentGraph(actions=actions)
case_workflow_service = CaseWorkflowService(case_service=case_service, actions=actions)
actions.bind_case_workflow(case_workflow_service)
