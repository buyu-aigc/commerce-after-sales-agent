import asyncio
import hashlib
import logging
import os
import re
import time

from langgraph.types import interrupt

from app import observability
from app.after_sales.address import missing_address_fields
from app.after_sales.complexity import assess_request
from app.agent.state import AgentState
from app.intent.turn_contract import clarification_message, execution_tasks
from app.llm.intent_planner import IntentPlanner
from app.llm.response_generator import OlistTaskExtractor, PolicyResponseGenerator, QaResponseGenerator
from app.llm.types import OlistTaskResult
from app.olist.category_labels import category_label
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import (
    InMemoryCaseService,
    OlistService,
    format_after_sales_report,
)
from app.retrieval.hybrid import HybridSupportRetriever
from app.tool_call import RuntimeStore, ToolCallContext, ToolCallManager, build_business_tool_manager
from app.tools.repair import repair_order_id

logger = logging.getLogger(__name__)


class AgentActions:
    """Business actions wired as LangGraph nodes."""

    def __init__(
        self,
        intent_planner: IntentPlanner,
        qa_generator: QaResponseGenerator,
        policy_generator: PolicyResponseGenerator,
        task_extractor: OlistTaskExtractor,
        olist_service: OlistService,
        knowledge_base: MarkdownKnowledgeBase,
        support_retriever: HybridSupportRetriever,
        case_service: InMemoryCaseService,
        tool_manager: ToolCallManager | None = None,
        runtime_store: RuntimeStore | None = None,
        after_sales_control_mode: str = "full",
        policy_engine: object | None = None,
        policy_guard: object | None = None,
        shadow_service: object | None = None,
        enable_complex_decomposition: bool = True,
        force_single_request: bool = False,
    ) -> None:
        self._intent_planner = intent_planner
        self._qa_generator = qa_generator
        self._policy_generator = policy_generator
        self._task_extractor = task_extractor
        self._olist_service = olist_service
        self._knowledge_base = knowledge_base
        self._support_retriever = support_retriever
        self._case_service = case_service
        self._runtime_store = runtime_store
        self._policy_engine = policy_engine
        self._policy_guard = policy_guard
        self._shadow_service = shadow_service
        self._enable_complex_decomposition = enable_complex_decomposition
        self._force_single_request = force_single_request
        if after_sales_control_mode not in {"full", "handoff_all", "execute_on_intent"}:
            raise ValueError(f"Unsupported after-sales control mode: {after_sales_control_mode}")
        self._after_sales_control_mode = after_sales_control_mode
        self._case_workflow = None
        self._tool_manager = tool_manager or build_business_tool_manager(
            olist_service=olist_service,
            knowledge_base=knowledge_base,
            support_retriever=support_retriever,
            case_service=case_service,
            runtime_store=runtime_store,
        )

    def bind_case_workflow(self, service: object) -> None:
        self._case_workflow = service

    def _evaluate_policy(
        self,
        action_type: str,
        draft: dict,
        decision: dict,
        state: dict | None = None,
        task: dict | None = None,
    ) -> dict | None:
        # Risk must be scoped to the current sub-request.  A high-risk sibling
        # earlier in the same turn must not poison a later low-risk action; the
        # converse is more dangerous, so an explicitly high-risk task always
        # wins over a medium decision from the local rule engine.
        task = task or {}
        task_risk = str(task.get("risk_level") or "")
        decision_risk = str(decision.get("risk_level") or "")
        state_risk = str((state or {}).get("risk_level") or "")
        risk_level = (
            "high" if "high" in {task_risk, decision_risk} else (task_risk or decision_risk or state_risk)
        )
        if self._policy_guard is not None:
            order_id = str(draft.get("order_id") or "")
            if not order_id:
                return None
            return self._policy_guard.check(
                action_type,
                order_id,
                risk_level=risk_level,
                repeat_claim=bool((state or {}).get("repeat_claim")),
                refund_failure=bool((state or {}).get("refund_failure")),
            )
        if self._policy_engine is None:
            return None
        order_id = str(draft.get("order_id") or "")
        order = self._olist_service.get_order_status(order_id) if order_id else None
        if order is None:
            return None
        facts = {
            "action_type": action_type,
            "order_status": order.status,
            "delay_days": int(order.delay_days or 0),
            "amount": float(order.payment_value or 0),
            "prior_compensation": bool(self._case_service.count_order_events(order_id)),
            "repeat_claim": False,
            "risk_level": str(decision.get("risk_level") or ""),
        }
        try:
            return self._policy_engine.evaluate(facts)
        except Exception:
            return None

    def _ensure_auto_case(self, draft: dict, state: AgentState, policy: dict) -> str:
        """Create an audit-visible Case before an auto execution.

        Auto execution used to update only the order projection.  That made the
        action invisible in customer progress and the review/audit trail.
        """
        existing = str(draft.get("review_case_id") or "")
        if existing:
            return existing
        case_service = getattr(self, "_case_service", None)
        creator = getattr(case_service, "create_auto_case", None)
        if creator is None:
            return ""
        try:
            request_analysis = dict(state.get("request_analysis") or {})
            record = creator(
                action_type=str(draft.get("action_type") or "open_support_case"),
                order_id=str(draft.get("order_id") or ""),
                message_text=str(draft.get("message_text") or ""),
                session_id=str(state.get("session_id") or ""),
                user_id=str(state.get("user_id") or "demo-customer"),
                tenant_id=str(state.get("tenant_id") or "olist-demo"),
                case_payload={
                    "auto_execute": True,
                    "policy_decision": policy,
                    "draft": draft,
                    "sub_task": draft.get("sub_task"),
                    "request_id": request_analysis.get("request_id"),
                    "request_tier": request_analysis.get("tier"),
                    "complexity": {
                        key: request_analysis.get(key)
                        for key in (
                            "tier",
                            "risk_level",
                            "intent_count",
                            "action_types",
                            "evidence_completeness",
                            "sub_task_count",
                        )
                    },
                },
            )
        except Exception:
            return ""
        case_id = str(record.get("case_id") or "")
        if case_id:
            draft["review_case_id"] = case_id
        return case_id

    def _link_shadow_case(self, case_id: str, state: AgentState, task: dict, action_type: str) -> None:
        if not case_id or self._shadow_service is None:
            return
        try:
            request_id = str((state.get("request_analysis") or {}).get("request_id") or "")
            self._shadow_service.link_case(
                case_id=case_id,
                request_id=request_id,
                sub_task_index=int(task.get("sub_task_index") or 0),
                action_type=action_type,
            )
        except Exception as exc:
            logger.warning("shadow case link failed: %s", exc)

    @staticmethod
    def _mark_auto_case(case_service: object, case_id: str, executed: bool, receipt: dict) -> None:
        marker = getattr(case_service, "mark_case_result", None)
        if not case_id or marker is None:
            return
        marker(
            case_id,
            "executed" if executed else "execution_failed",
            reviewer_id="system_auto",
            receipt=receipt,
        )

    async def plan_tasks(self, state: AgentState) -> dict:
        if state.get("validated_turn_plan") is not None:
            tasks = execution_tasks(
                state["validated_turn_plan"],
                user_id=str(state["user_id"]),
                session_id=str(state["session_id"]),
                allowed=set(state.get("allowed_order_ids", [])),
                message=state["messages"][-1]["content"],
            )
            request_id = _request_id(state, state["messages"][-1]["content"])
            for index, task in enumerate(tasks, 1):
                task.update(sub_task_index=index, sub_task_total=len(tasks), request_id=request_id)
            writes = [t for t in tasks if t.get("side_effect")]
            summary = {
                "request_id": request_id,
                "planning_mode": "validated_turn_contract",
                "decomposition_mode": "validated_turn_contract",
                "tier": "complex" if len(tasks) > 1 else "simple",
                "risk_level": "pending_policy" if writes else "low",
                "intent_count": len(tasks),
                "sub_task_count": len(tasks),
                "write_task_count": len(writes),
                "read_task_count": len(tasks) - len(writes),
                "action_types": [t["action_type"] for t in writes],
            }
            return {
                "task_plan": tasks,
                "route_intent": tasks[0]["intent"],
                "request_analysis": summary,
                "trajectory_events": [
                    *state.get("trajectory_events", []),
                    _event(
                        "plan_tasks",
                        "contract",
                        "validated",
                        {"tasks": tasks, "origin": state["validated_turn_plan"]["origin"]},
                    ),
                ],
            }
        started_at = time.perf_counter()
        last_message: str = state["messages"][-1]["content"]
        requested_action = str(state.get("requested_action") or "").strip()
        selected_order = str(dict(state.get("page_context", {})).get("selected_order_id") or "")
        analysis = assess_request(
            last_message,
            selected_order_id=selected_order,
            order_lookup=self._olist_service.get_order_status,
            extra_high_risk_flags=(
                "repeat_claim" if state.get("repeat_claim") else "",
                "refund_failure" if state.get("refund_failure") else "",
            ),
        )
        if analysis.restricted_actions:
            # Textual prohibitions apply before UI shortcuts, compound
            # splitting and planner fallbacks. Keep independently affirmative
            # requests, but never turn an excluded mention into a new Case.
            tasks = analysis.to_tasks()
            for unit, task in zip(analysis.units, tasks):
                if unit.semantic_fallback:
                    task.update(intent="clarify", side_effect=False, action_type="none")
                if task.get("order_id_scope") == "inherited" and task.get("order_id"):
                    task["text"] = f"订单 {task['order_id']} {task['text']}"
            planning_mode = "scoped_request_constraints"
            analysis_summary = analysis.summary()
            if not tasks:
                tasks = [
                    {
                        "intent": "clarify",
                        "text": last_message,
                        "side_effect": False,
                        "action_type": "none",
                        "depends_on": [],
                    }
                ]
        elif requested_action:
            tasks = _structured_action_plan(requested_action, last_message)
            planning_mode = "structured_page_action"
            analysis_summary = _structured_request_summary(analysis, requested_action, planning_mode)
        elif self._force_single_request:
            tasks = _single_request_control_task(last_message, analysis)
            planning_mode = "single_request_control"
            analysis_summary = analysis.summary()
            analysis_summary.update(
                {
                    "decomposition_mode": "single_request_control",
                    "sub_task_count": 1,
                    "write_task_count": 1 if analysis.write_units else 0,
                    "read_task_count": 0 if analysis.write_units else 1,
                }
            )
        else:
            selected_action = _selected_order_action_type(last_message, state)
            if (
                analysis.units
                and not analysis.write_units
                and not any(unit.semantic_fallback for unit in analysis.units)
            ):
                tasks = analysis.to_tasks()
                planning_mode = "read_request_decomposition"
                analysis_summary = analysis.summary()
            elif analysis.is_complex and analysis.write_units and self._enable_complex_decomposition:
                tasks = analysis.to_tasks()
                planning_mode = "complex_request_decomposition"
                analysis_summary = analysis.summary()
            elif selected_action:
                tasks = _structured_action_plan(selected_action, last_message)
                planning_mode = "selected_order_action"
                analysis_summary = _structured_request_summary(analysis, selected_action, planning_mode)
            else:
                plan = await self._intent_planner.plan(last_message)
                tasks = [task.model_dump() for task in plan.tasks]
                tasks = _enrich_tasks_with_analysis(tasks, analysis, complex_request=analysis.is_complex)
                planning_mode = plan.planning_mode
                analysis_summary = analysis.summary()
        if not tasks:
            tasks = analysis.to_tasks() or [
                {
                    "intent": "clarify",
                    "text": last_message,
                    "side_effect": False,
                    "action_type": "none",
                    "depends_on": [],
                }
            ]
        request_id = _request_id(state, last_message)
        analysis_summary["request_id"] = request_id
        analysis_summary["planning_mode"] = planning_mode
        shadow_suggestions: list[dict[str, object]] = []
        if self._shadow_service is not None and state.get("shadow_mode"):
            try:
                shadow_suggestions = list(
                    self._shadow_service.record_analysis(
                        request_id=request_id,
                        session_id=str(state.get("session_id") or ""),
                        analysis=analysis,
                        order_lookup=self._olist_service.get_order_status,
                        policy_guard=self._policy_guard,
                        repeat_claim=bool(state.get("repeat_claim")),
                        refund_failure=bool(state.get("refund_failure")),
                    )
                )
            except Exception as exc:  # shadow must never break the customer path
                logger.warning("shadow suggestion recording failed: %s", exc)
        for index, task in enumerate(tasks, start=1):
            task.setdefault("sub_task_index", index)
            task.setdefault("sub_task_total", len(tasks))
            task.setdefault("request_id", request_id)
        route_intent = str(tasks[0]["intent"]) if tasks else "policy"
        dialog_state = _merge_dialog_state(
            state,
            current_intent=route_intent,
            task_count=len(tasks),
            turn_count=_turn_count(state),
        )
        return {
            "route_intent": route_intent,
            "task_plan": tasks,
            "request_analysis": analysis_summary,
            "shadow_suggestions": shadow_suggestions,
            "turn_count": dialog_state["turn_count"],
            "dialog_state": dialog_state,
            "trajectory_events": [
                *state.get("trajectory_events", []),
                _event(
                    "plan_tasks",
                    "planner",
                    "completed",
                    {
                        "task_count": len(tasks),
                        "tasks": tasks,
                        "planning_mode": planning_mode,
                        "request_tier": analysis_summary.get("tier"),
                        "request_risk": analysis_summary.get("risk_level"),
                        "shadow_suggestions": len(shadow_suggestions),
                        "dialog_state": dialog_state,
                        "duration_ms": _elapsed_ms(started_at),
                    },
                ),
            ],
        }

    def select_next_task(self, state: AgentState) -> dict:
        completed = list(state.get("completed_tasks", []))
        terminal_indices = _terminal_task_indices(completed)
        completed_indices = _completed_task_indices(completed)
        tasks = state.get("task_plan", [])

        for idx, task in _execution_order(tasks):
            if idx in terminal_indices:
                continue
            dependencies = [int(dep) for dep in task.get("depends_on", []) if isinstance(dep, int)]
            missing = [dep for dep in dependencies if dep not in completed_indices]
            if missing:
                completed = _upsert_task_status(
                    completed,
                    idx,
                    str(task.get("intent", "policy")),
                    "blocked",
                )
                answer_parts = [
                    *state.get("answer_parts", []),
                    "[任务阻断]\n前置任务没有完成，已停止后续可能有副作用的动作。",
                ]
                return {
                    "completed_tasks": completed,
                    "answer_parts": answer_parts,
                    "workflow_complete": True,
                    "trajectory_events": [
                        *state.get("trajectory_events", []),
                        _event(
                            "select_next_task",
                            str(task.get("intent", "policy")),
                            "blocked",
                            {"task_index": idx, "missing_dependencies": missing},
                        ),
                    ],
                }

            return {
                "current_task_index": idx,
                "current_task": task,
                "current_task_text": str(task.get("text") or _latest_user_task_message(state)),
                "workflow_complete": False,
                "trajectory_events": [
                    *state.get("trajectory_events", []),
                    _event(
                        "select_next_task",
                        str(task.get("intent", "policy")),
                        "selected",
                        {"task_index": idx, "side_effect": bool(task.get("side_effect"))},
                    ),
                ],
            }

        return {
            "current_task": {},
            "current_task_text": "",
            "workflow_complete": True,
            "trajectory_events": [
                *state.get("trajectory_events", []),
                _event("select_next_task", "workflow", "completed"),
            ],
        }

    async def extract_slots(self, state: AgentState) -> dict:
        if state.get("workflow_complete"):
            return {}
        started_at = time.perf_counter()
        task = dict(state.get("current_task", {}))
        intent = str(task.get("intent", "policy"))
        text = str(state.get("current_task_text") or _latest_user_task_message(state))
        task_order_id = str(task.get("order_id") or "").strip().lower()
        task_order_scope = str(task.get("order_id_scope") or "")
        task_order_repair = repair_order_id(task_order_id) if task_order_id else None
        if task.get("contract_task_id") or task_order_scope == "missing":
            # A compiled task never borrows its siblings' raw text.
            # A decomposed sub-request that explicitly has no order must not
            # silently borrow a sibling sub-request's order id.
            slot_text = text
        else:
            slot_text = _with_order_context(text, _latest_user_task_message(state))
        selected_order_id = str(
            dict(state.get("page_context", {})).get("selected_order_id") or state.get("active_order_id", "")
        ).lower()
        selected_repair = repair_order_id(selected_order_id) if selected_order_id else None
        task_repair = repair_order_id(slot_text)
        if intent in {"order_status", "escalation"} and task_order_repair and task_order_repair.ok:
            extracted = OlistTaskResult(
                order_id=task_order_repair.value, category="", user_goal="task_decomposition"
            )
            extraction_mode = "task_decomposition"
        elif intent in {"order_status", "escalation"} and task_order_scope == "missing":
            extracted = OlistTaskResult(order_id="", category="", user_goal="task_decomposition_missing")
            extraction_mode = "task_decomposition_missing"
        elif intent in {"order_status", "escalation"} and task_repair.ok:
            # An explicit order in this task wins over a previous page selection;
            # multi-task turns may legitimately reference different orders.
            extracted = OlistTaskResult(order_id=task_repair.value, category="", user_goal="task_text")
            extraction_mode = "explicit_task_order_id"
        elif intent in {"order_status", "escalation"} and selected_repair and selected_repair.ok:
            # A native order-page control carries a server-validated selection.
            # Avoid a second model call to rediscover an ID the product already
            # knows, while preserving the same downstream tool gates.
            extracted = OlistTaskResult(order_id=selected_repair.value, category="", user_goal="page_context")
            extraction_mode = "trusted_page_context"
        elif intent in {
            "small_talk",
            "clarify",
            "customer_profile",
            "policy",
            "qa",
            "ops_decision",
            "customer_orders",
            "case_progress",
        }:
            # Retrieval consumes the full task text. These read-only intents do
            # not need an order/category slot before retrieval can begin.
            extracted = OlistTaskResult(order_id="", category="", user_goal="not_required")
            extraction_mode = "deterministic_not_required"
        else:
            extracted = self._task_extractor.fast_extract(slot_text)
            extraction_mode = "deterministic_exact_order_id"
            if extracted is None:
                extracted = await self._task_extractor.extract(slot_text)
                extraction_mode = "llm"
        if task_order_repair and task_order_repair.ok:
            order_repair = task_order_repair
        elif task_order_scope == "missing":
            order_repair = repair_order_id("")
        else:
            order_repair = repair_order_id(extracted.order_id or slot_text)
            if not order_repair.ok and selected_order_id:
                if selected_repair and selected_repair.ok:
                    order_repair = selected_repair
                    extraction_mode = "trusted_page_context"
        slots = {
            "order_id": order_repair.value if order_repair.ok else "",
            "order_id_ok": order_repair.ok,
            "order_id_error": "" if order_repair.ok else order_repair.message,
            "category": extracted.category,
            "user_goal": extracted.user_goal,
        }
        if intent == "escalation":
            slots["action_type"] = str(task.get("action_type") or self._infer_action_type(text))
        dialog_state = _merge_dialog_state(
            state,
            current_intent=intent,
            slots=slots,
            missing_slots=_missing_slots(intent, slots),
        )
        return {
            "current_slots": slots,
            "active_order_id": str(dialog_state.get("active_order_id", "")),
            "filled_slots": dict(dialog_state.get("filled_slots", {})),
            "missing_slots": list(dialog_state.get("missing_slots", [])),
            "dialog_state": dialog_state,
            "trajectory_events": [
                *state.get("trajectory_events", []),
                _event(
                    "extract_slots",
                    intent,
                    "completed",
                    {
                        "task_index": state.get("current_task_index"),
                        "has_order_id": bool(slots["order_id"]),
                        "category": slots["category"],
                        "missing_slots": dialog_state.get("missing_slots", []),
                        "extraction_mode": extraction_mode,
                        "duration_ms": _elapsed_ms(started_at),
                    },
                ),
            ],
        }

    async def retrieve_context(self, state: AgentState) -> dict:
        if state.get("workflow_complete"):
            return {}
        started_at = time.perf_counter()
        task = dict(state.get("current_task", {}))
        intent = str(task.get("intent", "policy"))
        text = str(state.get("current_task_text") or _latest_user_task_message(state))
        slots = dict(state.get("current_slots", {}))
        context: dict[str, object] = {}
        update: dict[str, object] = {}

        if intent in {"small_talk", "clarify"}:
            context["primary_tool"] = "none"
        elif intent in {"customer_orders", "customer_profile"}:
            context["primary_tool"] = "list_owned_orders"
            result = await self._tool_manager.call(
                "list_owned_orders",
                {"filter": str(task.get("order_filter", "all"))},
                self._tool_context(state),
            )
            context["owned_orders"] = list(result.data.get("orders", [])) if result.ok else []
        elif intent == "order_status":
            context["primary_tool"] = "get_order_status"
            if slots.get("order_id_ok"):
                order_id = str(slots["order_id"])
                result = await self._tool_manager.call(
                    "get_order_status",
                    {"order_id": order_id},
                    self._tool_context(state),
                )
                context["order_status_result"] = result.model_dump()
            else:
                context["order_id_error"] = str(slots.get("order_id_error", "没有检测到有效订单号。"))
        elif intent == "qa":
            context["primary_tool"] = "search_category_risk"
            insights, support_docs = await asyncio.gather(
                self._category_insights(text, state),
                self._search_support_docs(text, state),
            )
            context["insights"] = insights
            context["support_docs"] = support_docs
            update["retrieved_insights"] = [*state.get("retrieved_insights", []), *insights]
            update["retrieved_support_docs"] = [*state.get("retrieved_support_docs", []), *support_docs]
        elif intent == "ops_decision":
            context["primary_tool"] = "generate_after_sales_priority_report"
            report = await self._after_sales_priority_report(text, state)
            context["ops_report"] = report
            update["retrieved_insights"] = [
                *state.get("retrieved_insights", []),
                *list(report.get("high_risk_categories", [])),
            ]
        elif intent in {"policy", "escalation"}:
            context["primary_tool"] = (
                "search_policy_knowledge" if intent == "policy" else "prepare_side_effect"
            )
            policy_query = text
            if intent == "escalation":
                action_type = str(slots.get("action_type") or task.get("action_type") or "open_support_case")
                policy_query = f"{text}\n售后动作：{_ACTION_LABELS.get(action_type, action_type)}"
            # The policy pack is the decision source of truth. Public support
            # conversations are retained for a separate retrieval experiment,
            # not injected into this customer-facing policy/write path.
            sections = await self._search_policy_knowledge(policy_query, state)
            support_docs: list[dict[str, object]] = []
            context["policy_sections"] = sections
            context["support_docs"] = support_docs
            update["retrieved_policy"] = [*state.get("retrieved_policy", []), *sections]
            update["retrieved_support_docs"] = [*state.get("retrieved_support_docs", []), *support_docs]

        update["current_context"] = context
        update["dialog_state"] = _merge_dialog_state(
            state,
            current_intent=intent,
            retrieved_policy_count=len(context.get("policy_sections", []))
            if isinstance(context.get("policy_sections"), list)
            else 0,
            retrieved_support_count=len(context.get("support_docs", []))
            if isinstance(context.get("support_docs"), list)
            else 0,
            retrieved_insight_count=len(context.get("insights", []))
            if isinstance(context.get("insights"), list)
            else 0,
        )
        update["trajectory_events"] = [
            *state.get("trajectory_events", []),
            _event(
                "retrieve_context",
                intent,
                "completed",
                {
                    "task_index": state.get("current_task_index"),
                    "tool": context.get("primary_tool"),
                    "policy_hits": len(context.get("policy_sections", []))
                    if isinstance(context.get("policy_sections"), list)
                    else 0,
                    "support_hits": len(context.get("support_docs", []))
                    if isinstance(context.get("support_docs"), list)
                    else 0,
                    "insight_hits": len(context.get("insights", []))
                    if isinstance(context.get("insights"), list)
                    else 0,
                    "duration_ms": _elapsed_ms(started_at),
                },
            ),
        ]
        if intent == "policy":
            update["trajectory_events"].append(
                _event(
                    "search_policy_knowledge",
                    intent,
                    "completed",
                    {
                        "task_index": state.get("current_task_index"),
                        "tool": "search_policy_knowledge",
                        "hit_count": len(context.get("policy_sections", []))
                        if isinstance(context.get("policy_sections"), list)
                        else 0,
                    },
                )
            )
        return update

    async def execute_read_task(self, state: AgentState) -> dict:
        started_at = time.perf_counter()
        task = dict(state.get("current_task", {}))
        intent = str(task.get("intent", "policy"))
        text = str(state.get("current_task_text") or _latest_user_task_message(state))
        context = dict(state.get("current_context", {}))
        answer = ""

        if intent == "small_talk":
            answer = "你好，我可以帮你查询订单、了解售后政策或提交售后申请。请告诉我想处理什么问题。"
        elif intent == "clarify":
            answer = (
                clarification_message(task)
                if task.get("contract_task_id")
                else (
                    "可以的。请告诉我想查询订单、了解退款/取消政策，还是申请退款、取消订单或修改地址；"
                    "你也可以先在订单列表中选中对应订单。"
                )
            )
        elif intent == "case_progress":
            record = self._case_service.get(str(task.get("case_id") or "")) if task.get("case_id") else None
            if (
                record
                and record.get("user_id") == state.get("user_id")
                and record.get("session_id") == state.get("session_id")
            ):
                labels = {
                    "pending_review": "正在审核",
                    "appealed_pending_review": "正在复核",
                    "approving": "正在处理",
                    "executed": "已处理完成",
                    "rejected": "审核未通过",
                    "timeout_canceled": "已关闭",
                }
                answer = (
                    "这项申请" + labels.get(record["status"], "正在核查") + "；本次查询没有重新提交申请。"
                )
            else:
                answer = "本会话没有查到对应售后申请，本次不会替您重新提交。"
        elif intent == "customer_orders":
            answer = _format_owned_orders(
                list(context.get("owned_orders", [])),
                str(task.get("order_filter", "all")),
            )
        elif intent == "customer_profile":
            answer = _format_customer_order_summary(list(context.get("owned_orders", [])))
        elif intent == "order_status":
            if context.get("order_id_error"):
                answer = str(context["order_id_error"])
            else:
                result = context.get("order_status_result")
                data = dict(result.get("data", {})) if isinstance(result, dict) else {}
                answer = str(data.get("answer", "订单事实工具没有返回结果。"))
        elif intent == "qa":
            answer = await self._qa_generator.generate(
                text,
                list(context.get("insights", [])),
                list(context.get("support_docs", [])),
            )
        elif intent == "ops_decision":
            answer = format_after_sales_report(dict(context.get("ops_report", {})))
        elif intent == "policy":
            policy_sections = list(context.get("policy_sections", []))
            answer = (
                self._policy_generator.customer_safe_template(text, policy_sections)
                if str(state.get("channel", "")).startswith("customer")
                else None
            )
            if answer is None:
                answer = await self._policy_generator.generate(
                    text,
                    policy_sections,
                    list(context.get("support_docs", [])),
                    completed_context="\n\n".join(state.get("answer_parts", [])),
                )

        idx = int(state.get("current_task_index", 0))
        completed = _upsert_task_status(list(state.get("completed_tasks", [])), idx, intent, "completed")
        return {
            "completed_tasks": completed,
            "customer_read_results": [
                *state.get("customer_read_results", []),
                {
                    "task_index": idx,
                    "intent": intent,
                    "order_id": state.get("current_slots", {}).get("order_id") or task.get("order_id", ""),
                    "task_id": task.get("contract_task_id"),
                    "required_fields": task.get("required_fields", []),
                    "answer": answer,
                    "tool_result": context.get("order_status_result"),
                },
            ],
            "answer_parts": [
                *state.get("answer_parts", []),
                _answer_part(intent, answer, state),
            ],
            "trajectory_events": [
                *state.get("trajectory_events", []),
                _event(
                    "execute_read_task",
                    intent,
                    "completed",
                    {
                        "task_index": idx,
                        "tool": context.get("primary_tool"),
                        "answer_chars": len(answer),
                        "duration_ms": _elapsed_ms(started_at),
                    },
                ),
            ],
        }

    async def apply_policy_auto(self, state: AgentState) -> dict:
        """Execute an auto-approved policy decision in an isolated graph node."""
        draft = dict(state.get("escalation_draft") or {})
        idx = int(state.get("current_task_index") or 0)
        policy = dict(state.get("policy_decision") or {})
        case_id = self._ensure_auto_case(draft, state, policy)
        execution_answer, executed, receipt = await self._execute_escalation_draft(
            draft,
            state,
            role_override="after_sales_operator",
            auth_scopes_override=["after_sales:write"],
        )
        self._mark_auto_case(getattr(self, "_case_service", None), case_id, executed, receipt)
        observability.record_auto_outcome(policy, executed=executed)
        completed = _upsert_task_status(
            state.get("completed_tasks", []),
            idx,
            "escalation",
            "completed" if executed else "failed",
        )
        answer_parts = [
            *state.get("answer_parts", []),
            f"这笔申请符合自动处理条件，系统已直接处理。\n{execution_answer}",
        ]
        return {
            "case_id": case_id,
            "completed_tasks": completed,
            "answer_parts": answer_parts,
            "final_answer": "\n\n".join(answer_parts),
            "policy_pending": False,
            "trajectory_events": [
                *state.get("trajectory_events", []),
                _event(
                    "execute_write_action",
                    "escalation",
                    "completed" if executed else "failed",
                    {
                        "task_index": idx,
                        "tool": "execute_side_effect",
                        "action_type": draft.get("action_type"),
                        "auto_execute": True,
                        "policy_id": policy.get("policy_id"),
                    },
                ),
            ],
        }

    async def build_after_sales_case(self, state: AgentState) -> dict:
        task = dict(state.get("current_task", {}))
        if state.get("validated_turn_plan") is not None:
            execution_tasks(
                {**state["validated_turn_plan"], "tasks": [task]},
                user_id=str(state["user_id"]),
                session_id=str(state["session_id"]),
                allowed=set(state.get("allowed_order_ids", [])),
            )
            if not task.get("side_effect") or task.get("authorization") != "explicit":
                raise ValueError("case_creation_without_customer_authorization")
        idx = int(state.get("current_task_index", 0))
        text = str(state.get("current_task_text") or _latest_user_task_message(state))
        slots = dict(state.get("current_slots", {}))
        action_type = str(
            slots.get("action_type") or task.get("action_type") or self._infer_action_type(text)
        )
        if task.get("order_id_scope") == "missing" and not slots.get("order_id"):
            completed = _upsert_task_status(
                list(state.get("completed_tasks", [])), idx, "escalation", "completed"
            )
            return {
                "completed_tasks": completed,
                "answer_parts": [
                    *state.get("answer_parts", []),
                    "[售后升级]\n这笔子申请还缺少订单号，请补充要处理的订单号后再提交。",
                ],
                "trajectory_events": [
                    *state.get("trajectory_events", []),
                    _event(
                        "decision_engine",
                        "escalation",
                        "ask_clarification",
                        {
                            "task_index": idx,
                            "action_type": action_type,
                            "reason": "missing_order_id_for_sub_task",
                        },
                    ),
                ],
            }
        slot_text = (
            text
            if task.get("contract_task_id") or task.get("order_id_scope") == "missing"
            else _with_order_context(text, _latest_user_task_message(state))
        )
        answer, draft = await self._prepare_escalation_from_text(slot_text, action_type, state)
        completed = list(state.get("completed_tasks", []))
        after_sales_cases = list(state.get("after_sales_cases", []))
        events = list(state.get("trajectory_events", []))

        if draft is None:
            completed = _upsert_task_status(completed, idx, "escalation", "completed")
            return {
                "completed_tasks": completed,
                "answer_parts": [*state.get("answer_parts", []), f"[售后升级]\n{answer}"],
                "trajectory_events": [
                    *events,
                    _event(
                        "decision_engine",
                        "escalation",
                        "ask_clarification",
                        {
                            "task_index": idx,
                            "tool": "prepare_side_effect",
                            "action_type": action_type,
                        },
                    ),
                ],
            }

        draft["task_index"] = idx
        if isinstance(draft.get("after_sales_case"), dict):
            after_sales_cases.append(dict(draft["after_sales_case"]))
        decision = dict(draft.get("decision", {}))
        verification = dict(draft.get("verification", {}))
        if self._after_sales_control_mode != "full":
            decision, verification = _control_ablation_decision(
                self._after_sales_control_mode,
                action_type,
                decision,
            )
            draft["decision"] = decision
            draft["verification"] = verification
            draft["requires_confirmation"] = verification.get("required_next_step") == "hitl"
            answer = _format_after_sales_decision_answer(
                str(draft.get("order_id", "")),
                action_type,
                draft,
                requires_confirmation=bool(draft["requires_confirmation"]),
                customer_view=_is_customer_channel(state),
            )
        handoff_reasons = list(decision.get("handoff_reasons", []))
        request_analysis = dict(state.get("request_analysis") or {})
        draft["sub_task"] = {
            "index": task.get("sub_task_index"),
            "total": task.get("sub_task_total"),
            "action_type": action_type,
            "risk_level": str(task.get("risk_level") or decision.get("risk_level") or ""),
            "risk_reasons": list(dict.fromkeys([*list(task.get("risk_reasons") or []), *handoff_reasons])),
            "auto_eligible": bool(task.get("auto_eligible")),
            "evidence_complete": task.get("evidence_complete"),
            "missing_evidence": list(task.get("missing_evidence") or []),
            "request_id": request_analysis.get("request_id"),
            "request_tier": request_analysis.get("tier"),
        }
        dialog_state = _merge_dialog_state(
            state,
            current_intent="escalation",
            slots={**slots, "action_type": action_type},
            handoff_reasons=handoff_reasons,
            risk_level=str(decision.get("risk_level", "")),
            decision_outcome=str(decision.get("outcome", "")),
            verifier_next_step=str(verification.get("required_next_step", "")),
        )
        events.extend(
            [
                _event(
                    "decision_engine",
                    "escalation",
                    str(decision.get("outcome", "unknown")),
                    {
                        "task_index": idx,
                        "tool": "prepare_side_effect",
                        "action_type": action_type,
                        "risk_level": decision.get("risk_level"),
                        "requires_human": decision.get("requires_human"),
                        "handoff_reasons": handoff_reasons,
                        "control_mode": self._after_sales_control_mode,
                    },
                ),
                _event(
                    "verifier",
                    "escalation",
                    str(verification.get("required_next_step", "unknown")),
                    {"task_index": idx, "flags": verification.get("flags", [])},
                ),
            ]
        )

        if _decision_stops_execution(decision, verification):
            completed = _upsert_task_status(completed, idx, "escalation", "completed")
            return {
                "completed_tasks": completed,
                "after_sales_cases": after_sales_cases,
                "dialog_state": dialog_state,
                "handoff_reasons": handoff_reasons,
                "answer_parts": [*state.get("answer_parts", []), f"[售后升级]\n{answer}"],
                "trajectory_events": events,
            }

        policy = self._evaluate_policy(action_type, draft, decision, state, task)
        if policy is not None:
            draft["policy_decision"] = policy
            review_sub_task = dict(draft.get("sub_task") or {})
            if review_sub_task:
                if policy.get("decision") == "auto":
                    review_sub_task["risk_reasons"] = list(task.get("risk_reasons") or [])
                    review_sub_task["policy_id"] = policy.get("policy_id")
                    review_sub_task["policy_decision"] = "auto"
                else:
                    review_sub_task["risk_reasons"] = list(
                        dict.fromkeys(
                            [
                                *list(task.get("risk_reasons") or []),
                                *list(handoff_reasons or []),
                            ]
                        )
                    )
                    review_sub_task["policy_id"] = policy.get("policy_id")
                    review_sub_task["policy_decision"] = policy.get("decision")
                draft["sub_task"] = review_sub_task
            observability.record_policy_decision(
                policy,
                action_type=action_type,
                session_id=str(state.get("session_id") or ""),
                user_id=str(state.get("user_id") or ""),
            )
        shadow_mode = bool(state.get("shadow_mode"))
        if shadow_mode:
            # Shadow traffic always waits for a human label.  The suggestion is
            # stored out-of-band and is never shown to the reviewer before the
            # decision, so the comparison stays blind.
            draft["requires_confirmation"] = True
            answer = (
                f"订单 {str(draft.get('order_id') or '')[:8]} 的"
                f"{_ACTION_LABELS.get(action_type, '售后申请')}已进入人工审核，"
                "确认前不会执行。"
            )
        if (
            policy is not None
            and policy.get("decision") == "auto"
            and draft.get("requires_confirmation")
            and not shadow_mode
            and self._after_sales_control_mode == "full"
        ):
            self._ensure_auto_case(draft, state, policy)
            events.append(
                _event(
                    "policy_decision",
                    "escalation",
                    "auto",
                    {
                        "task_index": idx,
                        "policy_id": policy.get("policy_id"),
                        "decision": policy.get("decision"),
                        "action_type": action_type,
                    },
                )
            )
            return {
                "completed_tasks": completed,
                "after_sales_cases": after_sales_cases,
                "dialog_state": dialog_state,
                "handoff_reasons": handoff_reasons,
                "answer_parts": [*state.get("answer_parts", []), f"[售后升级]\n{answer}"],
                "escalation_draft": draft,
                "policy_pending": True,
                "policy_decision": policy,
                "current_task_index": idx,
                "current_task": task,
                "current_slots": slots,
                "trajectory_events": events,
            }

        policy_allows_auto = (self._after_sales_control_mode == "execute_on_intent" and policy is None) or (
            self._after_sales_control_mode == "full" and (policy is None or policy.get("decision") == "auto")
        )
        if draft.get("requires_confirmation") is False and not policy_allows_auto:
            # The deterministic decision engine may approve a low-risk action,
            # but the active policy pack is the source of truth.  A policy
            # review decision must route this sub-request to a human instead of
            # silently executing it.
            answer = (
                f"订单 {draft.get('order_id', '')[:8]} 的{_ACTION_LABELS.get(action_type, '售后申请')}"
                "未命中自动政策，已转人工审核，确认前不会执行。"
            )
        if draft.get("requires_confirmation") is False and policy_allows_auto:
            # The customer never receives a write scope. A low-risk action is
            # executed by the constrained system actor only after the
            # deterministic decision, verifier and active policy all allow it.
            # An auto execution must always leave an audit-visible Case, even
            # in an evaluation control that intentionally runs without policy.
            auto_case_id = self._ensure_auto_case(draft, state, policy or {})
            execution_answer, executed, receipt = await self._execute_escalation_draft(
                draft,
                state,
                role_override="after_sales_operator",
                auth_scopes_override=["after_sales:write"],
            )
            self._mark_auto_case(getattr(self, "_case_service", None), auto_case_id, executed, receipt)
            if policy is not None:
                observability.record_auto_outcome(policy, executed=executed)
            completed = _upsert_task_status(
                completed, idx, "escalation", "completed" if executed else "failed"
            )
            answer_parts = [
                *state.get("answer_parts", []),
                f"[售后升级]\n{answer}\n\n{execution_answer}",
            ]
            return {
                "case_id": auto_case_id,
                "completed_tasks": completed,
                "after_sales_cases": after_sales_cases,
                "dialog_state": dialog_state,
                "handoff_reasons": handoff_reasons,
                "answer_parts": answer_parts,
                "final_answer": "\n\n".join(answer_parts),
                "trajectory_events": [
                    *events,
                    _event(
                        "execute_write_action",
                        "escalation",
                        "completed",
                        {
                            "task_index": idx,
                            "tool": "execute_side_effect",
                            "action_type": action_type,
                            "auto_execute": True,
                        },
                    ),
                ],
            }

        created_at = time.time()
        timeout_seconds = _hitl_timeout_seconds()
        if self._case_workflow is not None and state.get("decouple_cases"):
            case_record = await self._case_workflow.dispatch_case(
                action_type=str(draft.get("action_type", "open_support_case")),
                order_id=str(draft.get("order_id", "")),
                message_text=str(draft.get("message_text", "")),
                state=state,
                draft=draft,
                expires_at=created_at + timeout_seconds,
                timeout_seconds=timeout_seconds,
                handoff_reasons=handoff_reasons,
            )
            case_id = str(case_record.get("case_id", ""))
            workflow_thread_id = str(case_record.get("workflow_thread_id", ""))
            self._link_shadow_case(case_id, state, task, action_type)
            draft["review_case_id"] = case_id
            draft["case_status"] = str(case_record.get("status", "pending_review"))
            completed = _upsert_task_status(completed, idx, "escalation", "delegated_pending")
            answer_parts = [*state.get("answer_parts", []), f"[售后升级]\n{answer}"]
            final_answer = "\n\n".join(answer_parts)
            delegated = {
                "case_id": case_id,
                "workflow_thread_id": workflow_thread_id,
                "workflow_attempt": int(case_record.get("workflow_attempt") or 1),
                "status": "delegated_pending",
            }
            return {
                "completed_tasks": completed,
                "after_sales_cases": after_sales_cases,
                "dialog_state": dialog_state,
                "handoff_reasons": handoff_reasons,
                "answer_parts": answer_parts,
                "final_answer": final_answer,
                "messages": [*state["messages"], {"role": "assistant", "content": final_answer}],
                "escalation_draft": draft,
                "pending_side_effect": {},
                "case_id": case_id,
                "workflow_thread_id": workflow_thread_id,
                "delegated_pending": delegated,
                "trajectory_events": [
                    *events,
                    _event(
                        "case_dispatched",
                        "escalation",
                        "delegated_pending",
                        {
                            "task_index": idx,
                            "tool": "dispatch_case_workflow",
                            "case_id": case_id,
                            "workflow_thread_id": workflow_thread_id,
                        },
                    ),
                ],
            }

        review_case = self._case_service.submit_for_review(
            action_type=str(draft.get("action_type", "open_support_case")),
            order_id=str(draft.get("order_id", "")),
            message_text=str(draft.get("message_text", "")),
            session_id=str(state.get("session_id", "unknown")),
            user_id=str(state.get("user_id", "demo-user")),
            expires_at=created_at + timeout_seconds,
            case_payload={
                "after_sales_case": dict(draft.get("after_sales_case", {})),
                "decision": dict(draft.get("decision", {})),
                "verification": dict(draft.get("verification", {})),
                "policy_decision": draft.get("policy_decision"),
                "sub_task": draft.get("sub_task"),
                "request_id": (state.get("request_analysis") or {}).get("request_id"),
                "request_tier": (state.get("request_analysis") or {}).get("tier"),
                "policy_sources": [
                    str(item.get("section_title", ""))
                    for item in state.get("retrieved_policy", [])
                    if isinstance(item, dict) and item.get("section_title")
                ],
            },
        )
        review_case_id = str(review_case.get("result_id", ""))
        self._link_shadow_case(review_case_id, state, task, action_type)
        draft["review_case_id"] = review_case_id
        draft["case_status"] = dict(review_case.get("record", {})).get("status", "pending_review")
        pending_side_effect = {
            "type": draft.get("action_type", "open_support_case"),
            "requires_confirmation": True,
            "task_intent": "escalation",
            "task_index": idx,
            "review_case_id": review_case_id,
            "created_at": created_at,
            "expires_at": created_at + timeout_seconds,
            "timeout_seconds": timeout_seconds,
            "handoff_reasons": handoff_reasons,
        }
        if self._runtime_store is not None:
            await self._runtime_store.put_pending_confirmation(
                str(state.get("session_id", "unknown")),
                pending_side_effect,
                timeout_seconds,
            )
        completed = _upsert_task_status(completed, idx, "escalation", "awaiting_confirmation")
        answer_parts = [*state.get("answer_parts", []), f"[售后升级]\n{answer}"]
        final_answer = "\n\n".join(answer_parts)
        return {
            "completed_tasks": completed,
            "after_sales_cases": after_sales_cases,
            "dialog_state": dialog_state,
            "handoff_reasons": handoff_reasons,
            "answer_parts": answer_parts,
            "final_answer": final_answer,
            "messages": [*state["messages"], {"role": "assistant", "content": final_answer}],
            "escalation_draft": draft,
            "pending_side_effect": pending_side_effect,
            "trajectory_events": [
                *events,
                _event(
                    "hitl_gate",
                    "escalation",
                    "awaiting_confirmation",
                    {
                        "task_index": idx,
                        "tool": "prepare_side_effect",
                        "review_case_id": review_case_id,
                    },
                ),
            ],
        }

    async def finalize_answer(self, state: AgentState) -> dict:
        answers = list(state.get("answer_parts", []))
        final_answer = (
            "\n\n".join(answers) if answers else "没有识别到可执行的电商客服任务，请补充订单号或问题。"
        )
        rollup = _complex_progress_rollup(state)
        if rollup and rollup not in final_answer:
            final_answer = f"{final_answer}\n\n{rollup}" if final_answer else rollup
        if _is_customer_channel(state):
            final_answer = _strip_internal_answer_labels(final_answer)
        retrieved_policy = list(state.get("retrieved_policy", []))
        retrieved_insights = list(state.get("retrieved_insights", []))
        retrieved_support_docs = list(state.get("retrieved_support_docs", []))
        return {
            "final_answer": final_answer,
            "messages": [*state["messages"], {"role": "assistant", "content": final_answer}],
            "artifacts": {
                "policy_sources": [str(item["section_title"]) for item in retrieved_policy],
                "insight_sources": [
                    str(item.get("name") or item.get("category") or "")
                    for item in retrieved_insights
                    if item.get("name") or item.get("category")
                ],
                "support_sources": [str(item["doc_id"]) for item in retrieved_support_docs],
            },
            "trajectory_events": [
                *state.get("trajectory_events", []),
                _event("finalize_answer", "workflow", "completed", {"answer_parts": len(answers)}),
            ],
        }

    async def _answer_order_status(self, user_message: str, state: AgentState) -> str:
        task = self._task_extractor.fast_extract(user_message)
        if task is None:
            task = await self._task_extractor.extract(user_message)
        repair = repair_order_id(task.order_id or user_message)
        if not repair.ok:
            return repair.message
        result = await self._tool_manager.call(
            "get_order_status",
            {"order_id": repair.value},
            self._tool_context(state),
        )
        if result.ok:
            return str(result.data.get("answer", "订单事实工具没有返回结果。"))
        return f"订单事实工具调用失败：{result.error_message}"

    async def _prepare_escalation_from_text(
        self,
        user_message: str,
        action_type: str,
        state: AgentState,
    ) -> tuple[str, dict[str, object] | None]:
        # ``extract_slots`` has already resolved a selected order for native
        # controls. Reusing that verified value prevents a product regression
        # where a customer clicked "申请退款" on an order page but was asked to
        # type the same order id again.
        slot_order_id = str(dict(state.get("current_slots", {})).get("order_id") or "")
        repair = repair_order_id(slot_order_id) if slot_order_id else None
        if repair is None or not repair.ok:
            task = self._task_extractor.fast_extract(user_message)
            if task is None:
                task = await self._task_extractor.extract(user_message)
            repair = repair_order_id(task.order_id or user_message)
        if not repair.ok:
            return repair.message, None
        order_id = repair.value

        policy_sections = await self._search_policy_knowledge(
            f"{user_message}\n售后动作：{_ACTION_LABELS.get(action_type, action_type)}",
            state,
        )
        case_result = await self._tool_manager.call(
            "assess_after_sales_case",
            {
                "action_type": action_type,
                "order_id": order_id,
                "user_request": user_message,
                "policy_sections": policy_sections,
            },
            self._tool_context(state),
        )
        result = await self._tool_manager.call(
            "prepare_side_effect",
            {"order_id": order_id},
            self._tool_context(state),
        )
        raw_draft = result.data.get("draft") if result.ok else None
        draft = dict(raw_draft) if isinstance(raw_draft, dict) else None
        if draft is None:
            return "没有找到该订单，无法生成升级处理草稿。", None
        draft["action_type"] = action_type
        if case_result.ok:
            after_sales_case = case_result.data.get("case")
            decision = dict(case_result.data.get("decision", {}))
            verification = dict(case_result.data.get("verification", {}))
            customer_reply = str(case_result.data.get("customer_reply", ""))
            if isinstance(after_sales_case, dict):
                draft["after_sales_case"] = after_sales_case
            draft["decision"] = decision
            draft["verification"] = verification
            if customer_reply:
                draft["message_text"] = customer_reply
        else:
            decision = {}
            verification = {}
        draft["requires_confirmation"] = str(verification.get("required_next_step", "hitl")) == "hitl"

        if _decision_stops_execution(decision, verification):
            draft["requires_confirmation"] = False
            answer = _format_after_sales_decision_answer(
                order_id,
                action_type,
                draft,
                requires_confirmation=False,
                customer_view=_is_customer_channel(state),
            )
            return answer, draft

        # Order-state prohibitions are decided before asking for optional write
        # fields. A delivered order must receive the truthful rejection first;
        # collecting a new address for an impossible direct change is misleading.
        if action_type == "change_address":
            missing_address_fields = _missing_address_fields(user_message)
            if missing_address_fields:
                missing_text = "、".join(missing_address_fields)
                return (
                    f"订单 {order_id[:8]} 的改地址申请还需要补充：{missing_text}。"
                    "请提供完整的新收货信息后，我再为您提交处理。",
                    None,
                )

        action_label = _ACTION_LABELS.get(action_type, "售后升级处理")
        decision_block = _format_after_sales_decision_answer(
            order_id,
            action_type,
            draft,
            requires_confirmation=bool(draft.get("requires_confirmation")),
            customer_view=_is_customer_channel(state),
        )
        if draft.get("requires_confirmation"):
            if _is_customer_channel(state):
                answer = decision_block
            else:
                answer = f"我准备为订单 {order_id} 执行：{action_label}。\n\n{decision_block}"
        else:
            answer = f"订单 {order_id} 的{action_label}通过低风险自动执行门禁。\n\n{decision_block}"
        return answer, draft

    def await_confirmation(self, state: AgentState) -> dict:
        user_response: str = interrupt("Waiting for escalation confirmation")
        return {
            "user_response": user_response,
            "messages": [*state["messages"], {"role": "user", "content": user_response}],
        }

    async def finalize_escalation(self, state: AgentState) -> dict:
        user_response: str = state.get("user_response", "")
        normalized_response = user_response.lower().strip()
        confirmed: bool = normalized_response in (
            "yes",
            "yeah",
            "y",
            "confirm",
            "ok",
            "okay",
            "确认",
            "创建",
            "升级",
            "__review_approve__",
        )
        draft = state.get("escalation_draft", {})
        pending = state.get("pending_side_effect", {})
        task_index = _maybe_int(pending.get("task_index") or draft.get("task_index"))
        case_id = str(draft.get("review_case_id") or state.get("case_id") or "")
        executed = False
        receipt: dict[str, object] = {}
        if confirmed and draft:
            reviewer_override = (
                "after_sales_operator" if normalized_response == "__review_approve__" else None
            )
            answer, executed, receipt = await self._execute_escalation_draft(
                draft,
                state,
                role_override=reviewer_override,
                auth_scopes_override=["after_sales:write"] if reviewer_override else None,
            )
            status = "completed" if executed else "failed"
            if case_id:
                self._case_service.mark_case_result(
                    case_id,
                    "executed" if executed else "execution_failed",
                    reviewer_id=reviewer_override or "system",
                    receipt=receipt,
                )
            elif reviewer_override:
                self._case_service.mark_review_result(
                    str(state.get("session_id", "unknown")),
                    "executed",
                    reviewer_id=reviewer_override,
                )
        elif confirmed:
            answer = "没有找到可提交的升级草稿，请重新发起。"
            status = "failed"
        elif normalized_response == "__hitl_timeout__":
            answer = "待审核售后申请已超时自动取消；如果仍需处理，可以在原 case 上发起二次申诉。"
            status = "timeout_canceled"
            if case_id:
                self._case_service.mark_case_result(case_id, "timeout_canceled", reviewer_id="system_timeout")
            else:
                self._case_service.mark_review_result(
                    str(state.get("session_id", "unknown")),
                    "timeout_canceled",
                    reviewer_id="system_timeout",
                )
        else:
            answer = "售后申请已被审核拒绝；如果你有新的凭证或理由，可以在原 case 上发起二次申诉。"
            status = "canceled"
            if case_id:
                self._case_service.mark_case_result(case_id, "rejected", reviewer_id="after_sales_operator")
            else:
                self._case_service.mark_review_result(
                    str(state.get("session_id", "unknown")),
                    "rejected",
                    reviewer_id="after_sales_operator",
                )

        completed = list(state.get("completed_tasks", []))
        if task_index is not None:
            completed = _upsert_task_status(completed, task_index, "escalation", status)
        if not case_id and self._runtime_store is not None:
            await self._runtime_store.clear_pending_confirmation(str(state.get("session_id", "unknown")))

        return {
            "final_answer": answer,
            "messages": [*state["messages"], {"role": "assistant", "content": answer}],
            "completed_tasks": completed,
            "answer_parts": [*state.get("answer_parts", []), answer],
            "pending_side_effect": {},
            "escalation_draft": {},
            "case_id": case_id,
            "trajectory_events": [
                *state.get("trajectory_events", []),
                _event(
                    "finalize_escalation",
                    "escalation",
                    status,
                    {
                        "confirmed": confirmed,
                        "case_id": case_id,
                        "action_type": str(draft.get("action_type", "")) if draft else "",
                        "task_index": task_index,
                        "executed": executed,
                        "receipt": receipt,
                    },
                ),
            ],
        }

    async def _execute_escalation_draft(
        self,
        draft: dict[str, object],
        state: AgentState,
        *,
        role_override: str | None = None,
        auth_scopes_override: list[str] | None = None,
    ) -> tuple[str, bool, dict[str, object]]:
        action_type = str(draft.get("action_type", "open_support_case"))
        arguments = {
            "action_type": action_type,
            "order_id": str(draft["order_id"]),
            "message_text": str(draft["message_text"]),
        }
        case_id = str(draft.get("review_case_id") or state.get("case_id") or "")
        if case_id:
            arguments["case_id"] = case_id
        tool_result = await self._tool_manager.call(
            "execute_side_effect",
            arguments,
            self._tool_context(state, role=role_override, auth_scopes=auth_scopes_override),
        )
        action_label = _ACTION_LABELS.get(action_type, "售后升级处理")
        if not tool_result.ok:
            reason = tool_result.error_message or tool_result.error_code or "unknown error"
            return f"未执行{action_label}：{reason}。", False, {}
        result = tool_result.data["result"]
        if not isinstance(result, dict) or not result.get("result_id"):
            return f"未执行{action_label}：工具没有返回有效 receipt。", False, {}
        duplicate_hint = "（重复请求，已返回已有结果）" if result.get("duplicate") else ""
        receipt = {
            "result_id": str(result.get("result_id", "")),
            "duplicate": bool(result.get("duplicate")),
            "order_event": result.get("order_event"),
        }
        return f"已执行{action_label}：{result['result_id']}。{duplicate_hint}", True, receipt

    @staticmethod
    def _infer_action_type(text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("refund", "退款", "赔付", "compensat", "coupon", "发券")):
            return "refund_request"
        if any(token in lowered for token in ("cancel order", "cancel purchase", "取消订单")):
            return "cancel_order"
        if any(
            token in lowered for token in ("change shipping address", "change address", "改地址", "修改地址")
        ):
            return "change_address"
        if any(token in lowered for token in ("invoice", "发票", "开票")):
            return "invoice_request"
        if any(token in lowered for token in ("complaint", "投诉", "升级投诉", "转人工", "人工介入")):
            return "complaint_escalation"
        return "open_support_case"

    async def _category_insights(self, query: str, state: AgentState) -> list[dict[str, object]]:
        result = await self._tool_manager.call(
            "search_category_risk",
            {"query": query},
            self._tool_context(state),
        )
        return list(result.data.get("insights", [])) if result.ok else []

    async def _after_sales_priority_report(self, query: str, state: AgentState) -> dict[str, object]:
        result = await self._tool_manager.call(
            "generate_after_sales_priority_report",
            {"query": query},
            self._tool_context(state),
        )
        if result.ok:
            return dict(result.data.get("report", {}))
        return {
            "summary": f"权限不足或运营报表工具不可用：{result.error_message or result.error_code}",
            "high_risk_categories": [],
            "priority_orders": [],
        }

    async def _search_policy_knowledge(
        self,
        query: str,
        state: AgentState,
        k: int = 3,
    ) -> list[dict[str, object]]:
        result = await self._tool_manager.call(
            "search_policy_knowledge",
            {"query": query, "k": k},
            self._tool_context(state),
        )
        return list(result.data.get("sections", [])) if result.ok else []

    async def _search_support_docs(
        self,
        query: str,
        state: AgentState,
        k: int = 3,
    ) -> list[dict[str, object]]:
        result = await self._tool_manager.call(
            "search_support_examples",
            {"query": query, "k": k},
            self._tool_context(state),
        )
        return list(result.data.get("docs", [])) if result.ok else []

    @staticmethod
    def _tool_context(
        state: AgentState,
        role: str | None = None,
        auth_scopes: list[str] | None = None,
    ) -> ToolCallContext:
        return ToolCallContext(
            session_id=state.get("session_id", "unknown"),
            tenant_id=state.get("tenant_id", "olist-demo"),
            user_id=state.get("user_id", "demo-user"),
            role=role or state.get("role", "support_agent"),
            auth_scopes=list(auth_scopes if auth_scopes is not None else state.get("auth_scopes", [])),
            allowed_order_ids=list(state.get("allowed_order_ids", [])),
        )


def _missing_address_fields(text: str) -> list[str]:
    """Return the minimum recipient fields needed for an address-change request.

    The action is a request to a downstream fulfilment system. A user can ask
    whether address change is possible without these fields, but creating the
    request requires enough data for an operator or OMS to act on it.
    """
    return missing_address_fields(text)


def _decision_stops_execution(decision: dict[str, object], verification: dict[str, object]) -> bool:
    outcome = str(decision.get("outcome", ""))
    next_step = str(verification.get("required_next_step", ""))
    return outcome in {"reject", "ask_clarification"} or next_step in {"clarify", "stop"}


def _control_ablation_decision(
    mode: str,
    action_type: str,
    original: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Construct intentionally weaker controls for a matched ablation.

    These modes exist only in the evaluation runner. They retain the same
    planner, tools, facts, and policy retrieval as `full`, then remove the
    decision/verifier behavior to quantify its business effect.
    """
    policy_refs = list(original.get("policy_refs", []))
    evidence = list(original.get("evidence", []))
    if mode == "handoff_all":
        decision = {
            "outcome": "needs_human_review",
            "action_type": action_type,
            "reason_code": "ablation_handoff_all",
            "confidence": 1.0,
            "risk_level": "medium",
            "requires_human": True,
            "handoff_reasons": ["ablation_all_writes_handoff"],
            "allowed_actions": [action_type],
            "blocked_actions": [],
            "evidence": evidence,
            "policy_refs": policy_refs,
            "customer_message_points": ["所有动作统一交由人工审核。"],
        }
        return decision, {"passed": True, "flags": [], "required_next_step": "hitl"}

    decision = {
        "outcome": "approve",
        "action_type": action_type,
        "reason_code": "ablation_execute_on_intent",
        "confidence": 1.0,
        "risk_level": "low",
        "requires_human": False,
        "handoff_reasons": [],
        "allowed_actions": [action_type],
        "blocked_actions": [],
        "evidence": evidence,
        "policy_refs": policy_refs,
        "customer_message_points": ["检测到动作意图后直接执行。"],
    }
    return decision, {"passed": True, "flags": [], "required_next_step": "execute"}


def _format_after_sales_decision_answer(
    order_id: str,
    action_type: str,
    draft: dict[str, object],
    *,
    requires_confirmation: bool,
    customer_view: bool = False,
) -> str:
    decision = dict(draft.get("decision", {}))
    verification = dict(draft.get("verification", {}))
    action_label = _ACTION_LABELS.get(action_type, "售后处理")
    outcome = str(decision.get("outcome", "needs_human_review"))
    risk = str(decision.get("risk_level", "medium"))
    confidence = decision.get("confidence", "")
    refs = decision.get("policy_refs", [])
    refs_text = "、".join(str(ref) for ref in refs[:3]) if isinstance(refs, list) else ""
    evidence = decision.get("evidence", [])
    evidence_text = "；".join(str(item) for item in evidence[:4]) if isinstance(evidence, list) else ""
    flags = verification.get("flags", [])
    flags_text = "；".join(str(flag) for flag in flags) if isinstance(flags, list) and flags else "无"
    next_step = str(verification.get("required_next_step", "hitl"))

    if customer_view:
        return _format_customer_after_sales_answer(
            order_id=order_id,
            action_label=action_label,
            outcome=outcome,
            next_step=next_step,
            draft=draft,
            requires_confirmation=requires_confirmation,
        )

    lines = [
        "售后 case 决策：",
        f"- 动作：{action_label}",
        f"- 结论：{outcome}；风险等级：{risk}；置信度：{confidence}",
        f"- 证据：{evidence_text}",
        f"- 政策依据：{refs_text or '未命中明确政策，需谨慎处理'}",
        f"- Verifier：next_step={next_step}；flags={flags_text}",
        "",
        f"客户回复草稿：{draft['message_text']}",
    ]
    if requires_confirmation:
        lines.extend(
            [
                "",
                "该动作会改变业务状态，必须经过 HITL 确认后才会调用企业工具。",
                "是否确认执行？(yes/no)",
            ]
        )
    elif next_step == "execute":
        lines.extend(
            [
                "",
                "该动作满足低风险自动执行门禁，系统会继续调用企业写工具并记录审计日志。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"因此本轮不会执行 {order_id} 的副作用工具；可补充信息后重新发起。",
            ]
        )
    return "\n".join(lines)


def _is_customer_channel(state: AgentState) -> bool:
    return state.get("channel") == "customer_self_service" or state.get("role") == "customer"


def _format_customer_after_sales_answer(
    *,
    order_id: str,
    action_label: str,
    outcome: str,
    next_step: str,
    draft: dict[str, object],
    requires_confirmation: bool,
) -> str:
    reply = str(draft.get("message_text", "")).strip().replace("。。", "。")
    short_order_id = order_id[:8]

    if outcome == "reject" or next_step == "stop":
        return (
            f"订单 {short_order_id} 的{action_label}暂时无法直接办理。\n"
            f"{reply or '我已经根据订单状态和售后规则完成核查，如需继续处理可以补充新的凭证或诉求。'}"
        )
    if outcome == "ask_clarification" or next_step == "clarify":
        return (
            f"订单 {short_order_id} 的{action_label}还需要补充信息。\n"
            f"{reply or '请补充问题描述、期望处理方式或必要凭证，我会继续帮你生成售后申请。'}"
        )
    if requires_confirmation:
        return (
            f"已为订单 {short_order_id} 生成售后处理单：{action_label}，并提交给售后人员审核。\n"
            f"{reply or '工作人员会结合订单事实、物流状态和平台政策复核后处理。'}\n"
            "该类请求涉及退款、取消、改地址或投诉升级，系统会先冻结为待审核 case，审核通过后再调用企业工具。"
        )
    return (
        f"订单 {short_order_id} 的{action_label}已通过低风险自动处理门禁。\n"
        f"{reply or '系统会继续完成后续处理，并保留审计记录。'}"
    )


_ACTION_LABELS = {
    "open_support_case": "创建售后工单",
    "refund_request": "提交退款/补偿申请",
    "cancel_order": "提交取消订单申请",
    "change_address": "提交改地址申请",
    "invoice_request": "提交发票申请",
    "complaint_escalation": "提交投诉升级工单",
}


def _answer_part(intent: str, answer: str, state: AgentState) -> str:
    """Keep operational labels in staff/debug flows, not customer replies."""

    if _is_customer_channel(state):
        return answer.strip()
    return f"[{_INTENT_LABELS.get(intent, intent)}]\n{answer}"


def _strip_internal_answer_labels(answer: str) -> str:
    """Defence in depth for legacy branches that still emit a labelled part."""

    labels = "咨询|信息确认|订单查询|我的订单|订单小结|运营分析|审核台优先处理|政策问答|售后处理|售后升级"
    return re.sub(rf"(?m)^\[(?:{labels})\]\s*", "", answer).strip()


_INTENT_LABELS = {
    "small_talk": "咨询",
    "clarify": "信息确认",
    "order_status": "订单查询",
    "customer_orders": "我的订单",
    "customer_profile": "订单小结",
    "qa": "运营分析",
    "ops_decision": "审核台优先处理",
    "policy": "政策问答",
    "escalation": "售后处理",
}


def _selected_order_action_type(message: str, state: AgentState) -> str | None:
    """Keep a selected-order action deterministic even when the LLM says clarify.

    The product already knows which order the customer is looking at.  A
    free-form "cancel this order" must not depend on the planner recognising
    the colloquial phrase; policy/verifier/HITL still decide whether it is
    allowed to execute.
    """
    page_context = state.get("page_context") or {}
    if not isinstance(page_context, dict) or not page_context.get("selected_order_id"):
        return None
    text = message.lower()
    if any(marker in text for marker in ("政策", "规则", "条件", "能不能", "可以吗", "多久")):
        return None
    if "退款" in text or "退货" in text:
        return "refund_request"
    if "取消" in text:
        return "cancel_order"
    if "地址" in text and any(marker in text for marker in ("改", "修改", "换")):
        return "change_address"
    if "发票" in text:
        return "invoice_request"
    if "投诉" in text or "申请售后" in text:
        return "complaint_escalation"
    if any(token in text for token in ("售后", "处理单", "工单", "建单", "升级处理")):
        return "open_support_case"
    return None


def _structured_action_plan(action: str, message: str) -> list[dict[str, object]]:
    """Translate a native order-page control into a deterministic task.

    Standard order actions should not depend on the planner recognising the
    label of a button that the product itself rendered.  Free-form messages
    still use the LLM planner; this path simply carries the UI intent into the
    same retrieval, decision, verifier, HITL, and audit workflow.
    """

    normalized = action.strip().lower()
    if normalized in {"track_order", "order_status"}:
        return [{"intent": "order_status", "text": message, "side_effect": False, "depends_on": []}]
    if normalized in {"list_active_shipments", "list_attention_orders"}:
        return [
            {
                "intent": "customer_orders",
                "text": message,
                "side_effect": False,
                "depends_on": [],
                "order_filter": "in_transit" if normalized == "list_active_shipments" else "attention",
            }
        ]
    if normalized in {"order_summary", "customer_profile"}:
        return [{"intent": "customer_profile", "text": message, "side_effect": False, "depends_on": []}]
    if normalized in {"refund_policy", "after_sales_policy"}:
        return [{"intent": "policy", "text": message, "side_effect": False, "depends_on": []}]
    action_map = {
        "refund_request": "refund_request",
        "cancel_order": "cancel_order",
        "change_address": "change_address",
        "invoice_request": "invoice_request",
        "complaint_escalation": "complaint_escalation",
        "open_support_case": "open_support_case",
    }
    action_type = action_map.get(normalized)
    if action_type:
        return [
            {
                "intent": "escalation",
                "text": message,
                "side_effect": True,
                "depends_on": [],
                "action_type": action_type,
            }
        ]
    return [{"intent": "clarify", "text": message, "side_effect": False, "depends_on": []}]


def _format_owned_orders(orders: list[dict[str, object]], filter_name: str) -> str:
    if not orders:
        if filter_name == "in_transit":
            return "当前没有正在运输中的订单。"
        if filter_name == "attention":
            return "当前没有需要特别留意的订单。"
        return "当前账号下没有可展示的订单。"
    if filter_name == "in_transit":
        lines = [f"我查到 {len(orders)} 笔订单还在处理中："]
    elif filter_name == "attention":
        lines = [f"我查到 {len(orders)} 笔订单建议优先留意："]
    else:
        lines = [f"我查到您有 {len(orders)} 笔订单："]
    for order in orders[:5]:
        status = _customer_order_status_label(str(order.get("status", "")))
        category = _customer_category_label(str(order.get("category", "商品")))
        delay = order.get("delay_days")
        suffix = f"，预计延迟 {delay} 天" if isinstance(delay, int) and delay > 0 else ""
        lines.append(f"- {category}：{status}{suffix}")
    if filter_name == "in_transit":
        lines.append("需要的话，您可以选择其中一笔，我继续帮您查看进度或处理售后。")
    return "\n".join(lines)


def _format_customer_order_summary(orders: list[dict[str, object]]) -> str:
    """Summarise account-visible purchase categories without inferring user traits."""

    if not orders:
        return "我暂时没有查到可供参考的订单记录。"
    categories: list[str] = []
    for order in orders:
        category = _customer_category_label(str(order.get("category", "商品")))
        if category not in categories:
            categories.append(category)
    active_count = sum(
        str(order.get("status", "")) in {"approved", "invoiced", "processing", "shipped"} for order in orders
    )
    category_text = "、".join(categories[:4])
    more = "等" if len(categories) > 4 else ""
    progress = f"目前有 {active_count} 笔订单仍在处理中。" if active_count else "目前没有正在处理的订单。"
    return (
        f"从您最近的 {len(orders)} 笔订单看，您买过 {category_text}{more}。{progress}"
        "需要的话，我可以继续帮您查看在途订单，或处理某一笔订单的售后。"
    )


def _customer_category_label(category: str) -> str:
    return category_label(category)


def _customer_order_status_label(status: str) -> str:
    labels = {
        "created": "待付款",
        "approved": "已付款",
        "invoiced": "待发货",
        "processing": "处理中",
        "shipped": "运输中",
        "delivered": "已送达",
        "canceled": "已取消",
    }
    return labels.get(status, status or "处理中")


def _execution_order(tasks: list[dict[str, object]]) -> list[tuple[int, dict[str, object]]]:
    indexed = list(enumerate(tasks))
    read_only = [item for item in indexed if not item[1].get("side_effect")]
    side_effects = [item for item in indexed if item[1].get("side_effect")]
    return [*read_only, *side_effects]


def _with_order_context(task_text: str, full_message: str) -> str:
    if repair_order_id(task_text).ok:
        return task_text
    repair = repair_order_id(full_message)
    if not repair.ok:
        return task_text
    return f"{task_text}\n上下文订单号：{repair.value}"


def _hitl_timeout_seconds() -> int:
    raw = os.environ.get("HITL_TIMEOUT_SECONDS", "900")
    try:
        value = int(raw)
    except ValueError:
        return 900
    return max(value, 1)


def _latest_user_task_message(state: AgentState) -> str:
    confirmation_replies = {
        "yes",
        "yeah",
        "y",
        "confirm",
        "ok",
        "okay",
        "no",
        "n",
        "cancel",
        "确认",
        "创建",
        "升级",
        "取消",
        "不用",
        "不要",
        "算了",
        "放弃",
        "否",
        "__hitl_timeout__",
    }
    for message in reversed(state.get("messages", [])):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        if content.lower().strip() not in confirmation_replies:
            return content
    return str(state["messages"][-1]["content"])


def _complex_progress_rollup(state: AgentState) -> str:
    analysis = dict(state.get("request_analysis") or {})
    if analysis.get("tier") != "complex" or int(analysis.get("sub_task_count") or 0) < 2:
        return ""
    tasks = list(state.get("task_plan") or [])
    write_indices = {index for index, task in enumerate(tasks) if bool(task.get("side_effect"))}
    completed = list(state.get("completed_tasks") or [])
    processed = 0
    pending = 0
    for row in completed:
        index = _maybe_int(row.get("index"))
        if index is None or index not in write_indices:
            continue
        status = str(row.get("status") or "")
        if status == "completed":
            processed += 1
        elif status in {"delegated_pending", "awaiting_confirmation"}:
            pending += 1
    total = len(write_indices)
    if total < 2:
        return ""
    return (
        f"本次复杂请求已拆成 {total} 个子申请："
        f"已处理 {processed} 个，待审核 {pending} 个；高风险子申请不会自动执行。"
    )


def _request_id(state: AgentState, message: str) -> str:
    seed = f"{state.get('session_id', '')}:{_turn_count(state)}:{message}"
    return f"req-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:10]}"


def _single_request_control_task(message: str, analysis) -> list[dict[str, object]]:
    """Evaluation-only control: collapse the request to one disposition."""
    write_units = list(analysis.write_units)
    if write_units:
        unit = write_units[0]
        return [
            {
                "intent": "escalation",
                "text": message,
                "side_effect": True,
                "action_type": unit.action_type,
                "order_filter": "all",
                "order_id": unit.order_id,
                "order_id_scope": unit.order_scope,
                "risk_level": unit.risk_level,
                "risk_reasons": list(unit.risk_reasons),
                "auto_eligible": unit.auto_eligible,
                "evidence_complete": unit.evidence_complete,
                "missing_evidence": list(unit.missing_evidence),
                "depends_on": [],
                "sub_task_index": 1,
                "sub_task_total": 1,
            }
        ]
    intent = analysis.units[0].intent if analysis.units else "clarify"
    return [
        {
            "intent": intent,
            "text": message,
            "side_effect": False,
            "action_type": "none",
            "order_filter": "all",
            "depends_on": [],
            "sub_task_index": 1,
            "sub_task_total": 1,
        }
    ]


def _enrich_tasks_with_analysis(
    tasks: list[dict[str, object]], analysis, *, complex_request: bool
) -> list[dict[str, object]]:
    """Attach per-sub-request risk to LLM/fallback planner tasks.

    The planner is good at intent wording but does not classify risk.  Without
    this enrichment a simple complaint would inherit only the local decision
    engine's medium risk and could miss the high-risk policy gate.
    """
    write_units = list(analysis.write_units)
    enriched: list[dict[str, object]] = []
    for task in tasks:
        current = dict(task)
        if current.get("side_effect") and not current.get("risk_level"):
            action_type = str(current.get("action_type") or "")
            if not complex_request and action_type != "complaint_escalation":
                enriched.append(current)
                continue
            order_id = str(current.get("order_id") or "").lower()
            match = next(
                (
                    unit
                    for unit in write_units
                    if unit.action_type == action_type
                    and (not order_id or not unit.order_id or unit.order_id == order_id)
                ),
                None,
            )
            if match is not None:
                current.setdefault("order_id", match.order_id)
                current.setdefault("order_id_scope", match.order_scope)
                current["risk_level"] = match.risk_level
                current["risk_reasons"] = list(match.risk_reasons)
                current["auto_eligible"] = match.auto_eligible
                current["evidence_complete"] = match.evidence_complete
                current["missing_evidence"] = list(match.missing_evidence)
        enriched.append(current)
    return enriched


def _structured_request_summary(analysis, action_type: str, planning_mode: str) -> dict[str, object]:
    """A structured page action is one explicit request by construction."""
    summary = analysis.summary()
    summary.update(
        {
            "tier": "simple",
            "decomposition_mode": planning_mode,
            "sub_task_count": 1,
            "write_task_count": 1,
            "read_task_count": 0,
            "action_types": [action_type],
            "intent_count": 1,
        }
    )
    return summary


def _completed_task_indices(completed: list[dict[str, object]]) -> set[int]:
    return {
        int(item["index"])
        for item in completed
        if item.get("status") == "completed" and _maybe_int(item.get("index")) is not None
    }


def _terminal_task_indices(completed: list[dict[str, object]]) -> set[int]:
    terminal = {"completed", "canceled", "timeout_canceled", "failed", "blocked", "delegated_pending"}
    return {
        int(item["index"])
        for item in completed
        if item.get("status") in terminal and _maybe_int(item.get("index")) is not None
    }


def _upsert_task_status(
    completed: list[dict[str, object]],
    index: int,
    intent: str,
    status: str,
) -> list[dict[str, object]]:
    row = {"index": index, "intent": intent, "status": status}
    updated = []
    replaced = False
    for item in completed:
        if item.get("index") == index:
            updated.append(row)
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append(row)
    return updated


def _should_include_previous_answer(state: AgentState) -> bool:
    events = state.get("trajectory_events", [])
    return bool(state.get("final_answer") and events and events[-1].get("node") == "finalize_escalation")


def _maybe_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _turn_count(state: AgentState) -> int:
    return sum(1 for message in state.get("messages", []) if message.get("role") == "user")


def _merge_dialog_state(
    state: AgentState,
    *,
    current_intent: str | None = None,
    task_count: int | None = None,
    turn_count: int | None = None,
    slots: dict[str, object] | None = None,
    missing_slots: list[str] | None = None,
    retrieved_policy_count: int | None = None,
    retrieved_support_count: int | None = None,
    retrieved_insight_count: int | None = None,
    handoff_reasons: list[str] | None = None,
    risk_level: str | None = None,
    decision_outcome: str | None = None,
    verifier_next_step: str | None = None,
) -> dict[str, object]:
    dialog_state = dict(state.get("dialog_state", {}))
    filled_slots = dict(dialog_state.get("filled_slots", state.get("filled_slots", {})))
    if slots:
        for key, value in slots.items():
            if value not in ("", None, [], {}):
                filled_slots[key] = value
    active_order_id = str(
        filled_slots.get("order_id")
        or dialog_state.get("active_order_id")
        or state.get("active_order_id", "")
    )
    merged_handoff = list(dialog_state.get("handoff_reasons", state.get("handoff_reasons", [])))
    if handoff_reasons:
        merged_handoff.extend(handoff_reasons)

    next_missing_slots = missing_slots if missing_slots is not None else dialog_state.get("missing_slots", [])
    next_turn_count = (
        turn_count if turn_count is not None else dialog_state.get("turn_count", _turn_count(state))
    )
    dialog_state.update(
        {
            "current_intent": current_intent or dialog_state.get("current_intent", ""),
            "active_order_id": active_order_id,
            "filled_slots": filled_slots,
            "missing_slots": list(next_missing_slots),
            "handoff_reasons": list(dict.fromkeys(str(item) for item in merged_handoff if item)),
            "turn_count": int(next_turn_count),
            "failure_count": int(dialog_state.get("failure_count", state.get("failure_count", 0))),
        }
    )
    if task_count is not None:
        dialog_state["task_count"] = task_count
    if retrieved_policy_count is not None:
        dialog_state["retrieved_policy_count"] = retrieved_policy_count
    if retrieved_support_count is not None:
        dialog_state["retrieved_support_count"] = retrieved_support_count
    if retrieved_insight_count is not None:
        dialog_state["retrieved_insight_count"] = retrieved_insight_count
    if risk_level:
        dialog_state["risk_level"] = risk_level
    if decision_outcome:
        dialog_state["decision_outcome"] = decision_outcome
    if verifier_next_step:
        dialog_state["verifier_next_step"] = verifier_next_step
    return dialog_state


def _missing_slots(intent: str, slots: dict[str, object]) -> list[str]:
    missing: list[str] = []
    if intent in {"order_status", "escalation"} and not slots.get("order_id"):
        missing.append("order_id")
    if intent == "escalation" and not slots.get("action_type"):
        missing.append("action_type")
    if intent == "qa" and not slots.get("category"):
        missing.append("category")
    return missing


def _event(
    node: str,
    intent: str,
    status: str,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "node": node,
        "intent": intent,
        "status": status,
        "details": details or {},
    }


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
