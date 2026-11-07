from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.intent.decomposer import decompose_business_message
from app.llm.json_fallback import add_json_instruction, native_structured_output_enabled, parse_json_model
from app.llm.prompts import INTENT_PLANNER_PROMPT
from app.llm.types import IntentRouteResult, PlannedTask, TaskPlanResult

VALID_INTENTS = {
    "small_talk", "clarify", "customer_profile", "customer_orders", "qa", "order_status", "policy",
    "ops_decision", "escalation",
}
logger = logging.getLogger(__name__)


class IntentPlanner:
    """Plan ordered business tasks from a user request."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._base_llm = llm
        self._llm = (
            llm.with_structured_output(TaskPlanResult)
            if native_structured_output_enabled(llm)
            else None
        )

    async def plan(self, message: str) -> TaskPlanResult:
        fast_plan = _fast_plan(message)
        if fast_plan is not None:
            return fast_plan
        messages = [
            SystemMessage(content=INTENT_PLANNER_PROMPT),
            HumanMessage(content=message),
        ]
        try:
            if self._llm is None:
                raw = await self._base_llm.ainvoke(add_json_instruction(messages, TaskPlanResult))
                result = parse_json_model(raw, TaskPlanResult)
            else:
                result = await self._llm.ainvoke(messages)
        except Exception as exc:
            logger.warning("Structured planner failed, trying JSON-text fallback: %s", exc)
            if _is_transport_error(exc):
                logger.warning("Planner transport error; using deterministic task plan without retry")
                return fallback_plan(message)
            try:
                raw = await self._base_llm.ainvoke(add_json_instruction(messages, TaskPlanResult))
                result = parse_json_model(raw, TaskPlanResult)
            except Exception as fallback_exc:
                logger.warning("LLM intent planner failed, using deterministic fallback: %s", fallback_exc)
                return fallback_plan(message)
        tasks = _dedupe_redundant_tasks([task for task in result.tasks if task.intent in VALID_INTENTS])
        tasks = _ensure_explicit_read_requirements(message, tasks)
        if not tasks:
            return fallback_plan(message)
        return TaskPlanResult(tasks=_normalize_dependencies(tasks), planning_mode="llm")

    async def classify(self, message: str) -> IntentRouteResult:
        """Compatibility helper for evals that still measure single-label routing."""
        plan = await self.plan(message)
        intent = plan.tasks[0].intent if plan.tasks else "policy"
        return IntentRouteResult(intent=intent)


def fallback_plan(message: str) -> TaskPlanResult:
    tasks = [
        PlannedTask(
            intent=task.intent,
            text=task.text,
            side_effect=task.side_effect,
            action_type=task.action_type,
            order_filter=task.order_filter,
            depends_on=[],
        )
        for task in decompose_business_message(message)
    ]
    action_type = _direct_action_type(message.lower())
    if action_type and not any(task.intent == "escalation" for task in tasks):
        # The fallback must never turn an explicit write request into a
        # read-only policy answer merely because the model is unavailable.
        # Keep an explicit rule/condition question if present; otherwise the
        # action is the complete user request.
        if not any(token in message.lower() for token in ("政策", "规则", "条件", "资料", "材料")):
            tasks = [task for task in tasks if task.intent != "policy"]
        tasks.append(
            PlannedTask(
                intent="escalation",
                text=message,
                side_effect=True,
                action_type=action_type,
                depends_on=[],
            )
        )
    return TaskPlanResult(
        tasks=_normalize_dependencies(tasks),
        planning_mode="deterministic_fallback",
    )


def _fast_plan(message: str) -> TaskPlanResult | None:
    """Use a deterministic fast lane only for unambiguous one-step requests."""

    text = message.lower()
    if _is_customer_profile_request(text):
        return TaskPlanResult(
            tasks=[PlannedTask(intent="customer_profile", text=message)],
            planning_mode="deterministic_fast_path",
        )
    if _needs_business_clarification(text):
        return TaskPlanResult(
            tasks=[PlannedTask(intent="clarify", text=message)],
            planning_mode="deterministic_fast_path",
        )
    order_filter = _customer_orders_filter(text)
    if order_filter is not None:
        return TaskPlanResult(
            tasks=[PlannedTask(intent="customer_orders", text=message, order_filter=order_filter)],
            planning_mode="deterministic_fast_path",
        )
    if _is_small_talk(text):
        return TaskPlanResult(
            tasks=[PlannedTask(intent="small_talk", text=message)],
            planning_mode="deterministic_fast_path",
        )
    has_order_id = bool(re.search(r"[0-9a-f]{32}", text))
    # Status-confirmation wording contains an action verb but asks about a
    # completed state. Route it to the read tool before generic action rules.
    if has_order_id and _is_status_confirmation(text):
        return TaskPlanResult(
            tasks=[PlannedTask(intent="order_status", text=message)],
            planning_mode="deterministic_fast_path",
        )
    action_type = _direct_action_type(text)
    compound_markers = ("并且", "然后", "同时", "先", "以及")
    # A compound request needs the planner even when its action is expressed
    # indirectly (for example, "不想等了，帮我把手续办掉").  Returning an
    # order-status fast path here would silently drop the write intent.
    if any(marker in text for marker in compound_markers) or (
        has_order_id and not action_type and _needs_llm_disambiguation(text)
    ):
        return None
    # Direct write requests can stay on the low-latency lane only when they
    # are genuinely one-step.  A request that also asks for facts or policy
    # needs the planner to preserve each user-visible obligation.
    if has_order_id and action_type and _requires_action_planning(text):
        return None
    if has_order_id and action_type:
        return TaskPlanResult(
            tasks=[
                PlannedTask(
                    intent="escalation",
                    text=message,
                    side_effect=True,
                    action_type=action_type,
                )
            ],
            planning_mode="deterministic_fast_path",
        )
    if has_order_id and not action_type:
        return TaskPlanResult(
            tasks=[PlannedTask(intent="order_status", text=message)],
            planning_mode="deterministic_fast_path",
        )

    policy_topics = ("退款", "退货", "取消", "发票", "地址", "补偿", "配送", "手续费")
    policy_cues = ("政策", "规则", "条件", "手续费", "怎么", "如何", "多久", "能不能", "可以吗", "是否")
    if (
        not has_order_id
        and not any(marker in text for marker in compound_markers)
        and any(topic in text for topic in policy_topics)
        and any(cue in text for cue in policy_cues)
    ):
        return TaskPlanResult(
            tasks=[PlannedTask(intent="policy", text=message)],
            planning_mode="deterministic_fast_path",
        )
    return None


def _is_small_talk(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？!?~～]+", "", text)
    return normalized in {
        "你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "早上好", "晚上好",
        "谢谢", "感谢", "多谢", "再见", "拜拜", "你能做什么", "你可以做什么", "帮助",
    }


def _is_customer_profile_request(text: str) -> bool:
    """Recognise account-history summaries without mistaking an after-sales policy question for one."""

    normalized = re.sub(r"[\s，。！？!?~～]+", "", text)
    if any(term in normalized for term in ("退款", "取消", "地址", "发票", "投诉", "物流")):
        return False
    return any(
        phrase in normalized
        for phrase in (
            "我喜欢什么样的产品", "我喜欢什么产品", "我买过什么", "我买了什么",
            "我的购买记录", "我的订单小结", "我的购物偏好", "我的购物习惯", "我的消费习惯",
            "我平时买什么", "我经常买什么",
        )
    )


def _customer_orders_filter(text: str) -> str | None:
    """Recognise account-level order overviews before single-order routing."""

    normalized = re.sub(r"[\s，。！？!?~～]+", "", text)
    if "订单" not in normalized:
        return None
    if any(cue in normalized for cue in ("在运输中", "运输中的", "在途", "配送中", "待发货", "还没发货")):
        return "in_transit"
    if any(cue in normalized for cue in ("待处理", "需要处理", "需要关注", "要留意")):
        return "attention"
    if any(cue in normalized for cue in ("有哪些订单", "我的订单", "全部订单", "所有订单")):
        return "all"
    return None


def _needs_business_clarification(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？!?~～]+", "", text)
    if normalized in {
        "帮我处理订单", "帮我处理一下", "订单有问题", "我的订单有问题", "我要售后",
        "订单不太对", "我的订单不太对", "售后怎么处理", "我需要帮助", "帮帮我", "怎么办", "怎么弄",
    }:
        return True
    return (
        "订单有问题" in normalized or "订单不太对" in normalized
    ) and any(word in normalized for word in ("处理", "帮", "售后", "看看"))


def _is_status_confirmation(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "是否已经取消",
            "是否已取消",
            "是不是已经取消",
            "是不是已取消",
            "是否取消成功",
            "取消了吗",
            "取消了没",
            "取消成功了吗",
            "取消状态",
        )
    )


def _direct_action_type(text: str) -> str:
    has_order_id = bool(re.search(r"[0-9a-f]{32}", text))
    if "退款" in text or "补偿申请" in text or "refund" in text:
        return "refund_request"
    if (
        "取消订单" in text
        or "申请取消" in text
        or "终止订单" in text
        or "撤销订单" in text
        or "cancel order" in text
        or (
            has_order_id
            and any(word in text for word in ("取消", "终止", "撤销", "cancel"))
        )
    ):
        return "cancel_order"
    address_change_words = ("change", "update", "改", "调整")
    if (
        "改地址" in text
        or "修改地址" in text
        or ("地址" in text and "修改" in text)
        or (
            has_order_id
            and ("address" in text or "地址" in text)
            and any(word in text for word in address_change_words)
        )
    ):
        return "change_address"
    invoice_words = ("申请", "开", "request", "issue")
    if (
        "申请发票" in text
        or "开票申请" in text
        or (
            has_order_id
            and ("invoice" in text or "发票" in text)
            and any(word in text for word in invoice_words)
        )
    ):
        return "invoice_request"
    if "投诉" in text or (has_order_id and ("complaint" in text or "升级" in text)):
        return "complaint_escalation"
    return ""


def _needs_llm_disambiguation(text: str) -> bool:
    """Keep terse factual reads fast; send implied after-sales language to the planner."""

    signals = (
        "地址", "收货", "票据", "报销", "开具", "取消手续", "不想等",
        "不满意", "进一步处理", "售后", "怎么办", "怎么处理", "核对",
        "政策", "规则", "条件", "资料", "材料",
    )
    return any(signal in text for signal in signals)


def _requires_action_planning(text: str) -> bool:
    """Detect reads that must be preserved before an otherwise direct write."""

    signals = (
        "先", "然后", "同时", "查", "核对", "物流", "状态", "政策", "规则", "条件", "限制",
        "explain", "policy", "status", "track",
    )
    return any(signal in text for signal in signals)


def _is_transport_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(marker in name for marker in ("timeout", "connection", "network"))


def _normalize_dependencies(tasks: list[PlannedTask]) -> list[PlannedTask]:
    order_status_indices = [idx for idx, task in enumerate(tasks) if task.intent == "order_status"]
    if not order_status_indices:
        return tasks
    first_order_idx = order_status_indices[0]
    normalized = []
    for idx, task in enumerate(tasks):
        if task.intent == "escalation" and idx > first_order_idx and first_order_idx not in task.depends_on:
            normalized.append(task.model_copy(update={"depends_on": [*task.depends_on, first_order_idx]}))
        else:
            normalized.append(task)
    return normalized


def _dedupe_redundant_tasks(tasks: list[PlannedTask]) -> list[PlannedTask]:
    deduped: list[PlannedTask] = []
    seen_read_only_without_order: set[str] = set()
    for task in tasks:
        if not task.side_effect and not _contains_order_id(task.text):
            if task.intent in seen_read_only_without_order:
                continue
            seen_read_only_without_order.add(task.intent)
        deduped.append(task)
    return deduped


def _ensure_explicit_read_requirements(message: str, tasks: list[PlannedTask]) -> list[PlannedTask]:
    """Preserve a stated policy question when the planner also emits a write task.

    The policy task is a requirement of the user's request, not a fallback
    answer decoration.  It must therefore run before a side-effect task even
    if the model only emitted the action.
    """

    asks_for_policy = any(token in message.lower() for token in (
        "政策", "规则", "条件", "资料", "材料", "需要什么", "怎么处理", "如何处理",
    ))
    has_write = any(task.side_effect or task.intent == "escalation" for task in tasks)
    has_policy = any(task.intent == "policy" for task in tasks)
    if not asks_for_policy or not has_write or has_policy:
        return tasks

    policy_task = PlannedTask(intent="policy", text=message)
    first_write = next(
        (index for index, task in enumerate(tasks) if task.side_effect or task.intent == "escalation"),
        len(tasks),
    )
    return [*tasks[:first_write], policy_task, *tasks[first_write:]]


def _contains_order_id(text: str) -> bool:
    import re

    return bool(re.search(r"[0-9a-fA-F]{32}", text))
