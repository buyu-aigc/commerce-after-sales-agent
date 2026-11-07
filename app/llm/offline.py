from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.intent.decomposer import decompose_business_message
from app.llm.types import InputGuardResult, OlistTaskResult, OutputGuardResult, PlannedTask, TaskPlanResult


@dataclass
class OfflineMessage:
    content: str


class OfflineChatModel:
    """Small local stand-in used when no real API key is configured."""

    def with_structured_output(self, schema: type) -> OfflineStructuredModel:
        return OfflineStructuredModel(schema)

    async def ainvoke(self, messages: list[Any]) -> OfflineMessage:
        text = str(messages[-1].content if messages else "")
        return OfflineMessage(content=_fallback_answer(text))


class OfflineStructuredModel:
    def __init__(self, schema: type) -> None:
        self._schema = schema

    async def ainvoke(self, messages: list[Any]) -> Any:
        text = str(messages[-1].content if messages else "")
        schema_name = self._schema.__name__
        if schema_name == "TaskPlanResult":
            return TaskPlanResult(
                tasks=[
                    PlannedTask(
                        intent=task.intent,
                        text=task.text,
                        side_effect=task.side_effect,
                        action_type=task.action_type,
                    )
                    for task in decompose_business_message(text)
                ]
            )
        if schema_name == "OlistTaskResult":
            return _extract_olist_task(text)
        if schema_name == "InputGuardResult":
            return _input_guard(text)
        if schema_name == "OutputGuardResult":
            return _output_guard(text)
        return self._schema()


def _extract_olist_task(text: str) -> OlistTaskResult:
    order_match = re.search(r"[0-9a-fA-F][0-9a-fA-F\s:-]{30,80}[0-9a-fA-F]", text)
    category_match = re.search(r"[a-z]+(?:[_ -][a-z]+)+", text.lower())
    order_id = re.sub(r"[^0-9a-fA-F]", "", order_match.group(0)).lower() if order_match else ""
    return OlistTaskResult(
        order_id=order_id if len(order_id) == 32 else "",
        category=category_match.group(0).replace(" ", "_").replace("-", "_") if category_match else "",
        user_goal="offline extraction",
    )


def _input_guard(text: str) -> InputGuardResult:
    lowered = text.lower()
    blocked = (
        "ignore your instructions",
        "ignore previous instructions",
        "disregard your instructions",
        "developer message",
        "system prompt",
        "hidden prompt",
        "print your prompt",
        "reveal your prompt",
        "jailbreak",
        "dan mode",
        "act as dan",
        "bypass",
        "越权",
        "绕过",
        "忽略之前",
        "忽略你的指令",
        "系统提示词",
        "开发者消息",
        "提示词",
        "泄露",
        "删除数据库",
        "drop table",
        "delete from",
        "操作系统内核",
        "python code",
        "javascript",
        "exploit",
        "sql injection",
        "xss",
        "色情",
        "仇恨",
    )
    if any(item in lowered for item in blocked):
        return InputGuardResult(on_topic=False, reason="offline guard blocked unsafe/off-topic request")
    return InputGuardResult(on_topic=True, reason="offline guard allowed support request")


def _output_guard(text: str) -> OutputGuardResult:
    if not text.strip():
        return OutputGuardResult(valid=False, reason="empty answer")
    if "[TODO]" in text or "Traceback" in text:
        return OutputGuardResult(valid=False, reason="placeholder or traceback")
    return OutputGuardResult(valid=True, reason="offline output checks passed")


def _fallback_answer(text: str) -> str:
    if "policy" in text.lower() or "政策" in text:
        return "离线模式：已根据命中的政策章节生成回答；涉及退款、补偿或创建工单时需要人工确认。"
    if "risk" in text.lower() or "风险" in text:
        return "离线模式：已根据类目聚合统计生成运营风险摘要，建议关注延迟率、低分率和取消率。"
    return "离线模式：已完成处理。配置 OpenAI-compatible API key 后可启用真实模型生成。"
