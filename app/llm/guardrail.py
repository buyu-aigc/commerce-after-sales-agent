"""Fast, layered guardrails for the customer-facing request path."""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.llm.json_fallback import add_json_instruction, native_structured_output_enabled, parse_json_model
from app.llm.prompts import INPUT_GUARD_PROMPT, OUTPUT_GUARD_PROMPT
from app.llm.types import InputGuardResult, OutputGuardResult

logger = logging.getLogger(__name__)


class Guardrail:
    """Use deterministic checks for clear cases and reserve the LLM for ambiguity."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._base_llm = llm
        native_structured = native_structured_output_enabled(llm)
        self._input_llm = llm.with_structured_output(InputGuardResult) if native_structured else None
        self._output_llm = llm.with_structured_output(OutputGuardResult) if native_structured else None

    def check_input_deterministic(self, message: str) -> InputGuardResult:
        """Blocking without a model call, for turns whose scope a validated plan already decided."""
        if _contains_blocked_signal(message.lower()):
            return InputGuardResult(on_topic=False, reason="deterministic blocked pattern")
        return InputGuardResult(on_topic=True, reason="scope decided by validated turn plan")

    async def check_input(self, message: str, history: list[dict] | None = None) -> InputGuardResult:
        text = message.lower()
        if _contains_blocked_signal(text):
            return InputGuardResult(on_topic=False, reason="deterministic blocked pattern")
        if (
            _looks_like_marketplace_support(text)
            or _looks_like_account_history_request(text)
            or _looks_like_safe_followup(text)
            or _looks_like_small_talk(text)
        ):
            return InputGuardResult(on_topic=True, reason="deterministic marketplace allowlist")

        context_text = ""
        if history:
            last_exchanges = history[-4:]  # last 2 turns (user + assistant)
            context_text = (
                "Conversation so far:\n"
                + "\n".join(f"[{m['role']}] {m['content'][:200]}" for m in last_exchanges)
                + "\n\n"
            )

        messages = [
            SystemMessage(content=INPUT_GUARD_PROMPT),
            HumanMessage(content=f"{context_text}User message: {message}"),
        ]
        try:
            if self._input_llm is None:
                raw = await self._base_llm.ainvoke(add_json_instruction(messages, InputGuardResult))
                return _apply_business_allow_override(message, parse_json_model(raw, InputGuardResult))
            return _apply_business_allow_override(message, await self._input_llm.ainvoke(messages))
        except Exception as exc:
            logger.warning("Structured input guard failed, trying JSON-text fallback: %s", exc)
            if _is_transport_error(exc):
                return _heuristic_input_guard(message)
            try:
                raw = await self._base_llm.ainvoke(add_json_instruction(messages, InputGuardResult))
                return _apply_business_allow_override(message, parse_json_model(raw, InputGuardResult))
            except Exception as fallback_exc:
                logger.warning("Input guard LLM failed, using heuristic fallback: %s", fallback_exc)
                return _heuristic_input_guard(message)

    async def check_output(self, answer: str) -> OutputGuardResult:
        # Output health is intentionally local: workflow grounding, verifier
        # decisions, and the customer presenter enforce business truthfulness.
        return _deterministic_output_guard(answer)

    async def check_output_with_llm(self, answer: str) -> OutputGuardResult:
        """Optional diagnostic classifier; keep it out of the serving critical path."""
        messages = [
            SystemMessage(content=OUTPUT_GUARD_PROMPT),
            HumanMessage(content=answer),
        ]
        try:
            if self._output_llm is None:
                raw = await self._base_llm.ainvoke(add_json_instruction(messages, OutputGuardResult))
                return parse_json_model(raw, OutputGuardResult)
            return await self._output_llm.ainvoke(messages)
        except Exception as exc:
            logger.warning("Structured output guard failed, trying JSON-text fallback: %s", exc)
            try:
                raw = await self._base_llm.ainvoke(add_json_instruction(messages, OutputGuardResult))
                return parse_json_model(raw, OutputGuardResult)
            except Exception as fallback_exc:
                logger.warning("Output guard LLM failed, using deterministic fallback: %s", fallback_exc)
            return _deterministic_output_guard(answer)


def _heuristic_input_guard(message: str) -> InputGuardResult:
    text = message.lower()
    if _contains_blocked_signal(text):
        return InputGuardResult(on_topic=False, reason="heuristic off-topic or unsafe")
    return InputGuardResult(on_topic=True, reason="llm unavailable; fail-open for support availability")


def _deterministic_output_guard(answer: str) -> OutputGuardResult:
    if not answer.strip():
        return OutputGuardResult(valid=False, reason="empty answer")
    lowered = answer.lower()
    if "[todo]" in lowered or "traceback" in lowered or "exception:" in lowered:
        return OutputGuardResult(valid=False, reason="placeholder or runtime error")
    return OutputGuardResult(valid=True, reason="deterministic output health checks passed")


def _apply_business_allow_override(message: str, result: InputGuardResult) -> InputGuardResult:
    """Reduce LLM guard false rejects on known marketplace operations language."""

    if result.on_topic:
        return result
    text = message.lower()
    if _contains_blocked_signal(text):
        return result
    if _looks_like_marketplace_support(text):
        return InputGuardResult(
            on_topic=True,
            reason=f"allowed by marketplace-support allowlist after LLM guard rejected: {result.reason}",
        )
    return result


def _contains_blocked_signal(text: str) -> bool:
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
    return any(item in text for item in blocked)


def _looks_like_marketplace_support(text: str) -> bool:
    business_terms = (
        "order",
        "category",
        "refund",
        "invoice",
        "cancel",
        "delivery",
        "payment",
        "review",
        "compensation",
        "escalation",
        "support",
        "risk",
        "health beauty",
        "health_beauty",
        "类目",
        "品类",
        "订单",
        "退款",
        "补偿",
        "发票",
        "取消",
        "物流",
        "配送",
        "延迟",
        "支付",
        "评价",
        "低分",
        "售后",
        "客服",
        "投诉",
        "升级",
        "运营",
        "风险",
        "取消率",
        "低分率",
    )
    return any(term in text for term in business_terms)


def _looks_like_account_history_request(text: str) -> bool:
    """Allow safe, read-only account questions without a gateway round trip."""

    normalized = re.sub(r"[\s，。！？!?~～]+", "", text)
    return any(
        phrase in normalized
        for phrase in (
            "我的购物习惯", "我的消费习惯", "我的购物偏好", "我喜欢什么产品",
            "我喜欢什么样的产品", "我买过什么", "我买了什么", "我的购买记录",
            "我平时买什么", "我经常买什么",
        )
    )


def _looks_like_safe_followup(text: str) -> bool:
    return text.strip() in {"yes", "no", "确认", "取消", "好的", "好", "继续", "人工客服"}


def _looks_like_small_talk(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？!?~～]+", "", text)
    return normalized in {
        "你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "早上好", "晚上好",
        "谢谢", "感谢", "多谢", "再见", "拜拜", "你能做什么", "你可以做什么", "帮助",
    }


def _is_transport_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(marker in name for marker in ("timeout", "connection", "network"))
