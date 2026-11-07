"""LLM-based response generators and slot extractors."""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.llm.json_fallback import add_json_instruction, native_structured_output_enabled, parse_json_model
from app.llm.offline import OfflineChatModel
from app.llm.prompts import OLIST_TASK_PROMPT, POLICY_ANSWER_PROMPT, QA_ANSWER_PROMPT
from app.llm.types import OlistTaskResult

logger = logging.getLogger(__name__)


class QaResponseGenerator:
    """Generates marketplace Q&A answers using LLM with retrieved insights as context."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def generate(
        self,
        user_message: str,
        products: list[dict],
        support_docs: list[dict[str, object]] | None = None,
    ) -> str:
        if not products:
            return "I couldn't find matching marketplace insights for your query."
        if isinstance(self._llm, OfflineChatModel):
            return _format_insight_fallback(products, support_docs or [])

        context = "\n".join(
            f"- {p['name']}: value={p['price']:.2f} | {p['description']} | rules: {p['rules']}"
            for p in products
        )
        if support_docs:
            context += "\n\nRelevant support conversation patterns:\n"
            context += "\n".join(
                f"- {doc.get('intent', '')}/{doc.get('capability', '')}: {doc.get('text', '')}"
                for doc in support_docs[:3]
            )
        prompt = QA_ANSWER_PROMPT.format(product_context=context)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]
        try:
            response = await self._llm.ainvoke(messages)
            return _with_canonical_category_prefix(str(response.content), products)
        except Exception as exc:
            logger.warning("QA generation LLM failed, using grounded template fallback: %s", exc)
            return _format_insight_fallback(products, support_docs or [])


class PolicyResponseGenerator:
    """Generates policy answers using retrieved KB sections as grounded context."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def generate(
        self,
        user_message: str,
        policy_sections: list[dict[str, object]],
        support_docs: list[dict[str, object]] | None = None,
        completed_context: str = "",
    ) -> str:
        if not policy_sections:
            return "没有找到匹配的客服政策，请转人工确认。"
        if isinstance(self._llm, OfflineChatModel):
            titles = "、".join(str(item["section_title"]) for item in policy_sections)
            snippets = "；".join(str(item["text"])[:80] for item in policy_sections[:2])
            examples = _format_support_examples(support_docs or [])
            return (
                f"根据已命中的政策章节：{titles}。\n"
                f"政策要点：{snippets}\n"
                f"{examples}"
                "结论：涉及退款、补偿、取消订单、改地址或创建工单时，需要人工确认后再执行。"
            )
        context = "\n".join(
            f"- {item['section_title']} [{item.get('source_type', 'knowledge')}] "
            f"({item['source']}): {item['text']}"
            for item in policy_sections
        )
        if support_docs:
            context += "\n\nRelevant support conversation patterns:\n"
            context += "\n".join(
                f"- {doc.get('intent', '')}/{doc.get('capability', '')}: {doc.get('text', '')}"
                for doc in support_docs[:3]
            )
        if completed_context:
            context += "\n\nAlready completed deterministic tool results:\n"
            context += completed_context
        prompt = POLICY_ANSWER_PROMPT.format(policy_context=context)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]
        try:
            response = await self._llm.ainvoke(messages)
            return str(response.content)
        except Exception as exc:
            logger.warning("Policy generation LLM failed, using grounded template fallback: %s", exc)
            titles = "、".join(str(item["section_title"]) for item in policy_sections)
            return (
                f"已命中政策章节：{titles}。当前 LLM 不可用，建议客服按上述政策原文处理，"
                "涉及补偿或创建 case 时转人工确认。"
            )

    def customer_safe_template(
        self,
        user_message: str,
        policy_sections: list[dict[str, object]],
    ) -> str | None:
        """Serve frequent, policy-grounded FAQs without an unnecessary LLM turn."""

        if not policy_sections:
            return None
        text = user_message.lower()
        if "发票" in text:
            return (
                "申请发票需要核对订单、客户身份或税务资料，"
                "并提供发票抬头、税号或账单标识及联系邮箱。"
                "当前只是说明所需资料，尚未提交开票申请。"
            )
        if "取消" in text:
            return (
                "能否取消订单取决于当前订单和物流状态。"
                "订单尚未进入配送时可以提交申请，工作人员会为您核查。"
            )
        if "地址" in text:
            return "修改收货地址需要先确认订单尚未进入配送。请提交申请，工作人员会核对后告知您结果。"
        if any(term in text for term in ("退款", "补偿", "赔付")):
            return (
                "退款或补偿需要先核查订单状态和具体原因。"
                "确认符合处理条件后才能继续办理，当前不能直接承诺金额或到账时间。"
            )
        return None


class OlistTaskExtractor:
    """Extracts order/category slots from an Olist support request."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._base_llm = llm
        self._llm = (
            llm.with_structured_output(OlistTaskResult)
            if native_structured_output_enabled(llm)
            else None
        )

    async def extract(self, user_message: str) -> OlistTaskResult:
        messages = [
            SystemMessage(content=OLIST_TASK_PROMPT),
            HumanMessage(content=user_message),
        ]
        try:
            if self._llm is None:
                raw = await self._base_llm.ainvoke(add_json_instruction(messages, OlistTaskResult))
                return parse_json_model(raw, OlistTaskResult)
            return await self._llm.ainvoke(messages)
        except Exception as exc:
            logger.warning("Structured task extractor failed, trying JSON-text fallback: %s", exc)
            if _is_transport_error(exc):
                return _regex_task_result(user_message)
            try:
                raw = await self._base_llm.ainvoke(add_json_instruction(messages, OlistTaskResult))
                return parse_json_model(raw, OlistTaskResult)
            except Exception as fallback_exc:
                logger.warning("Task extractor LLM failed, using regex fallback: %s", fallback_exc)
            return _regex_task_result(user_message)

    def fast_extract(self, user_message: str) -> OlistTaskResult | None:
        """Extract an exact order id locally when semantic inference is unnecessary."""

        order_match = re.search(r"[0-9a-fA-F][0-9a-fA-F\s:-]{30,80}[0-9a-fA-F]", user_message)
        if not order_match:
            return None
        order_id = re.sub(r"[^0-9a-fA-F]", "", order_match.group(0)).lower()
        if len(order_id) != 32:
            return None
        return OlistTaskResult(order_id=order_id, category="", user_goal="exact order id")


def _format_insight_fallback(products: list[dict], support_docs: list[dict[str, object]]) -> str:
    lines = ["当前 LLM 生成不可用，先返回已检索到的结构化运营事实："]
    for item in products[:3]:
        lines.append(f"- {item['name']}: {item['description']}")
    examples = _format_support_examples(support_docs)
    if examples:
        lines.append(examples.strip())
    lines.append("建议优先排查延迟率、低分率和取消率最高的类目，并用样例订单继续下钻。")
    return "\n".join(lines)


def _regex_task_result(user_message: str) -> OlistTaskResult:
    order_match = re.search(r"[0-9a-fA-F][0-9a-fA-F\s:-]{30,80}[0-9a-fA-F]", user_message)
    category_match = re.search(r"[a-z]+(?:[_ -][a-z]+)+", user_message.lower())
    order_id = re.sub(r"[^0-9a-fA-F]", "", order_match.group(0)).lower() if order_match else ""
    return OlistTaskResult(
        order_id=order_id if len(order_id) == 32 else "",
        category=(
            category_match.group(0).replace(" ", "_").replace("-", "_")
            if category_match
            else ""
        ),
        user_goal="fallback extraction",
    )


def _is_transport_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(marker in name for marker in ("timeout", "connection", "network"))


def _with_canonical_category_prefix(answer: str, products: list[dict]) -> str:
    names = [str(item.get("name", "")) for item in products[:3] if item.get("name")]
    if not names:
        return answer
    missing = [name for name in names if name not in answer]
    if not missing:
        return answer
    return f"命中类目：{', '.join(names)}。\n{answer}"


def _format_support_examples(support_docs: list[dict[str, object]]) -> str:
    if not support_docs:
        return ""
    examples = [
        f"{doc.get('intent', '')}/{doc.get('capability', '')}"
        for doc in support_docs[:3]
        if doc.get("intent") or doc.get("capability")
    ]
    if not examples:
        return ""
    return f"参考历史客服语料模式：{', '.join(examples)}。\n"
