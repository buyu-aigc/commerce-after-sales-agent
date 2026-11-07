from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BusinessTask:
    intent: str
    text: str
    side_effect: bool
    action_type: str = "none"
    order_filter: str = "all"


ORDER_PATTERNS = (
    "order status",
    "current status",
    "track order",
    "track my order",
    "eta",
    "delivered",
    "purchase",
    "订单状态",
    "查订单",
    "状态",
    "送达",
    "物流",
)

QA_PATTERNS = (
    "category risk",
    "review risk",
    "logistics risk",
    "operations",
    "operational risk",
    "category",
    "类目",
    "运营",
    "风险",
    "评价风险",
    "物流风险",
    "低分率",
    "延迟率",
    "取消率",
)

OPS_DECISION_PATTERNS = (
    "after-sales report",
    "support operations report",
    "priority queue",
    "prioritize",
    "action plan",
    "运营决策",
    "风险日报",
    "审核台",
    "处理队列",
    "售后队列",
    "优先级",
    "优先跟进",
    "重点跟进",
    "高风险订单",
    "高风险类目",
    "处理建议",
    "行动建议",
    "经营建议",
)

POLICY_PATTERNS = (
    "policy",
    "fee",
    "charge",
    "penalty",
    "payment method",
    "invoice",
    "delivery period",
    "refund policy",
    "termination",
    "withdrawal",
    "手续费",
    "政策",
    "发票",
    "支付方式",
    "退款规则",
    "配送时效",
    "补偿边界",
)

POLICY_QUESTION_CUES = (
    "what",
    "how",
    "can ",
    "could ",
    "should ",
    "是否",
    "能否",
    "能不能",
    "可不可以",
    "可以吗",
    "怎么",
    "规则",
    "政策",
    "边界",
    "直接承诺",
)

ESCALATION_PATTERNS = (
    "cancel order",
    "canceling order",
    "cancelling order",
    "cancel purchase",
    "canceling purchase",
    "cancelling purchase",
    "cancel it",
    "change order",
    "change shipping address",
    "get refund",
    "complaint",
    "human agent",
    "customer service",
    "open a case",
    "escalat",
    "compensat",
    "取消订单",
    "改地址",
    "退款",
    "投诉",
    "人工",
    "升级",
    "延迟",
    "低分",
    "安抚",
    "跟进",
    "话术",
    "创建 case",
    "发券",
)

SPLIT_RE = re.compile(
    (
        r"(?:，另外|另外|并且|并说明|并查询|并查看|而且|顺便|然后|再帮我|再|同时|"
        r"接着|;|；|, and |\balso\b|\band also\b|\bthen\b)"
    ),
    re.IGNORECASE,
)


def decompose_business_message(message: str) -> list[BusinessTask]:
    """Split a user message into ordered business tasks with duplicate task suppression."""
    segments = [part.strip() for part in SPLIT_RE.split(message) if part.strip()] or [message]
    tasks: list[BusinessTask] = []
    seen: set[tuple[str, str]] = set()
    for segment in segments:
        intent = _classify_segment(segment)
        signature = (intent, _task_signature(segment))
        if signature in seen:
            continue
        seen.add(signature)
        tasks.append(
            BusinessTask(
                intent=intent,
                text=segment,
                side_effect=(intent == "escalation"),
                action_type=_action_type(segment, intent),
                order_filter=_customer_orders_filter(segment) or "all",
            )
        )
    return tasks


def _classify_segment(segment: str) -> str:
    text = segment.lower()
    if _looks_like_customer_profile_request(text):
        return "customer_profile"
    if _is_small_talk(text):
        return "small_talk"
    if _needs_business_clarification(text):
        return "clarify"
    if _customer_orders_filter(text) is not None:
        return "customer_orders"
    if _has_any(text, OPS_DECISION_PATTERNS):
        return "ops_decision"
    has_order_id = bool(re.search(r"\b[a-f0-9]{32}\b", text))
    if has_order_id and _has_any(text, ORDER_PATTERNS) and not _looks_like_side_effect_action(text):
        return "order_status"
    if _looks_like_order_cancel_action(text):
        return "escalation"
    if _has_any(text, POLICY_PATTERNS) or _looks_like_policy_question(text):
        return "policy"
    if _looks_like_side_effect_action(text) or _has_any(text, ESCALATION_PATTERNS):
        return "escalation"
    if _has_any(text, QA_PATTERNS):
        return "qa"
    if _has_any(text, ORDER_PATTERNS) or "{{order number}}" in text:
        return "order_status"
    return "policy"


def _is_small_talk(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？!?~～]+", "", text)
    return normalized in {
        "你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "早上好", "晚上好",
        "谢谢", "感谢", "多谢", "再见", "拜拜", "你能做什么", "你可以做什么", "帮助",
    }


def _looks_like_customer_profile_request(text: str) -> bool:
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
    return normalized in {
        "帮我处理订单", "帮我处理一下", "订单有问题", "我的订单有问题", "我要售后",
        "售后怎么处理", "我需要帮助", "帮帮我", "怎么办", "怎么弄",
    }


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _looks_like_policy_question(text: str) -> bool:
    side_effect_topic = _has_any(
        text,
        (
            "refund",
            "退款",
            "补偿",
            "赔付",
            "cancel",
            "取消",
            "address",
            "地址",
            "invoice",
            "发票",
        ),
    )
    return side_effect_topic and _has_any(text, POLICY_QUESTION_CUES)


def _looks_like_order_cancel_action(text: str) -> bool:
    has_cancel_verb = "cancel" in text or "cancell" in text
    has_order_object = _has_any(text, ("{{order number}}", "order", "purchase", "oorder", "puchase"))
    policy_only = _has_any(text, ("fee", "charge", "penalty", "termination", "withdrawal"))
    return has_cancel_verb and has_order_object and not policy_only


def _looks_like_side_effect_action(text: str) -> bool:
    action_topic = _has_any(
        text,
        (
            "refund",
            "compensat",
            "coupon",
            "cancel order",
            "cancel purchase",
            "change address",
            "change shipping address",
            "invoice",
            "complaint",
            "open a case",
            "escalat",
            "退款",
            "补偿",
            "赔付",
            "发券",
            "取消订单",
            "改地址",
            "修改地址",
            "地址",
            "发票",
            "开票",
            "投诉",
            "升级",
            "创建 case",
            "建单",
            "工单",
        ),
    )
    action_verb = _has_any(
        text,
        (
            "apply",
            "request",
            "submit",
            "create",
            "open",
            "file",
            "get refund",
            "帮我",
            "为订单",
            "给订单",
            "我要",
            "申请",
            "提交",
            "创建",
            "修改",
            "帮",
            "办理",
        ),
    )
    return action_topic and action_verb


def _task_signature(segment: str) -> str:
    normalized = re.sub(r"\s+", " ", segment.lower()).strip()
    order_ids = re.findall(r"\b[a-f0-9]{32}\b", normalized)
    if order_ids:
        return "|".join(order_ids)
    return normalized


def _action_type(segment: str, intent: str) -> str:
    if intent != "escalation":
        return "none"
    text = segment.lower()
    if _has_any(text, ("refund", "退款", "退钱", "赔付", "compensat", "coupon", "发券")):
        return "refund_request"
    if _has_any(text, ("cancel order", "cancel purchase", "取消订单", "取消这个订单")):
        return "cancel_order"
    if _has_any(text, ("change shipping address", "改地址", "修改地址", "change address")) or (
        "地址" in text and "修改" in text
    ):
        return "change_address"
    if _has_any(text, ("invoice", "发票", "开票")):
        return "invoice_request"
    if _has_any(text, ("complaint", "投诉", "升级投诉", "转人工", "人工介入")):
        return "complaint_escalation"
    return "open_support_case"
