"""Deterministic complexity/risk decomposition for after-sales requests.

Complex is not the same as risky.  A customer may ask for several low-risk
actions plus one high-risk action in a single turn.  This module makes that
distinction explicit before the graph executes anything:

* split an ordered request into independent sub-requests;
* attach a per-sub-request action type, order scope and risk label;
* expose evidence completeness and missing-evidence reasons;
* produce a stable summary for routing, customer progress and evaluation.

The splitter is intentionally deterministic.  It is the safety floor beneath
the LLM planner: even when the model is unavailable, a compound request must
not collapse into one monolithic write action.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from app.after_sales.address import address_fields
from app.olist.status import PRE_SHIPMENT_STATUSES

WRITE_INTENTS = {"escalation"}
READ_INTENTS = {"order_status", "policy", "qa", "customer_orders", "customer_profile"}
SUPPORTED_ACTIONS = {
    "refund_request",
    "cancel_order",
    "change_address",
    "invoice_request",
    "complaint_escalation",
    "open_support_case",
}

# Split only at request boundaries.  "和/与/并" are split when they connect an
# action ("退款和取消"), but not inside an order id or a noun phrase.
_SPLIT_RE = re.compile(
    (
        r"(?:，|,|、|；|;|。|！|!|\n|\t|"
        r"并且|并说明|并解释|并告知|并查询|并查看|并确认|同时|然后|接着|另外|此外|还要|还需|再帮我|以及|顺便|加上|"
        r"但是|不过|而是|但|改为|改成|只(?=查询|查|看|想查|要查)|"
        r"\band also\b|\balso\b|\bthen\b|\band\b|"
        r"再(?=[帮给把要申开改退取投处])|"
        r"并(?=[帮给把要申开改退取投处])|"
        r"和(?=[帮给把要申开改退取投处])|"
        r"与(?=[帮给把要申开改退取投处])"
        r")"
    ),
    re.IGNORECASE,
)

_ORDER_ID_RE = re.compile(r"\b[0-9a-f]{32}\b", re.IGNORECASE)

# Negation governs a request verb, not every occurrence of 不/没 in a reason
# (e.g. "没收到货，我要退款"). Only request scaffolding may occur between
# the negation and the action. Ambiguous double negations remain non-writing.
_DENIED_ACTION_PREFIX = re.compile(
    r"(?:不要|不用|不必|无需|不想|不愿|不打算|不需要|不能|不可以|不允许|不得|没有|不是|并非|不再|禁止|别|勿|不|未|没|"
    r"\bdo\s+not\b|\bdon['’]t\b|\bnever\b|\bnot\b|\bno\s+longer\b)"
    r"(?:\s|[：:]|我|你|我们|帮忙|帮我|替我|给我|为我|给|替|帮|为|对|把|将|这|那|个|笔|单|订单|该|"
    r"再|先|现在|暂时|立即|直接|进行|执行|提交|办理|发起|申请|处理|任何|做|"
    r"\bplease\b|\bapply\b|\brequest\b|\bsubmit\b|\bprocess\b|\bissue\b|"
    r"\bmy\b|\bthis\b|\bthat\b|\border\b|\bfor\b|\bme\b|\bwant\b|\bto\b|\bneed\b|\ba\b|\ban\b|\bthe\b)*$",
    re.IGNORECASE,
)
# A topic can precede its governing prohibition: "退款先不办理".
# Match only request predicates, not reasons such as "退款不影响发票".
_DENIED_ACTION_SUFFIX = re.compile(
    r"^\s*(?:(?:我|先|暂时|现在|这次|目前|就|也|都)\s*)*"
    r"(?:不要|不用|不必|无需|不能|别|不)\s*(?:再|继续)?\s*"
    r"(?:办理|办|申请|提交|执行|处理|进行|做|要|(?=$|[，,；;。！？!?]))",
    re.IGNORECASE,
)
_COORDINATED_ACTIONS = re.compile(
    r"(?=[\s\S]*(?:和|与|或者|或|以及|及|、|\band\b|\bor\b|\balso\b))"
    r"(?:\s|和|与|或者|或|以及|及|、|给|对|把|为|订单|\band\b|\bor\b|\balso\b)+$",
    re.IGNORECASE,
)

_STRONG_POLICY_CUES = (
    "政策", "规则", "条件", "标准", "流程", "手续费", "多久", "需要什么",
    "需要哪些", "材料", "凭证", "一般", "通常", "policy", "rule", "how long", "what do i need",
)
_WEAK_POLICY_CUES = ("是否", "能不能", "可不可以", "可以吗", "怎么", "如何")
_STRONG_REQUEST_CUES = (
    "我要", "我想", "帮我", "给我", "申请", "提交", "办理", "要求", "麻烦",
    "请帮", "请给", "退一下", "开一下", "改一下", "取消掉", "处理掉",
    "发起", "立刻", "马上", "request", "please", "i want", "i need",
)
_STATUS_CUES = (
    "状态", "物流", "送到", "送达", "到哪", "进度", "发货了吗", "是否已取消",
    "取消了吗", "取消成功", "发货", "发没发", "现状", "取消已经生效", "配送",
    "track", "status", "eta", "delivered",
    "delivery", "shipment",
)
_READ_TOPIC_CUES = ("退款", "取消", "发票", "地址", "投诉", "物流", "状态", "进度")

# Explicit action phrases are stronger evidence than a bare topic word.  A
# policy question such as "退款需要什么条件" must not become a write request.
_ACTION_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "complaint_escalation",
        (
            "投诉", "举报", "升级投诉", "转人工", "人工介入", "找人工", "客服介入",
            "complaint", "file a complaint", "escalate to a human", "12315",
            "曝光", "起诉", "律师函", "监管投诉", "差评",
        ),
        ("投诉", "举报", "complaint"),
    ),
    (
        "cancel_order",
        (
            "取消订单", "申请取消", "帮我取消", "我要取消", "取消这笔", "取消该订单",
            "取消掉", "终止订单", "撤销订单", "取消一下", "取消",
            "cancel order", "cancel this order", "cancel it", "cancel",
        ),
        ("取消", "终止", "撤销", "cancel"),
    ),
    (
        "change_address",
        (
            "改地址", "修改地址", "换地址", "地址改", "改收货地址",
            "change address", "update address", "address change",
        ),
        ("改地址", "修改地址", "换地址", "change address", "update address"),
    ),
    (
        "invoice_request",
        (
            "开发票", "申请发票", "开票", "开具发票", "要发票", "发票申请", "开一下发票",
            "invoice request", "request invoice", "issue invoice",
        ),
        ("发票", "开票", "invoice"),
    ),
    (
        "refund_request",
        (
            "申请退款", "我要退款", "想退款", "要退款", "退款申请", "帮我退款",
            "给我退款", "退钱", "退货退款", "退款处理", "退款吧", "申请退货退款",
            "request a refund", "refund request", "get a refund", "want a refund",
            "退款", "退钱",
        ),
        ("退款", "退钱", "退还", "refund"),
    ),
    (
        "open_support_case",
        (
            "售后", "工单", "建单", "创建 case", "创建case", "open a case", "人工",
        ),
        ("售后", "工单", "建单"),
    ),
)

# Signals that must never be auto-executed even when the active policy would
# otherwise allow a low-value refund/cancel.
_HIGH_RISK_TERMS = (
    "重复索赔", "再次索赔", "第二次索赔", "之前退过", "已经退过", "已经退款",
    "退款失败", "退款异常", "无法退款", "补偿", "赔偿", "赔付", "曝光", "媒体",
    "起诉", "律师", "12315", "监管", "欺诈", "诈骗", "chargeback", "拒付",
    "威胁", "差评", "低评分", "投诉到",
)
_LEGAL_OR_ABUSE_TERMS = ("起诉", "律师", "12315", "监管", "欺诈", "诈骗", "chargeback", "拒付")
_HARD_RISK_FLAGS = ("repeat_claim", "refund_failure", "prior_compensation")

_ACTION_LABELS = {
    "refund_request": "退款",
    "cancel_order": "取消订单",
    "change_address": "修改地址",
    "invoice_request": "发票",
    "complaint_escalation": "投诉",
    "open_support_case": "售后跟进",
}


@dataclass(frozen=True)
class SubRequest:
    """One independently policy-checkable unit of a customer request."""

    intent: str
    text: str
    action_type: str
    order_id: str = ""
    order_scope: str = "missing"  # explicit | inherited | missing
    side_effect: bool = False
    risk_level: str = "medium"
    risk_reasons: list[str] = field(default_factory=list)
    auto_eligible: bool = False
    evidence_complete: bool = True
    missing_evidence: list[str] = field(default_factory=list)
    dependency_index: int | None = None
    # A bare topic such as "订单…地址" cannot establish read intent. Keep
    # existing planner/selected-order routing for that uncertain fallback.
    semantic_fallback: bool = False

    def to_task(self, *, index: int, total: int) -> dict[str, object]:
        return {
            "intent": self.intent,
            "text": self.text,
            "side_effect": self.side_effect,
            "action_type": self.action_type if self.side_effect else "none",
            "order_filter": "all",
            "order_id": self.order_id,
            "order_id_scope": self.order_scope,
            "risk_level": self.risk_level,
            "risk_reasons": list(self.risk_reasons),
            "auto_eligible": self.auto_eligible,
            "evidence_complete": self.evidence_complete,
            "missing_evidence": list(self.missing_evidence),
            "sub_task_index": index,
            "sub_task_total": total,
            "depends_on": [self.dependency_index] if self.dependency_index is not None else [],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "action_type": self.action_type,
            "order_scope": self.order_scope,
            "risk_level": self.risk_level,
            "auto_eligible": self.auto_eligible,
            "evidence_complete": self.evidence_complete,
            "missing_evidence": list(self.missing_evidence),
        }


@dataclass(frozen=True)
class ComplexityAssessment:
    """Request-level tier plus its independently routable units."""

    message: str
    tier: str
    risk_level: str
    units: list[SubRequest]
    decomposition_mode: str
    missing_evidence: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    amount_total: float = 0.0
    order_ids: list[str] = field(default_factory=list)
    restricted_actions: list[tuple[str, str]] = field(default_factory=list)

    @property
    def write_units(self) -> list[SubRequest]:
        return [unit for unit in self.units if unit.side_effect]

    @property
    def has_explicit_write_request(self) -> bool:
        # An entire turn may ask a policy question after a separate, explicit
        # submission request. Only an affirmative imperative before the action
        # can override that turn-level question label.
        for unit in self.write_units:
            for start, _end, action, denied in _action_mentions(unit.text, has_order=True):
                if denied or action != unit.action_type:
                    continue
                prefix = unit.text[:start].lower()
                if any(cue in prefix for cue in _WEAK_POLICY_CUES) or re.search(
                    r"知道|了解|咨询|询问|\b(?:whether|how|can|could|know|understand|ask|if)\b", prefix
                ):
                    continue
                if _has_strong_request(prefix):
                    return True
        return False

    @property
    def read_units(self) -> list[SubRequest]:
        return [unit for unit in self.units if not unit.side_effect]

    @property
    def is_complex(self) -> bool:
        return self.tier == "complex"

    @property
    def action_types(self) -> list[str]:
        return sorted({unit.action_type for unit in self.write_units})

    @property
    def evidence_completeness(self) -> float:
        if not self.units:
            return 1.0
        complete = sum(1 for unit in self.units if unit.evidence_complete)
        return round(complete / len(self.units), 4)

    def to_tasks(self) -> list[dict[str, object]]:
        total = len(self.units)
        return [
            unit.to_task(index=index + 1, total=total)
            for index, unit in enumerate(self.units)
        ]

    def summary(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "risk_level": self.risk_level,
            "intent_count": len(self.units),
            "action_types": self.action_types,
            "order_count": len(self.order_ids),
            "amount_total": round(self.amount_total, 2),
            "evidence_completeness": self.evidence_completeness,
            "missing_evidence": list(self.missing_evidence),
            "risk_flags": list(self.risk_flags),
            "decomposition_mode": self.decomposition_mode,
            "sub_task_count": len(self.units),
            "write_task_count": len(self.write_units),
            "read_task_count": len(self.read_units),
            "auto_eligible_count": sum(1 for unit in self.units if unit.auto_eligible),
            "high_risk_count": sum(1 for unit in self.units if unit.risk_level == "high"),
            "units": [unit.to_dict() for unit in self.units],
            "restricted_action_count": len(self.restricted_actions),
        }


def assess_request(
    message: str,
    *,
    selected_order_id: str = "",
    order_lookup: Callable[[str], Any] | None = None,
    extra_high_risk_flags: Iterable[str] = (),
    max_units: int = 12,
) -> ComplexityAssessment:
    """Split and label a request without executing any business action."""

    text = str(message or "").strip()
    selected = _normalize_order_id(selected_order_id)
    units: list[SubRequest] = []
    last_order = selected
    all_orders: list[str] = []
    write_count = 0
    restricted_actions: list[tuple[str, str]] = []

    for segment in _segments(text):
        segment_orders = _order_ids(segment)
        if segment_orders:
            last_order = segment_orders[-1]
            for order_id in segment_orders:
                if order_id not in all_orders:
                    all_orders.append(order_id)
        positions = [(match.start(), match.group(0).lower()) for match in _ORDER_ID_RE.finditer(segment)]
        mentions = _action_mentions(segment, has_order=True)
        for start, _end, action, denied in mentions:
            if denied:
                restricted_actions.append((action, _choose_order(positions, start, last_order)))
        segment_units = _segment_units(
            segment,
            explicit_orders=segment_orders,
            default_order=last_order,
            order_lookup=order_lookup,
            extra_high_risk_flags=extra_high_risk_flags,
        )
        if (
            not segment_units and units and not segment_orders
            and not any(denied for *_mention, denied in mentions)
        ):
            # Address recipient/phone/street, invoice title/tax id and refund
            # reasons are often separate comma-delimited fragments.  They are
            # evidence for the preceding sub-request, not a new request.
            units[-1] = _append_evidence(units[-1], segment)
            continue
        for unit in segment_units:
            if unit.side_effect:
                write_count += 1
            units.append(unit)
            if unit.order_id and unit.order_id not in all_orders:
                all_orders.append(unit.order_id)
        if len(units) >= max_units:
            break

    units = _dedupe_units(units)[:max_units]
    write_units = [unit for unit in units if unit.side_effect]
    read_units = [unit for unit in units if not unit.side_effect]
    is_complex = len(units) >= 2 and (
        len(write_units) >= 2 or (len(write_units) >= 1 and len(read_units) >= 1)
    )
    if is_complex:
        # Read-only tasks must not be blocked by a write that follows them.
        first_status = next(
            (idx for idx, unit in enumerate(units) if unit.intent == "order_status"),
            None,
        )
        if first_status is not None:
            units = [
                _with_dependency(unit, first_status + 1)
                if unit.side_effect and idx > first_status
                else unit
                for idx, unit in enumerate(units)
            ]

    risk_level = _request_risk_level(units)
    risk_flags = sorted({reason for unit in units for reason in unit.risk_reasons})
    missing_evidence = sorted({item for unit in units for item in unit.missing_evidence})
    amount_total = 0.0
    seen_orders: set[str] = set()
    for unit in units:
        if not unit.order_id or unit.order_id in seen_orders:
            continue
        seen_orders.add(unit.order_id)
        order = _lookup_order(order_lookup, unit.order_id)
        amount_total += _order_float(order, "payment_value", 0.0)

    return ComplexityAssessment(
        message=text,
        tier="complex" if is_complex else "simple",
        risk_level=risk_level,
        units=units,
        decomposition_mode=(
            "complex_sub_request_decomposition" if is_complex else "single_request"
        ),
        missing_evidence=missing_evidence,
        risk_flags=risk_flags,
        amount_total=amount_total,
        order_ids=[order_id for order_id in all_orders if order_id],
        restricted_actions=list(dict.fromkeys(restricted_actions)),
    )


def is_policy_only(message: str) -> bool:
    # Weak question words (e.g. 怎么) also occur in vague order queries.
    # Only explicit policy language can bypass authenticated order discovery.
    lowered = message.lower()
    analysis = assess_request(message)
    return bool(
        any(cue in lowered for cue in _STRONG_POLICY_CUES)
        and analysis.units and all(unit.intent == "policy" for unit in analysis.units)
    )


def _segments(text: str) -> list[str]:
    # Keep coordinated prohibitions in one clause so "不要退款和取消" does
    # not lose its negation when the conjunction is removed by splitting.
    segments: list[str] = []
    start = 0
    for match in _SPLIT_RE.finditer(text):
        fragment = text[start:match.start()]
        denied_fragment = any(
            denied for *_mention, denied in _action_mentions(fragment, has_order=True)
        )
        if match.group() in {"但是", "不过", "而是", "但", "改为", "改成", "只"}:
            following = _SPLIT_RE.search(text, match.end())
            next_fragment = text[match.end():following.start() if following else len(text)]
            denied_following = any(
                denied for *_mention, denied in _action_mentions(next_fragment, has_order=True)
            )
            # Only add contrast boundaries around prohibitions. Otherwise
            # preserve address verbs/values and names (e.g. 收件人但明).
            if not denied_fragment and not denied_following:
                continue
        if _COORDINATED_ACTIONS.fullmatch(match.group()):
            following = _SPLIT_RE.search(text, match.end())
            next_fragment = text[match.end():following.start() if following else len(text)]
            if denied_fragment or any(
                denied for *_mention, denied in _action_mentions(next_fragment, has_order=True)
            ):
                continue
        segments.append(fragment.strip(" \t\r\n，,、；;。！!"))
        start = match.end()
    segments.append(text[start:].strip(" \t\r\n，,、；;。！!"))
    segments = [part for part in segments if part]
    return segments or [text.strip()]


def _order_ids(text: str) -> list[str]:
    return [match.group(0).lower() for match in _ORDER_ID_RE.finditer(text)]


def _normalize_order_id(value: str) -> str:
    text = str(value or "").lower()
    return text if _ORDER_ID_RE.fullmatch(text) else ""


def _is_policy_question(text: str) -> bool:
    lowered = text.lower()
    if re.fullmatch(
        r"(?:我)?(?:这次)?(?:不是|不想|不需要|不要|不用)(?:在)?"
        r"(?:问|询问|咨询|了解)(?:一般)?(?:政策|规则|条件|流程)[。？?!\s]*", lowered
    ):
        return False
    if any(cue in lowered for cue in _STRONG_POLICY_CUES):
        return True
    if any(cue in lowered for cue in _WEAK_POLICY_CUES):
        # "我要退款怎么办" is still an action; "退款怎么办" is a policy ask.
        return not any(cue in lowered for cue in _STRONG_REQUEST_CUES)
    return False


def _is_status_read(text: str) -> bool:
    lowered = text.lower()
    has_status = any(cue in lowered for cue in _STATUS_CUES)
    request_text = re.sub(
        r"(?:请帮我|请帮|帮我|给我|替我|我要|我想|请)(?=查|看|确认|核实)|"
        r"\bplease\s+(?=check\b|track\b|show\b)",
        "", lowered,
    )
    has_request = any(cue in request_text for cue in _STRONG_REQUEST_CUES)
    return has_status and not has_request


def _has_strong_request(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in _STRONG_REQUEST_CUES)


def _action_mentions(text: str, *, has_order: bool) -> list[tuple[int, int, str, bool]]:
    lowered = text.lower()
    has_request = _has_strong_request(text)
    candidates: list[tuple[int, int, str]] = []
    for action_type, explicit, bare in _ACTION_RULES:
        phrases = set(explicit) | (set(bare) if has_order or has_request else set())
        for phrase in phrases:
            for match in re.finditer(re.escape(phrase), lowered):
                candidates.append((match.start(), match.end(), action_type))
    # A concrete destination with an explicit change/send verb is address
    # evidence even when the user omits the noun "地址".
    destination_verb = re.search(r"改到|送到|寄到", lowered)
    if destination_verb and address_fields(text).get("address"):
        candidates.append((destination_verb.start(), destination_verb.end(), "change_address"))
    # The semantic entry accepts colloquial 退 as a refund cue. Recognize its
    # prohibitions too, without granting writes from an ambiguous bare verb.
    denial_only = set()
    for match in re.finditer("退", lowered):
        span = (match.start(), match.end(), "refund_request")
        candidates.append(span)
        denial_only.add(span)
    mentions: list[tuple[int, int, str, bool]] = []
    for start, end, action in sorted(set(candidates), key=lambda item: (item[0], -item[1], item[2])):
        if any(old_action == action and old_start <= start and old_end >= end
               for old_start, old_end, old_action, _ in mentions):
            continue
        prefix = _ORDER_ID_RE.sub("", lowered[:start])
        denied = bool(
            _DENIED_ACTION_PREFIX.search(prefix) or _DENIED_ACTION_SUFFIX.search(lowered[end:])
            or re.search(r"(?:先|暂时)(?:不说|不谈|不提|不聊)\s*$", prefix)
            # Deferring a topic does not authorize another application.
            or re.match(r"^\s*(?:先|暂时|继续)(?:等着|等一等|等等|放着|搁着|等待审核|等审核)", lowered[end:])
        )
        if mentions:
            _previous_start, previous_end, _previous_action, previous_denied = mentions[-1]
            connector = _ORDER_ID_RE.sub("", lowered[previous_end:start])
            denied = denied or bool(previous_denied and (
                not connector.strip() or _COORDINATED_ACTIONS.fullmatch(connector)
            ))
        mentions.append((start, end, action, denied))
    # Coordinated objects can share a predicate after the final object:
    # "退款和取消都别办理". Do not propagate across another clause.
    for index in range(len(mentions) - 1, 0, -1):
        start, _end, _action, denied = mentions[index]
        previous_start, previous_end, previous_action, previous_denied = mentions[index - 1]
        connector = _ORDER_ID_RE.sub("", lowered[previous_end:start])
        if denied and _COORDINATED_ACTIONS.fullmatch(connector):
            mentions[index - 1] = (previous_start, previous_end, previous_action, True)
    return [m for m in mentions if m[3] or m[:3] not in denial_only]


def _action_hits(text: str, *, has_order: bool) -> list[tuple[int, str]]:
    hits = [
        (start, action)
        for start, _end, action, denied in _action_mentions(text, has_order=has_order)
        if not denied
    ]
    # "open_support_case" is the fallback, never a competitor of a specific action.
    specific = [(position, action) for position, action in hits if action != "open_support_case"]
    if specific:
        hits = specific
    return sorted(set(hits))


def _segment_units(
    segment: str,
    *,
    explicit_orders: list[str],
    default_order: str,
    order_lookup: Callable[[str], Any] | None,
    extra_high_risk_flags: Iterable[str],
) -> list[SubRequest]:
    lowered = segment.lower()
    policy_question = _is_policy_question(segment)
    status_read = _is_status_read(segment) and not any(
        cue in lowered for cue in _STRONG_POLICY_CUES
    )
    denied_actions = any(denied for *_mention, denied in _action_mentions(segment, has_order=True))
    action_hits = [] if policy_question else _action_hits(
        segment, has_order=bool(explicit_orders or default_order)
    )

    if status_read and (
        not action_hits
        or re.search(r"查|查看|确认|核实|进度|是否|是不是|了吗|已经生效|status|track|check", lowered)
    ):
        # A shared read predicate applies to every explicitly named order.
        # Keep independent targets; the executor accepts one order per task.
        return [
            _read_unit(
                intent="order_status",
                text=segment,
                order_id=order_id,
                order_scope="explicit" if explicit_orders else ("inherited" if default_order else "missing"),
            )
            for order_id in dict.fromkeys(explicit_orders or [default_order])
        ]

    if action_hits:
        units: list[SubRequest] = []
        multi_order_action = len(explicit_orders) > 1 and len(action_hits) == 1 and not denied_actions
        order_positions = [
            (match.start(), match.group(0).lower())
            for match in _ORDER_ID_RE.finditer(segment)
        ]
        for position, action_type in action_hits:
            if multi_order_action:
                for order_id in explicit_orders:
                    units.append(
                        _write_unit(
                            text=segment,
                            action_type=action_type,
                            order_id=order_id,
                            order_scope="explicit",
                            order_lookup=order_lookup,
                            extra_high_risk_flags=extra_high_risk_flags,
                        )
                    )
                continue
            order_id = _choose_order(order_positions, position, default_order)
            units.append(
                _write_unit(
                    text=segment,
                    action_type=action_type,
                    order_id=order_id,
                    order_scope=(
                        "explicit" if explicit_orders else ("inherited" if default_order else "missing")
                    ),
                    order_lookup=order_lookup,
                    extra_high_risk_flags=extra_high_risk_flags,
                )
            )
        return units

    if policy_question:
        order_id = explicit_orders[0] if explicit_orders else default_order
        return [
            _read_unit(
                intent="policy",
                text=segment,
                order_id=order_id,
                order_scope="explicit" if explicit_orders else ("inherited" if default_order else "missing"),
            )
        ]

    if status_read:
        # A shared read predicate applies to every explicitly named order.
        # Keep independent targets; the executor accepts one order per task.
        return [
            _read_unit(
                intent="order_status",
                text=segment,
                order_id=order_id,
                order_scope="explicit" if explicit_orders else ("inherited" if default_order else "missing"),
            )
            for order_id in dict.fromkeys(explicit_orders or [default_order])
        ]

    if denied_actions:
        # A prohibited action is neither a write nor an implied status query.
        # Explicit read/policy clauses above remain available.
        return []

    if explicit_orders and any(cue in lowered for cue in _READ_TOPIC_CUES):
        # A sentence that names an order but no explicit action is a read.
        return [
            _read_unit(
                intent="order_status",
                text=segment,
                order_id=explicit_orders[0],
                order_scope="explicit",
                semantic_fallback=True,
            )
        ]

    return []


def _choose_order(
    order_positions: list[tuple[int, str]], action_position: int, default_order: str
) -> str:
    if not order_positions:
        return default_order
    before = [order for position, order in order_positions if position <= action_position]
    if before:
        return before[-1]
    return order_positions[0][1]


def _read_unit(
    *, intent: str, text: str, order_id: str, order_scope: str, semantic_fallback: bool = False
) -> SubRequest:
    return SubRequest(
        intent=intent,
        text=text,
        action_type="none",
        order_id=order_id,
        order_scope=order_scope,
        side_effect=False,
        risk_level="low",
        auto_eligible=False,
        evidence_complete=True,
        semantic_fallback=semantic_fallback,
    )


def _write_unit(
    *,
    text: str,
    action_type: str,
    order_id: str,
    order_scope: str,
    order_lookup: Callable[[str], Any] | None,
    extra_high_risk_flags: Iterable[str],
) -> SubRequest:
    order = _lookup_order(order_lookup, order_id) if order_id else None
    missing = _missing_evidence(text, action_type, order_id)
    reasons: list[str] = []
    lowered = text.lower()

    if not order_id or order is None:
        reasons.append("missing_order_evidence")
    if action_type == "complaint_escalation":
        reasons.append("complaint_requires_human")
    if action_type in {"change_address", "invoice_request", "open_support_case"}:
        # These actions have no active auto policy; they are routed to review,
        # but are not labelled high risk merely because they are unsupported.
        pass
    if order is not None:
        if _order_float(order, "payment_value", 0.0) >= 300:
            reasons.append("high_value_order")
        review_score = _order_int(order, "review_score")
        if review_score is not None and review_score <= 2:
            reasons.append("low_customer_rating")
    for term in _HIGH_RISK_TERMS:
        if term in lowered:
            reasons.append(f"request_mentions_{_slug(term)}")
    for flag in extra_high_risk_flags:
        flag_text = str(flag or "").strip()
        if flag_text in _HARD_RISK_FLAGS and flag_text not in reasons:
            reasons.append(flag_text)

    reasons = list(dict.fromkeys(reasons))
    high_risk = bool(reasons)
    auto_eligible = not high_risk and _likely_auto(action_type, order)
    if high_risk:
        risk_level = "high"
    elif auto_eligible:
        risk_level = "low"
    else:
        risk_level = "medium"

    return SubRequest(
        intent="escalation",
        text=text,
        action_type=action_type if action_type in SUPPORTED_ACTIONS else "open_support_case",
        order_id=order_id,
        order_scope=order_scope,
        side_effect=True,
        risk_level=risk_level,
        risk_reasons=reasons,
        auto_eligible=auto_eligible,
        evidence_complete=not missing,
        missing_evidence=missing,
    )


def _likely_auto(action_type: str, order: Any) -> bool:
    if order is None:
        return False
    status = str(_order_attr(order, "status", "") or "")
    amount = _order_float(order, "payment_value", 0.0)
    delay = _order_int(order, "delay_days") or 0
    if action_type == "cancel_order":
        return status in PRE_SHIPMENT_STATUSES and amount <= 300
    if action_type == "refund_request":
        return status == "delivered" and delay >= 10 and amount <= 100
    return False


def _missing_evidence(text: str, action_type: str, order_id: str) -> list[str]:
    missing: list[str] = []
    if not order_id:
        missing.append("order_id")
        return missing
    lowered = text.lower()
    if action_type == "change_address":
        remaining = lowered
        for phrase in ("改地址", "修改地址", "换地址", "地址改", "change address", "update address"):
            remaining = remaining.replace(phrase, "")
        remaining = _ORDER_ID_RE.sub("", remaining)
        remaining = re.sub(r"[\s，,。.；;：:、!?！？]", "", remaining)
        if len(remaining) < 6:
            missing.append("new_address")
    if action_type == "refund_request":
        remaining = lowered
        for phrase in (
            "申请退款", "我要退款", "想退款", "要退款", "退款申请",
            "帮我退款", "退款", "退钱", "refund",
        ):
            remaining = remaining.replace(phrase, "")
        remaining = _ORDER_ID_RE.sub("", remaining)
        remaining = re.sub(r"[\s，,。.；;：:、!?！？]", "", remaining)
        if len(remaining) < 4:
            missing.append("refund_reason")
    if action_type == "invoice_request":
        if not any(cue in lowered for cue in ("抬头", "税号", "邮箱", "invoice title", "tax id")):
            missing.append("invoice_title_or_tax_id")
    return missing


def _append_evidence(unit: SubRequest, segment: str) -> SubRequest:
    text = f"{unit.text}，{segment}"
    if not unit.side_effect:
        return replace(unit, text=text)
    missing = _missing_evidence(text, unit.action_type, unit.order_id)
    return replace(
        unit,
        text=text,
        missing_evidence=missing,
        evidence_complete=not missing,
    )


def _dedupe_units(units: list[SubRequest]) -> list[SubRequest]:
    deduped: list[SubRequest] = []
    seen: set[tuple[str, str, str]] = set()
    for unit in units:
        normalized_text = re.sub(r"\s+", "", unit.text.lower())
        signature = (unit.intent, unit.action_type, f"{unit.order_id}:{normalized_text}")
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(unit)
    return deduped


def _with_dependency(unit: SubRequest, dependency: int) -> SubRequest:
    return replace(unit, dependency_index=dependency - 1)


def _request_risk_level(units: list[SubRequest]) -> str:
    if any(unit.risk_level == "high" for unit in units):
        return "high"
    if any(unit.risk_level == "medium" for unit in units):
        return "medium"
    return "low"


def _lookup_order(
    order_lookup: Callable[[str], Any] | None, order_id: str
) -> Any:
    if not order_lookup or not order_id:
        return None
    try:
        return order_lookup(order_id)
    except Exception:
        return None


def _order_attr(order: Any, name: str, default: Any = None) -> Any:
    if isinstance(order, dict):
        return order.get(name, default)
    return getattr(order, name, default)


def _order_float(order: Any, name: str, default: float = 0.0) -> float:
    value = _order_attr(order, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _order_int(order: Any, name: str) -> int | None:
    value = _order_attr(order, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"\W+", "_", value).strip("_")


__all__ = [
    "ComplexityAssessment",
    "SubRequest",
    "assess_request",
    "SUPPORTED_ACTIONS",
]
