"""Interpret owned-order references with the existing model, behind code-enforced bounds."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.after_sales.address import address_fields, missing_address_fields
from app.after_sales.complexity import assess_request, is_policy_only
from app.agent import order_discovery as rules

logger = logging.getLogger(__name__)
WRITE_GOALS = {
    "refund": "refund_request",
    "cancel": "cancel_order",
    "address": "change_address",
    "invoice": "invoice_request",
    "complaint": "complaint_escalation",
}


class ReferenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["select", "suggest", "list", "clarify"]
    indices: list[int] = Field(default_factory=list, max_length=5)
    goal: Literal[
        "status",
        "refund",
        "cancel",
        "address",
        "invoice",
        "complaint",
        "policy",
        "case_progress",
        "unknown",
        "outside_scope",
    ]
    request_kind: Literal["request", "question", "conditional", "none"]
    reference_quote: str = Field(default="", max_length=160)


def applicable(message: str, context: dict) -> bool:
    if re.search(r"\b[0-9a-fA-F]{32}\b", message):
        return False
    if is_policy_only(message):
        return False
    return not bool(re.fullmatch(r"(?:你好|您好|在吗|嗨|hello|hi|hey)[？?！!。\s]*", message, re.I))


def _choices(context: dict, rows: list[dict], *, suggestion: bool = False) -> dict:
    # Keep the complete displayed-order snapshot for ordinal replies.
    context["candidates"] = rows
    lines = [
        f"{i}. {row['category']} · {str(row['purchased_at'])[:10]} · "
        f"R${row['amount']} · {row['status_label']}"
        for i, row in enumerate(rows, 1)
    ]
    prefix = (
        "您的描述可能对应下面的订单。记录只有商品类别，还不能确认具体商品名称或规格："
        if suggestion
        else "我查到了您本人账号的这些订单："
    )
    suffix = "您说的是这笔吗？" if len(rows) == 1 else "请确认是哪一笔，可以回复序号或直接选择。"
    pending = str(context.get("pending_message") or "")
    action = str(context.get("pending_action") or "track_order")
    return {
        "context": context,
        "answer": prefix + "\n" + "\n".join(lines) + "\n" + suffix,
        "actions": [
            {
                "id": action,
                "label": label,
                "order_id": row["order_id"],
                "message": pending if action != "track_order" else "查询这笔订单的状态",
            }
            for label, row in zip(lines, rows)
        ],
    }


def _explicit_write(message: str, goal: str, order_id: str) -> bool:
    if goal not in WRITE_GOALS:
        return False
    analysis = assess_request(message, selected_order_id=order_id)
    if any(action == WRITE_GOALS[goal] for action, _ in analysis.restricted_actions):
        return False
    if re.search(r"如果|假设|可能|还没决定|没想好|建议|朋友说|不是.*要求", message):
        return False
    # The model classifies the request; code requires an actual action mention,
    # not merely an emotion or a model-invented instruction.
    cues = {
        "refund": r"退|不要了|不想要",
        "cancel": r"取消|不要了",
        "address": r"地址|收货地|送到|换个地方",
        "invoice": r"发票|开票",
        "complaint": r"投诉|举报|转人工",
    }
    return bool(re.search(cues[goal], message)) or (
        goal == "address" and "address" in address_fields(message)
    )


def _affirmation(message: str) -> bool:
    if re.search(r"不是|不对|选错|不要选|别选|没确认", message):
        return False
    return bool(
        re.match(
            r"^(?:对(?:$|[，。！!\s])|是(?:$|[，。！!\s])|对的|是的|没错|确认|就是这个|就是这笔|就这个|就是它)",
            message,
        )
    )


def _multiple_bindings(message: str, proposed: list[dict]) -> str | None:
    replacements = {}
    for row in proposed:
        cues = [
            cue for cue, label in rules._CATEGORIES.items() if label == row["category"] and cue in message
        ]
        if not cues or sum(other["category"] == row["category"] for other in proposed) != 1:
            return None
        replacements[cues[0]] = row["order_id"]
    pattern = "|".join(re.escape(cue) for cue in sorted(replacements, key=len, reverse=True))
    return re.sub(pattern, lambda match: f"订单 {replacements[match[0]]} {match[0]}", message)


def _contrast_tail(message: str) -> str:
    boundaries = list(re.finditer(r"我说的是|而是|[，,；;]\s*是(?=第|电脑|家居|母婴|健康)", message))
    return message[boundaries[-1].end() :] if boundaries else message


async def ground(
    agent,
    message: str,
    context: dict,
    allowed: set[str],
    history: list[dict],
    selected_order_id: str | None = None,
    cases: list[dict] | None = None,
) -> dict:
    context = dict(context)
    if context.get("allowed") != sorted(allowed) or time.time() - context.get("at", 0) > 900:
        context = {}
    fresh = not context.get("orders") or context.get("failed")
    if fresh:
        try:
            result = await asyncio.to_thread(agent._execute, "search_user_orders", {"limit": 5}, allowed)
            if result.get("error"):
                raise ValueError("lookup unavailable")
            rows = [row for row in result["orders"] if row.get("order_id") in allowed]
        except Exception:
            return {
                "context": {"allowed": sorted(allowed), "at": time.time(), "failed": True},
                "answer": "暂时没能读取您当前账号的订单，请稍后让我重试。现在不会替您提交售后。",
            }
        context = {
            "allowed": sorted(allowed),
            "at": time.time(),
            "orders": rows,
            "candidates": rows,
            "pending_message": message,
        }
    rows = [row for row in context["orders"] if row["order_id"] in allowed]
    displayed = [row for row in context.get("candidates", rows) if row["order_id"] in allowed]
    if not rows:
        return {
            "context": context,
            "answer": "当前登录账号下没有查到订单。如果是用其他账号购买，请切换到购买时的账号后再查询。",
        }
    if selected_order_id in allowed:
        context["selected_order_id"] = selected_order_id
    if not agent.available:
        fallback = rules.ground(agent, message, context, allowed)
        fallback["context"]["semantic_status"] = "offline_rule_fallback"
        return fallback
    # Indices follow the displayed list. Undisplayed orders get appended, so
    # category switches are possible without corrupting "第二个" semantics.
    pool = displayed + [row for row in rows if row["order_id"] not in {r["order_id"] for r in displayed}]
    try:
        decision = await asyncio.wait_for(
            agent.interpret_reference(message, pool, history, context),
            timeout=18,
        )
        if not isinstance(decision, ReferenceDecision):
            decision = ReferenceDecision.model_validate(decision)
    except Exception as exc:
        logger.warning("order reference unavailable: %s", type(exc).__name__)
        fallback = rules.ground(agent, message, context, allowed)
        fallback["context"]["semantic_status"] = "model_error_rule_fallback"
        return fallback
    context["semantic_status"] = "model_validated"
    context.pop("routing_override", None)
    context["reference_decision"] = decision.model_dump()
    if decision.goal == "outside_scope":
        return {"context": context, "passthrough": True}
    constraints = assess_request(message, selected_order_id=rows[0]["order_id"]).restricted_actions
    if constraints or re.search(r"先别|不要办|暂时不|先不|没决定", message):
        context["pending_message"] = ""
        context["pending_action"] = ""
    if len(set(decision.indices)) != len(decision.indices) or any(
        i < 1 or i > len(pool) for i in decision.indices
    ):
        context["semantic_status"] = "invalid_index_rejected"
        return _choices(context, displayed or rows)
    # An order lifecycle question is not an existing Case lookup. In the
    # absence of a Case/topic cue, validate a mislabel against explicit status
    # language before allowing the progress shortcut to end the turn.
    if (
        decision.goal == "case_progress"
        and not cases
        and re.search(r"买的|电脑|家居|母婴|健康", message)
        and re.search(r"现状|状态|怎么样|取消.*(?:生效|成功|了吗|了吧)", message)
        and not re.search(r"退款|售后|工单|申请|审核|申诉", message)
        and not assess_request(message).has_explicit_write_request
    ):
        decision = decision.model_copy(update={"goal": "status", "request_kind": "question"})
        context["routing_override"] = {
            "goal": "status", "reason": "explicit_order_status_without_case",
        }
    selected = None
    proposed = [pool[i - 1] for i in decision.indices]
    mentioned = [
        row
        for row in pool
        if any(cue in message and row["category"] == label for cue, label in rules._CATEGORIES.items())
    ]
    if (
        len(mentioned) > 1
        and _contrast_tail(message) == message
        and decision.decision == "select"
        and (decision.request_kind == "request" or decision.goal == "status")
        and not re.search(r"如果|假设|还没决定|朋友说|建议", message)
    ):
        if all(sum(other["category"] == row["category"] for other in pool) == 1 for row in mentioned):
            bound_message = _multiple_bindings(message, mentioned)
            if bound_message and (
                decision.request_kind == "request" or not assess_request(bound_message).write_units
            ):
                return {
                    "context": context,
                    "order_id": proposed[0]["order_id"],
                    "message": bound_message,
                    "read_only": False,
                }
    literal = rules._selected(message, pool, str(context.get("selected_order_id") or ""))
    if not literal and _contrast_tail(message) != message:
        literal = rules._selected(_contrast_tail(message), pool, "")
    # A validated page selection plus an unambiguous deictic status question
    # already supplies the target; a model "list" result must not discard it.
    if (
        selected_order_id in allowed
        and any(row["order_id"] == selected_order_id for row in pool)
        and re.search(r"这笔|这件|这个|它", message)
        and re.search(r"查|发货|配送|物流|状态|发没发", message)
        and not re.search(r"不是这|不是那|不选|不要选|别选|选错|不对", message)
        and (not literal or literal == selected_order_id)
        and not assess_request(message, selected_order_id=selected_order_id).write_units
    ):
        context["selected_order_id"] = selected_order_id
        context["routing_override"] = {"goal": "status", "reason": "validated_page_reference"}
        return {"context": context, "order_id": selected_order_id,
                "message": message, "read_only": True, "action": "track_order"}
    if decision.decision == "select" and literal and any(row["order_id"] == literal for row in proposed):
        # A unique literal reference is stronger than extra model suggestions.
        # Discard ungrounded additions, but never substitute an absent target.
        proposed = [row for row in proposed if row["order_id"] == literal]
    if decision.decision == "select" and len(proposed) == 1:
        proposed_id = proposed[0]["order_id"]
        # Semantic similarity alone is a suggestion, not factual identification.
        previous_suggestion = context.get("suggested_ids") == [proposed_id]
        category_words = re.findall(r"[a-z]+", str(proposed[0].get("category_key") or ""))
        direct_key = len(category_words) >= 2 and all(
            re.search(r"\b" + word + r"\b", message, re.I) for word in category_words
        )
        if direct_key:
            first_word = re.search(r"\b" + category_words[0] + r"\b", message, re.I)
            direct_key = not re.search(
                r"\b(?:not|isn't|wrong)\b[^,.;]{0,24}$", message[: first_word.start()], re.I
            )
        if literal == proposed_id or direct_key or (previous_suggestion and _affirmation(message)):
            selected = proposed_id
        elif fresh and len(allowed) == 1 and decision.goal == "status":
            selected = proposed_id
        elif selected_order_id == proposed_id and re.search(r"这笔|这件|这个|它", message):
            selected = proposed_id
    if selected and len(allowed) > 5 and not re.search(r"第[一二三四五12345](?:个|件|笔|张)", message):
        selected = None
    current_id = str(context.get("selected_order_id") or "")
    address_pending = context.get("address_pending") or {}
    if address_pending and re.search(r"先别|不要办|暂时不|先不|算了|不用|不改|没决定", message):
        context.pop("address_pending", None)
        address_pending = {}
    if any(action == "change_address" for action, _ in constraints):
        context.pop("address_pending", None)
        address_pending = {}
    if (
        not selected
        and address_pending.get("order_id") in allowed
        and decision.goal == "address"
        and decision.request_kind == "request"
        and address_fields(message)
        and len(proposed) == 1
        and proposed[0]["order_id"] == address_pending["order_id"]
        and not literal
    ):
        selected = address_pending["order_id"]
    if decision.goal == "case_progress":
        # A progress label is not permission to discard a separate explicit
        # order query. Only peel off read clauses with a unique literal target;
        # never infer a new write or resolve an ambiguous product reference.
        query_clauses = []
        query_ids = []
        for clause in re.split(r"[，,；;。\n]", message):
            target = rules._selected(clause, pool, "")
            if (
                target
                and any(cue in clause for cue in rules._CATEGORIES)
                and re.search(r"查|查看|看.*(?:发货|配送|状态|发没发)", clause)
                and not assess_request(clause, selected_order_id=target).write_units
            ):
                query_clauses.append(clause)
                query_ids.append(target)
        if query_clauses and not assess_request(message).has_explicit_write_request:
            if len(set(query_ids)) == 1:
                context["selected_order_id"] = query_ids[0]
                context["suggested_ids"] = []
            return {
                "context": context,
                "order_id": query_ids[0],
                "message": "；".join(
                    f"订单 {target} {clause}"
                    for target, clause in zip(query_ids, query_clauses)
                ),
                "read_only": False,
            }
        safe_cases = cases or []
        if safe_cases:
            status = {
                "pending_review": "待审核",
                "executed": "已处理",
                "rejected": "已拒绝",
                "appealed_pending_review": "申诉后待审核",
                "approving": "处理中",
            }
            descriptions = [status.get(str(case.get("status")), "处理中") for case in safe_cases]
            return {
                "context": context,
                "answer": "本会话已有售后申请："
                + "、".join(descriptions)
                + "。您可以在售后进度中查看详情；本次查询不会重复提交申请。",
            }
        return {
            "context": context,
            "answer": "本会话没有查到已有售后申请。若是在其他会话提交，请打开原会话的售后进度查看，"
            "我不会替您重新提交。",
        }
    if selected:
        context["selected_order_id"] = selected
        context["suggested_ids"] = []
        # A reference decision binds an order, not a whole compound task plan.
        # Preserve the existing decomposer when the user asks for several actions.
        analysis = assess_request(message, selected_order_id=selected)
        if (
            analysis.is_complex
            and (decision.request_kind == "request" or analysis.has_explicit_write_request)
            and not re.search(r"如果|假设|可能|还没决定|没想好|建议|朋友说", message)
        ):
            return {
                "context": context,
                "order_id": selected,
                "message": message,
                "read_only": False,
            }
        if decision.goal in WRITE_GOALS and decision.request_kind == "request":
            task = _contrast_tail(message) if literal == selected else message
            if decision.goal == "address" and address_pending.get("order_id") == selected:
                task = address_pending["text"] + "；" + task
            if re.search(r"如果|假设|可能|还没决定|没想好|建议|朋友说", message):
                return {
                    "context": context,
                    "answer": "这次先不提交售后申请。您可以先查询配送或了解处理条件。",
                }
            if not _explicit_write(task, decision.goal, selected):
                pending = str(context.get("pending_message") or "")
                if _affirmation(message) or literal == selected:
                    task = pending + "；" + message
            if _explicit_write(task, decision.goal, selected):
                context["pending_message"] = ""
                context["pending_action"] = ""
                if decision.goal == "address" and missing_address_fields(task):
                    context["address_pending"] = {"order_id": selected, "text": task}
                else:
                    context.pop("address_pending", None)
                return {
                    "context": context,
                    "order_id": selected,
                    "message": task,
                    "action": WRITE_GOALS[decision.goal],
                }
        if decision.goal == "status":
            return {
                "context": context,
                "order_id": selected,
                "message": message,
                "read_only": True,
                "action": "track_order",
            }
        if decision.request_kind == "conditional":
            return {
                "context": context,
                "answer": "这次先不提交售后申请。您可以先查询配送，或了解相应处理条件。",
            }
        if decision.goal == "policy" or decision.request_kind == "question":
            return {
                "context": context,
                "order_id": selected,
                "message": message,
                "read_only": False,
                "action": "refund_policy",
            }
        return {"context": context, "answer": "已确认这笔订单。您希望查询进度，还是具体处理什么问题？"}
    if decision.goal in WRITE_GOALS and decision.request_kind == "request":
        if _explicit_write(message, decision.goal, rows[0]["order_id"]):
            context["pending_message"] = message
            context["pending_action"] = WRITE_GOALS[decision.goal]
    if proposed and decision.decision in {"select", "suggest"}:
        context["suggested_ids"] = [row["order_id"] for row in proposed]
        return _choices(context, proposed, suggestion=True)
    if decision.decision == "list" and proposed:
        return _choices(context, proposed)
    # Preserve a confirmed target for a new explicit request referring to it,
    # but never grant authority from a model goal without user text.
    if current_id and current_id in allowed and _affirmation(message):
        return {"context": context, "answer": "已确认这笔订单。请告诉我希望查询或处理的具体问题。"}
    return _choices(context, displayed or rows)
