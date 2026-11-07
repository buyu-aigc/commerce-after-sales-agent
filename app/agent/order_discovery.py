"""Owned-order discovery and explicit selection; no business writes or approvals."""

from __future__ import annotations

import re
import time

from app.after_sales.complexity import assess_request

_CATEGORIES = {
    "电脑": "电脑配件",
    "母婴": "母婴用品",
    "家居": "家居用品",
    "健康与美容": "健康与美容用品",
}
_READ = ("查", "物流", "配送", "发货", "到哪", "状态", "没收到", "没来", "还没到", "看看")
_POLICY = ("政策", "规则", "一般", "条件", "需要什么", "哪些资料")


def applicable(message: str, context: dict) -> bool:
    if re.search(r"\b[0-9a-fA-F]{32}\b", message):
        return False
    if any(word in message for word in _POLICY):
        return False
    return bool(context) or any(
        word in message
        for word in (
            "买的",
            "买过",
            "买了",
            "东西",
            "快递",
            "电脑",
            "母婴",
            "家居",
            "售后",
        )
    )


def _writes(message: str, candidates: list[dict]) -> bool:
    if not candidates:
        return False
    return bool(assess_request(message, selected_order_id=candidates[0]["order_id"]).write_units)


def _selected(message: str, candidates: list[dict], selected_id: str) -> str | None:
    # Ordinals refer to the persisted display order, never an unordered set.
    if re.search(r"不选|不要选|别选|选错|不对|(?<!是)不是(?=这|那|第|我说|电脑|家居|母婴|健康)", message):
        return None
    if len(re.findall(r"第[一二三四五12345](?:个|件|笔|张)", message)) > 1:
        return None
    categories = {label for cue, label in _CATEGORIES.items() if cue in message}
    if len(categories) > 1:
        return None
    ordinal = re.search(r"第([一二三四五12345])(?:个|件|笔|张)", message)
    if ordinal:
        position = "一二三四五".find(ordinal[1]) + 1 if ordinal[1] in "一二三四五" else int(ordinal[1])
        return candidates[position - 1]["order_id"] if position <= len(candidates) else None
    matches = candidates
    category = next(iter(categories), None)
    if category:
        matches = [order for order in matches if order["category"] == category]
    if any(word in message for word in ("比较晚", "较晚", "最晚", "最近买")) and matches:
        newest = max(order["purchased_at"] for order in matches)
        matches = [order for order in matches if order["purchased_at"] == newest]
        category = category or "date"
    if "金额" in message and any(word in message for word in ("较高", "最高", "最贵")) and matches:
        highest = max(float(order["amount"]) for order in matches)
        matches = [order for order in matches if float(order["amount"]) == highest]
        category = category or "amount"
    if category:
        return matches[0]["order_id"] if len(matches) == 1 else None
    if any(word in message for word in ("另一个", "另一件", "别的")):
        return None
    if selected_id and any(word in message for word in ("这个", "这笔", "这件", "它", "再查", "刚才")):
        return selected_id if any(o["order_id"] == selected_id for o in candidates) else None
    return None


def ground(agent, message: str, context: dict, allowed: set[str]) -> dict:
    """Read the existing owned-order tool, then return a choice or a grounded turn.

    Arbitrary product names are not inferred from a category. Unrecognized
    references remain unbound; the original planner still handles intent
    after selection. Every stored candidate is rechecked against ownership.
    """
    context = dict(context)
    if context.get("allowed") != sorted(allowed) or time.time() - context.get("at", 0) > 900:
        context = {}
    fresh = not context.get("orders") or context.get("failed")
    if fresh:
        try:
            result = agent._execute("search_user_orders", {"limit": 5}, allowed)
            if result.get("error"):
                raise ValueError("owned order lookup unavailable")
            orders = list(result["orders"])
        except Exception:
            return {
                "context": {"allowed": sorted(allowed), "at": time.time(), "failed": True},
                "answer": "暂时没能读取您当前账号的订单，请稍后让我重试。"
                "现在还不能确认是哪件商品，也不会替您提交售后。",
            }
        context = {
            "allowed": sorted(allowed),
            "at": time.time(),
            "orders": orders,
            "candidates": orders,
            "pending_message": message,
        }
    orders = [order for order in context["orders"] if order["order_id"] in allowed]
    candidates = [order for order in context.get("candidates", orders) if order["order_id"] in allowed]
    if not orders:
        return {
            "context": context,
            "answer": "当前登录账号下没有查到订单。您可以核实是否用其他账号购买；我只能读取当前账号的订单。",
        }
    list_only = "列给我" in message or "列出来" in message
    if any(word in message for word in ("没收到", "快递", "没来", "还没到")):
        candidates = [
            order
            for order in orders
            if order["status"] in {"created", "approved", "invoiced", "processing", "shipped"}
        ]
    else:
        # Explicit category changes can leave the last filtered list.
        if any(cue in message for cue in _CATEGORIES):
            candidates = orders
    selected = _selected(message, candidates, str(context.get("selected_order_id") or ""))
    if len(allowed) > 5 and not re.search(r"第[一二三四五12345](?:个|件|笔|张)", message):
        # A unique category in a truncated preview is not a unique owned order.
        selected = None
    read = any(word in message for word in _READ)
    if (
        not selected
        and fresh
        and len(allowed) == 1
        and len(candidates) == 1
        and read
        and not _writes(message, candidates)
    ):
        selected = candidates[0]["order_id"]
    if selected and not list_only:
        context["selected_order_id"] = selected
        context["candidates"] = candidates
        analysis = assess_request(message, selected_order_id=selected)
        if (
            analysis.units
            and all(unit.intent == "policy" for unit in analysis.units)
            and not any(word in message for word in ("物流", "配送", "发货", "到哪", "快递"))
        ):
            # Resolving the order must not turn an eligibility/policy question
            # into shipping status just because it also contains "查一下".
            return {"context": context, "order_id": selected, "message": message, "read_only": False}
        task_message = message
        if not read and not _writes(message, candidates):
            pending = str(context.get("pending_message") or "")
            constraints = assess_request(message, selected_order_id=selected).restricted_actions
            if constraints or re.search(r"先别|不要办|暂时不|先不|没决定", message):
                context["pending_message"] = ""
                return {
                    "context": context,
                    "answer": "已确认这笔订单，本轮不会提交售后申请。您想先了解什么？",
                }
            if _writes(pending, candidates) or any(cue in pending for cue in _READ):
                task_message = pending + "；客户已确认是这笔订单。"
            else:
                return {
                    "context": context,
                    "answer": "已确认您指的是这笔订单。您想查询配送，还是具体有什么商品或售后问题？",
                }
        elif read and not _writes(message, candidates):
            task_message = "查询这笔订单的物流状态。" + message
        context["pending_message"] = ""
        return {
            "context": context,
            "order_id": selected,
            "message": task_message,
            "read_only": read and not _writes(message, candidates),
        }
    context["candidates"] = candidates
    if not candidates:
        return {
            "context": context,
            "answer": "当前账号中没有查到符合这次描述的未收货订单。可以换个描述，或查看已有订单记录。",
        }
    lines = []
    actions = []
    pending = str(context.get("pending_message") or "")
    pending_units = assess_request(pending, selected_order_id=candidates[0]["order_id"]).write_units
    action_id = pending_units[0].action_type if len(pending_units) == 1 else "track_order"
    for index, order in enumerate(candidates, 1):
        label = (
            f"{index}. {order['category']} · {str(order['purchased_at'])[:10]} · "
            f"R${order['amount']} · {order['status_label']}"
        )
        lines.append(label)
        actions.append(
            {
                "id": action_id,
                "label": label,
                "order_id": order["order_id"],
                "message": pending if pending_units else "查询这笔订单的当前状态",
            }
        )
    answer = "我先列出您当前账号最近的几笔订单记录：\n" + "\n".join(lines)
    answer += "\n请确认是哪一笔，可以回复序号或选择对应订单，无需提供订单号。"
    return {"context": context, "answer": answer, "actions": actions}
