"""Render the authoritative task outcomes without interpreting user intent again."""

from app.agent.customer_order_agent import CATEGORY_LABELS

STATUS = {
    "delivered": "已送达",
    "canceled": "已取消",
    "invoiced": "已开票、待发货",
    "shipped": "运输中",
    "processing": "处理中",
    "approved": "已确认",
    "created": "待处理",
}
LABELS = {
    "status": "当前状态",
    "purchase_timestamp": "下单时间",
    "estimated_delivery_date": "预计送达时间",
    "delivered_customer_date": "实际签收时间",
    "delay_days": "延迟天数",
    "payment_value": "支付金额",
    "cancellation_reason": "取消原因",
    "realtime_location": "实时位置",
    "product_name": "具体商品名称",
    "product_model": "商品型号",
    "suitability": "适用性",
}


def order_fields_reply(data: dict, required: list[str]) -> str:
    if not data.get("found"):
        return "暂未查到可靠的订单事实，请稍后重试；本项没有提交售后申请。"
    facts = data.get("facts") or {}
    parts = []
    answered = False
    for field in dict.fromkeys(required or ["status"]):
        value = facts.get(field)
        if value is None or value == "":
            parts.append(f"当前记录不包含{LABELS.get(field, '这项信息')}，无法据此确认。")
            continue
        answered = True
        if field == "status":
            parts.append("当前状态：" + STATUS.get(value, str(value)) + "。")
        elif field == "payment_value":
            parts.append(f"支付金额：R${value}。")
        elif field == "delay_days":
            parts.append(f"配送延迟：{value}天。")
        else:
            parts.append(f"{LABELS.get(field, field)}：{value}。")
    if not answered and facts.get("status"):
        # Nothing asked for is on record: give the nearest known fact instead of only "unknown".
        parts.append("目前能确认的是当前状态：" + STATUS.get(facts["status"], str(facts["status"])) + "。")
    return "".join(parts)


def contract_reply(state: dict) -> str:
    plan = state.get("validated_turn_plan")
    if not plan:
        return ""
    # Late import keeps the legacy presenter usable without a circular import.
    from app.llm.customer_presenter import _deterministic_customer_reply

    reads = {r["task_index"]: r for r in state.get("customer_read_results", [])}
    records = state.get("customer_case_records") or []
    orders = {o["order_id"]: o for o in (state.get("customer_order_context") or {}).get("orders", [])}
    answers = []
    for index, task in enumerate(state.get("task_plan") or []):
        result = reads.get(index) or {}
        order_id = task.get("order_id")
        order = orders.get(order_id, {})
        category = order.get("category", "")
        label = CATEGORY_LABELS.get(category, category) or f"第{index + 1}项"
        if order.get("amount") is not None:
            label += f"（R${order['amount']}）"
        if task["intent"] == "order_status":
            tool = result.get("tool_result") or {}
            answer = (
                order_fields_reply(tool.get("data") or {}, task.get("required_fields") or ["status"])
                if tool.get("ok")
                else "订单查询暂未完成，请稍后重试。"
            )
        elif task.get("side_effect"):
            action = task["action_type"]
            case = next(
                (c for c in records if c.get("order_id") == order_id and c.get("action_type") == action), None
            )
            assessed = [
                c
                for c in state.get("after_sales_cases", [])
                if c.get("order_id") == order_id and c.get("action_type") == action
            ]
            single = {
                **state,
                "validated_turn_plan": None,
                "task_plan": [task],
                "pending_side_effect": {},
                "after_sales_cases": assessed,
                "trajectory_events": [],
            }
            answer = (
                _deterministic_customer_reply(task["text"], single, case)
                or "尚未确认这项申请是否受理，请补充信息或查看进度。"
            )
        else:
            answer = result.get("answer") or "本项尚未完成，请稍后重试；没有因此提交新申请。"
        answers.append(f"{label}：{answer}" if len(state.get("task_plan", [])) > 1 or order_id else answer)
    return "\n\n".join(answers)
