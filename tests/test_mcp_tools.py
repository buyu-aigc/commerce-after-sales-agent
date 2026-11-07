from app.mcp_server import (
    assess_after_sales_case,
    draft_escalation,
    generate_after_sales_priority_report,
    get_order_status,
    list_enterprise_tool_boundaries,
    search_category_risk,
)

ORDER_ID = "203096f03d82e0dffbc41ebc2e2bcfb7"


def test_mcp_order_status_tool() -> None:
    result = get_order_status(ORDER_ID)
    assert ORDER_ID in result
    assert "delivered" in result


def test_mcp_category_risk_tool() -> None:
    result = search_category_risk("health beauty 类目")
    assert result[0]["name"] == "health_beauty"


def test_mcp_after_sales_priority_report_tool() -> None:
    result = generate_after_sales_priority_report("生成审核台优先处理队列")
    assert result["high_risk_categories"]
    assert result["priority_orders"]
    assert "HITL" in result["decision_rules"][-1]


def test_mcp_escalation_tool() -> None:
    result = draft_escalation(ORDER_ID)
    assert result["found"] is True
    assert result["order_id"] == ORDER_ID


def test_mcp_after_sales_case_assessment_tool() -> None:
    result = assess_after_sales_case(
        "refund_request",
        ORDER_ID,
        f"订单 {ORDER_ID} 延迟送达，我想申请退款",
    )

    assert result["found"] is True
    assert result["decision"]["outcome"] == "needs_human_review"
    assert result["verification"]["required_next_step"] == "hitl"


def test_mcp_enterprise_tool_boundaries_include_risk_and_auth() -> None:
    metadata = {item["tool"]: item for item in list_enterprise_tool_boundaries()}

    assert metadata["get_order_status"]["auth_scope"] == "orders:read"
    assert "input_schema" in metadata["get_order_status"]
    assert metadata["assess_after_sales_case"]["risk_level"] == "medium"
    assert metadata["execute_side_effect"]["side_effect"] is True
    assert metadata["execute_side_effect"]["idempotency_required"] is True
