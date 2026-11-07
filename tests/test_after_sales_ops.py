from app.olist.service import InMemoryCaseService, OlistService, SQLiteCaseService, format_after_sales_report


def test_after_sales_priority_report_is_ranked_and_read_only() -> None:
    service = OlistService()
    report = service.after_sales_priority_report("生成审核台优先处理队列")

    categories = report["high_risk_categories"]
    orders = report["priority_orders"]

    assert categories
    assert orders
    assert [item["risk_score"] for item in categories] == sorted(
        [item["risk_score"] for item in categories],
        reverse=True,
    )
    assert [item["priority_score"] for item in orders] == sorted(
        [item["priority_score"] for item in orders],
        reverse=True,
    )
    assert any("HITL" in rule for rule in report["decision_rules"])

    rendered = format_after_sales_report(report)
    assert "高风险类目 Top" in rendered
    assert "优先跟进订单 Top" in rendered
    assert "只读" in rendered


def test_case_service_returns_duplicate_for_same_business_action() -> None:
    service = InMemoryCaseService()
    first = service.execute_action(
        action_type="refund_request",
        order_id="203096f03d82e0dffbc41ebc2e2bcfb7",
        message_text="delivery delayed by 11 day(s); low review score 2",
    )
    second = service.execute_action(
        action_type="refund_request",
        order_id="203096f03d82e0dffbc41ebc2e2bcfb7",
        message_text="delivery delayed by 11 day(s); low review score 2",
    )

    assert first["result_id"] == second["result_id"]
    assert first["duplicate"] is False
    assert second["duplicate"] is True


def test_case_idempotency_ignores_minor_text_changes() -> None:
    service = InMemoryCaseService()
    first = service.execute_action(
        action_type="invoice_request",
        order_id="203096f03d82e0dffbc41ebc2e2bcfb7",
        message_text="please create invoice request",
    )
    second = service.execute_action(
        action_type="invoice_request",
        order_id="203096f03d82e0dffbc41ebc2e2bcfb7",
        message_text="麻烦为这个订单申请发票，谢谢",
    )

    assert first["result_id"] == second["result_id"]
    assert second["duplicate"] is True


def test_sqlite_case_service_persists_idempotency(tmp_path) -> None:
    db_path = tmp_path / "cases.db"
    first_service = SQLiteCaseService(db_path)
    first = first_service.execute_action(
        action_type="refund_request",
        order_id="203096f03d82e0dffbc41ebc2e2bcfb7",
        message_text="delivery delayed by 11 day(s); low review score 2",
    )

    second_service = SQLiteCaseService(db_path)
    second = second_service.execute_action(
        action_type="refund_request",
        order_id="203096f03d82e0dffbc41ebc2e2bcfb7",
        message_text="订单延迟且低分，申请退款",
    )

    assert first["result_id"] == second["result_id"]
    assert second["duplicate"] is True
    assert second_service.get(str(first["result_id"])) is not None
