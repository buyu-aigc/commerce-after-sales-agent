"""P0-1 Playwright E2E: customer case creation → review → progress.

Every test runs against a real FastAPI + LangGraph + SQLite instance in a
disposable directory.  The browser is a real headless Chromium; assertions are
made on the rendered product pages, with API calls only used for setup and for
idempotency/authorization checks that the UI intentionally does not expose.
"""

from __future__ import annotations

import uuid

from conftest import (
    ORDER_DELAYED,
    ORDER_HIGH_VALUE,
    ORDER_OTHER_OWNER,
    REVIEW_TOKEN,
    boot_customer,
    create_case,
    customer_progress,
    decide_case,
    new_context,
    review_case,
    review_login,
    save_screenshot,
)


def _session(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _wait_queue_count(page, count: int) -> None:
    page.wait_for_function(
        "(expected) => document.querySelector('#count')?.textContent.trim() === String(expected)",
        arg=count,
    )


def test_customer_creates_case_and_sees_pending_progress(page, app_server, artifacts_dir):
    session_id = boot_customer(page, app_server)
    card = page.locator(f'article.order[data-order-id="{ORDER_DELAYED}"]')
    card.wait_for()
    card.locator("button", has_text="申请售后").first.click()

    page.wait_for_selector("#progress .progress-case")
    page.locator("#progress").get_by_text("投诉处理 · 等待审核").wait_for()

    progress = customer_progress(page, app_server, session_id)
    assert progress["status"] == 200
    cases = progress["body"]["cases"]
    assert len(cases) == 1
    case = cases[0]
    assert case["status"] == "pending_review"
    assert case["action_type"] == "complaint_escalation"
    assert case["case_id"]
    assert "..." in case["order_id"]

    save_screenshot(page, artifacts_dir, "customer-pending-progress")


def test_review_approve_shows_executed_and_second_approve_is_idempotent(page, app_server, artifacts_dir):
    session_id = _session("e2e-approve")
    case_id = create_case(page, app_server, session_id=session_id, order_id=ORDER_HIGH_VALUE)

    review_login(page, app_server)
    page.wait_for_selector(".queue-item")
    page.click(".queue-item")
    page.wait_for_selector("#detail:not([hidden])")
    assert "待审核" in page.text_content("#caseStatus")
    assert page.is_enabled("#approve")

    page.click("#approve")
    page.wait_for_function(
        "() => document.querySelector('#caseStatus')?.textContent.includes('已执行')"
    )
    assert page.is_disabled("#approve")
    _wait_queue_count(page, 0)
    save_screenshot(page, artifacts_dir, "review-executed")

    second = decide_case(page, app_server, case_id, "approve")
    assert "已处理" in second["answer"]

    packet = review_case(page, app_server, case_id)
    assert packet["status"] == "executed"
    assert packet["has_pending"] is False

    boot_customer(page, app_server, session_id)
    page.wait_for_selector("#progress .progress-case")
    page.locator("#progress").get_by_text("退款申请 · 已处理").wait_for()
    save_screenshot(page, artifacts_dir, "customer-executed-progress")


def test_multiple_cases_can_be_selected_approved_and_rejected(page, app_server, artifacts_dir):
    session_id = _session("e2e-multi")
    refund_id = create_case(
        page,
        app_server,
        session_id=session_id,
        action_type="refund_request",
        order_id=ORDER_HIGH_VALUE,
    )
    complaint_id = create_case(
        page, app_server, session_id=session_id, action_type="complaint_escalation"
    )
    assert refund_id != complaint_id

    review_login(page, app_server)
    _wait_queue_count(page, 2)
    page.locator('.queue-item:has-text("退款申请")').first.click()
    page.wait_for_selector("#casePicker:not([hidden])")
    assert page.locator(".case-chip").count() == 2

    page.locator('.case-chip:has-text("投诉处理")').first.click()
    page.wait_for_function(
        "() => document.querySelector('#caseTitle')?.textContent.trim() === '投诉处理'"
    )
    page.click("#approve")
    page.wait_for_function(
        "() => document.querySelector('#caseStatus')?.textContent.includes('已执行')"
    )
    _wait_queue_count(page, 1)

    page.locator(".queue-item").first.click()
    page.wait_for_function(
        "() => document.querySelector('#caseTitle')?.textContent.trim() === '退款申请'"
    )
    page.click("#reject")
    page.wait_for_function(
        "() => document.querySelector('#caseStatus')?.textContent.includes('已驳回')"
    )
    _wait_queue_count(page, 0)
    save_screenshot(page, artifacts_dir, "review-multi-case")

    assert review_case(page, app_server, complaint_id)["status"] == "executed"
    assert review_case(page, app_server, refund_id)["status"] == "rejected"

    boot_customer(page, app_server, session_id)
    page.wait_for_selector("#progress .progress-case")
    page.locator("#progress").get_by_text("投诉处理 · 已处理").wait_for()
    page.locator("#progress").get_by_text("退款申请 · 未通过审核").wait_for()
    save_screenshot(page, artifacts_dir, "customer-multi-progress")


def test_cross_user_case_and_order_access_is_denied(page, app_server, browser):
    session_id = _session("e2e-owner")
    case_id = create_case(page, app_server, session_id=session_id, order_id=ORDER_HIGH_VALUE)

    response = page.request.get(
        f"{app_server.base_url}/customer/cases/{case_id}",
        headers={"X-Demo-Customer": "customer-smoke"},
    )
    assert response.status == 403

    response = page.request.get(
        f"{app_server.base_url}/customer/sessions/{session_id}/progress",
        headers={"X-Demo-Customer": "customer-smoke"},
    )
    assert response.status == 403

    response = page.request.get(
        f"{app_server.base_url}/customer/cases/{case_id}",
        headers={"X-Demo-Customer": "not-a-demo-user"},
    )
    assert response.status == 401

    response = page.request.post(
        f"{app_server.base_url}/customer/chat",
        headers={"X-Demo-Customer": "customer-smoke"},
        data={
            "message": "我想为这笔订单申请退款。",
            "session_id": "cross-user-selected-order",
            "requested_action": "refund_request",
            "page_context": {"surface": "order_detail", "selected_order_id": ORDER_OTHER_OWNER},
        },
    )
    assert response.status == 403

    # The message-level guard is a safe 200 answer, not a silent cross-user write.
    response = page.request.post(
        f"{app_server.base_url}/customer/chat",
        headers={"X-Demo-Customer": "customer-smoke"},
        data={"message": f"帮我处理订单 {ORDER_OTHER_OWNER} 的退款", "session_id": "cross-user-message"},
    )
    assert response.status == 200
    payload = response.json()
    assert payload.get("case_id") is None
    assert "隐私" in payload["answer"]

    # The customer UI for another allowed actor only renders that actor's orders.
    context = new_context(browser, extra_http_headers={"X-Demo-Customer": "customer-smoke"})
    try:
        other = context.new_page()
        other.goto(f"{app_server.base_url}/customer")
        other.wait_for_selector("#orders article.order")
        assert other.locator("#orders article.order").count() == 1
    finally:
        context.close()


def test_empty_review_queue_and_error_states(page, app_server, browser, artifacts_dir):
    page.goto(f"{app_server.base_url}/review")
    page.fill("#token", "wrong-token")
    page.click("#login button")
    page.wait_for_function(
        "() => document.querySelector('#notice')?.textContent.includes('访问凭证无效')"
    )

    page.fill("#token", REVIEW_TOKEN)
    page.click("#login button")
    _wait_queue_count(page, 0)
    page.wait_for_function(
        "() => document.querySelector('#notice')?.textContent.includes('当前没有待审 Case')"
    )
    assert page.locator(".queue-item").count() == 0
    save_screenshot(page, artifacts_dir, "review-empty")

    boot_customer(page, app_server)
    page.wait_for_selector("#availability:not([hidden])")
    assert "基础服务模式" in page.text_content("#availability")

    context = new_context(browser, extra_http_headers={"X-Demo-Customer": "not-a-demo-user"})
    try:
        invalid = context.new_page()
        invalid.goto(f"{app_server.base_url}/customer")
        invalid.wait_for_function(
            "() => document.querySelector('#orders')?.textContent.includes('暂时无法完成操作')"
        )
    finally:
        context.close()


def test_complex_request_shows_processed_and_pending_sub_cases_together(
    page, app_server, artifacts_dir
):
    low_value_pre_ship = "8aec3a066f732dd927ec8fef1752415b"
    session_id = _session("e2e-partial-auto")
    response = page.request.post(
        f"{app_server.base_url}/customer/chat",
        data={
            "message": (
                f"订单 {low_value_pre_ship} 帮我取消，"
                f"订单 {ORDER_HIGH_VALUE} 我要投诉"
            ),
            "session_id": session_id,
        },
    )
    assert response.ok, response.text()
    payload = response.json()
    assert payload["request_analysis"]["tier"] == "complex"
    assert payload["request_analysis"]["partial_auto"] is True
    actions = {item["action_type"]: item for item in payload["sub_cases"]}
    assert actions["cancel_order"]["status"] == "executed"
    assert actions["complaint_escalation"]["status"] == "pending_review"

    boot_customer(page, app_server, session_id)
    page.wait_for_selector("#progress .progress-case")
    page.locator("#progress .progress-heading").get_by_text(
        "已处理 1 · 待审核 1"
    ).wait_for()
    page.locator("#progress").get_by_text("取消订单 · 已处理").wait_for()
    page.locator("#progress").get_by_text("投诉处理 · 等待审核").wait_for()

    progress = customer_progress(page, app_server, session_id)
    assert progress["status"] == 200
    summary = progress["body"]["summary"]
    assert summary["processed"] == 1
    assert summary["pending_review"] == 1
    assert summary["partial_auto"] is True
    save_screenshot(page, artifacts_dir, "customer-partial-auto-progress")


def test_review_page_shows_shadow_comparison_panel(page, app_server):
    review_login(page, app_server)
    page.wait_for_selector("#shadow:not([hidden])")
    panel = page.text_content("#shadow")
    assert "Shadow 对照" in panel
    assert "高风险 auto" in panel
    assert "影子模式" in panel
