"""Opt-in live-LLM browser smoke.

Run with a real key:
  P0_E2E_LIVE=1 OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... \
    python -m pytest tests/e2e/test_live_llm_smoke.py -q

The normal E2E suite skips this file because it asserts on model behavior and
costs real tokens.
"""

from __future__ import annotations

import os

import pytest
from conftest import ORDER_DELAYED, review_login, save_screenshot

pytestmark = pytest.mark.skipif(
    os.environ.get("P0_E2E_LIVE") != "1",
    reason="live LLM browser smoke is opt-in; set P0_E2E_LIVE=1 and a real key",
)


def test_live_customer_refund_case_review_and_progress(page, app_server, artifacts_dir):
    page.goto(f"{app_server.base_url}/customer")
    page.wait_for_selector("#orders article.order")

    # Exercise the live policy path first (one real model call), then the
    # customer write flow.
    page.fill("#message", "退款一般需要满足什么条件？")
    page.click("#send")
    page.wait_for_function(
        "() => document.querySelectorAll('#chat .message.agent').length >= 2",
        timeout=120_000,
    )
    policy_answer = page.locator("#chat .message.agent").last.inner_text()
    assert "退款" in policy_answer
    for internal in ("order_status", "escalation", "policy_id", "HITL"):
        assert internal not in policy_answer

    page.fill(
        "#message",
        f"订单 {ORDER_DELAYED} 晚到了很多天，商品体验也不好，我想申请退款。",
    )
    page.click("#send")

    page.wait_for_selector("#progress .progress-case", timeout=120_000)
    page.wait_for_function(
        "() => document.querySelector('#progress')?.textContent.includes('退款申请')",
        timeout=120_000,
    )
    page.wait_for_function(
        "() => document.querySelector('#progress')?.textContent.includes('等待审核')",
        timeout=120_000,
    )
    save_screenshot(page, artifacts_dir, "live-customer-pending")

    review_login(page, app_server)
    page.wait_for_selector(".queue-item", timeout=60_000)
    page.click(".queue-item")
    page.wait_for_selector("#detail:not([hidden])", timeout=60_000)
    page.click("#approve")
    page.wait_for_function(
        "() => document.querySelector('#caseStatus')?.textContent.includes('已执行')",
        timeout=120_000,
    )

    page.goto(f"{app_server.base_url}/customer")
    page.evaluate(
        "(value) => sessionStorage.setItem('support-session', value)",
        page.evaluate("sessionStorage.getItem('support-session')"),
    )
    page.reload()
    page.wait_for_selector("#progress .progress-case", timeout=60_000)
    page.wait_for_function(
        "() => document.querySelector('#progress')?.textContent.includes('已处理')",
        timeout=120_000,
    )
    save_screenshot(page, artifacts_dir, "live-customer-executed")
