"""P1 frontend accessibility / loading-state checks.

These assertions are intentionally structural (landmarks, live regions, labels,
loading copy) rather than a full WCAG audit.  They protect the parts most likely
to regress when the product UI is edited.
"""

from __future__ import annotations

from conftest import REVIEW_TOKEN


def test_customer_and_review_pages_expose_accessible_landmarks(page, app_server):
    page.goto(f"{app_server.base_url}/customer")
    page.wait_for_selector("#orders article.order")

    assert page.locator("html").get_attribute("lang") == "zh-CN"
    assert page.locator("#chat").get_attribute("aria-live") == "polite"
    assert page.locator("#orders").get_attribute("aria-live") == "polite"
    assert page.locator("#status").get_attribute("role") == "status"
    assert page.locator("label[for='message']").count() == 1
    assert page.locator("#send").get_attribute("aria-label")
    assert page.locator("#resetSession").get_attribute("aria-label")

    action_buttons = page.locator("#orders article.order button")
    assert action_buttons.count() > 0
    for index in range(action_buttons.count()):
        assert action_buttons.nth(index).inner_text().strip()

    page.goto(f"{app_server.base_url}/review")
    assert page.locator("label[for='token']").count() == 1
    assert page.locator("#notice").get_attribute("role") == "status"
    assert page.locator("#conversation").get_attribute("aria-live") == "polite"
    page.fill("#token", REVIEW_TOKEN)
    page.click("#login button")
    page.wait_for_function(
        "() => document.querySelector('#refresh') && !document.querySelector('#refresh').hidden"
    )
    for selector in ("#refresh", "#takeover", "#automatic", "#suggest", "#replySend"):
        assert page.locator(selector).inner_text().strip()


def test_customer_orders_show_loading_copy_before_context_arrives(page, app_server):
    page.add_init_script(
        """
        const originalFetch = window.fetch;
        window.fetch = (url, options) => {
          if (String(url).includes('/customer/context')) {
            return new Promise(resolve => setTimeout(
              () => resolve(originalFetch(url, options)), 1200
            ));
          }
          return originalFetch(url, options);
        };
        """
    )
    page.goto(f"{app_server.base_url}/customer")
    assert "正在读取订单" in (page.text_content("#orders") or "")
    page.wait_for_selector("#orders article.order")
