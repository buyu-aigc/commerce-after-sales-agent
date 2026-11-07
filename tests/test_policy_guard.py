from app.olist.service import InMemoryCaseService, OlistService
from app.policy.engine import PolicyEngine
from app.policy.guard import PolicyGuard


def test_guard_reuses_same_facts_as_engine():
    guard = PolicyGuard(PolicyEngine(), OlistService(), InMemoryCaseService())
    low = guard.check("refund_request", "203096f03d82e0dffbc41ebc2e2bcfb7")
    assert low["decision"] == "review"
    assert low["policy_id"] == "demo_delay_refund_review_v1"


def test_guard_handles_unknown_order_as_review():
    guard = PolicyGuard(PolicyEngine(), OlistService(), InMemoryCaseService())
    result = guard.check("refund_request", "f" * 32)
    assert result["decision"] == "review"
    assert result["reason"] == "order_not_found"


def test_guard_customer_safe_hides_facts():
    guard = PolicyGuard(PolicyEngine(), OlistService(), InMemoryCaseService())
    result = guard.check("refund_request", "203096f03d82e0dffbc41ebc2e2bcfb7")
    safe = guard.customer_safe(result)
    assert "facts" not in safe
    assert "policy_id" in safe
