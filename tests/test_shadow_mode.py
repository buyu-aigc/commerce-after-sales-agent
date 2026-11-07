"""Shadow-mode unit tests: suggestions must never execute and metrics must be truthful."""

from __future__ import annotations

from pathlib import Path

from app.after_sales.complexity import assess_request
from app.after_sales.shadow import (
    SUGGEST_AUTO,
    SUGGEST_REVIEW,
    ShadowService,
    SQLiteShadowStore,
)
from app.olist.service import OrderStatusView, SQLiteCaseService
from app.policy.engine import PolicyEngine
from app.policy.guard import PolicyGuard

A = "a" * 32
B = "b" * 32


def _view(order_id: str, *, status: str, amount: float, delay: int = 0, review: int = 4):
    return OrderStatusView(
        order_id=order_id,
        status=status,
        customer_state="SP",
        purchase_timestamp="2017-09-18 14:31:30",
        estimated_delivery_date="2017-09-28 00:00:00",
        delivered_customer_date="2017-10-09 22:23:46" if status == "delivered" else "",
        delay_days=delay,
        payment_value=amount,
        review_score=review,
        category_summary="health_beauty",
    )


def _runtime(tmp_path: Path, *, enabled: bool = True):
    orders = {
        A: _view(A, status="invoiced", amount=45.0),
        B: _view(B, status="delivered", amount=500.0, delay=11),
    }
    case_service = SQLiteCaseService(tmp_path / "cases.db")
    olist = type(
        "Orders", (), {"get_order_status": lambda self, oid: orders.get(oid)}
    )()
    guard = PolicyGuard(PolicyEngine(), olist, case_service)
    shadow = ShadowService(
        SQLiteShadowStore(tmp_path / "shadow.db"),
        enabled=enabled,
        sample_rate=1.0,
    )
    analysis = assess_request(
        f"订单 {A} 帮我取消，订单 {B} 晚到了我要退款",
        order_lookup=orders.get,
    )
    return shadow, guard, orders, analysis


def test_shadow_suggestions_never_execute_and_high_risk_auto_is_zero(tmp_path: Path) -> None:
    shadow, guard, orders, analysis = _runtime(tmp_path)

    suggestions = shadow.record_analysis(
        request_id="req-shadow-1",
        session_id="session-shadow-1",
        analysis=analysis,
        order_lookup=orders.get,
        policy_guard=guard,
    )

    by_action = {item["action_type"]: item for item in suggestions}
    assert by_action["cancel_order"]["suggested_outcome"] == SUGGEST_AUTO
    assert by_action["refund_request"]["suggested_outcome"] == SUGGEST_REVIEW
    assert by_action["refund_request"]["risk_level"] == "high"

    metrics = shadow.metrics()
    assert metrics["summary"]["suggestions"] == 2
    assert metrics["summary"]["high_risk_auto_suggestions"] == 0
    assert metrics["gates"]["high_risk_auto_zero"] is True
    # Shadow mode is read-only: no order projection or order event was written.
    assert metrics["summary"]["linked_cases"] == 0


def test_shadow_decision_metrics_and_idempotency(tmp_path: Path) -> None:
    shadow, guard, orders, analysis = _runtime(tmp_path)
    suggestions = shadow.record_analysis(
        request_id="req-shadow-2",
        session_id="session-shadow-2",
        analysis=analysis,
        order_lookup=orders.get,
        policy_guard=guard,
    )
    by_action = {item["action_type"]: item for item in suggestions}
    shadow.link_case(
        case_id="case-cancel",
        request_id="req-shadow-2",
        sub_task_index=by_action["cancel_order"]["sub_task_index"],
        action_type="cancel_order",
    )
    shadow.link_case(
        case_id="case-refund",
        request_id="req-shadow-2",
        sub_task_index=by_action["refund_request"]["sub_task_index"],
        action_type="refund_request",
    )
    assert shadow.record_human_decision(
        case_id="case-cancel", decision="approve", reviewer_id="reviewer-1"
    ) == 1
    assert shadow.record_human_decision(
        case_id="case-refund", decision="reject", reviewer_id="reviewer-1", reason="high value"
    ) == 1
    # A second decision must not overwrite the first label.
    assert shadow.record_human_decision(
        case_id="case-cancel", decision="reject", reviewer_id="reviewer-2"
    ) == 0

    metrics = shadow.metrics()
    summary = metrics["summary"]
    assert summary["decided"] == 2
    assert summary["auto_decided"] == 1
    assert summary["auto_approved"] == 1
    assert summary["false_auto"] == 0
    assert summary["review_rejected"] == 1
    assert summary["routing_agreement"] == 1.0
    assert metrics["gates"]["false_auto_zero"] is True


def test_shadow_metrics_surface_false_auto(tmp_path: Path) -> None:
    shadow, guard, orders, analysis = _runtime(tmp_path)
    suggestions = shadow.record_analysis(
        request_id="req-shadow-3",
        session_id="session-shadow-3",
        analysis=analysis,
        order_lookup=orders.get,
        policy_guard=guard,
    )
    auto = next(item for item in suggestions if item["suggested_outcome"] == SUGGEST_AUTO)
    shadow.link_case(
        case_id="case-auto-rejected",
        request_id="req-shadow-3",
        sub_task_index=auto["sub_task_index"],
        action_type=auto["action_type"],
    )
    shadow.record_human_decision(
        case_id="case-auto-rejected", decision="reject", reviewer_id="reviewer-3"
    )

    metrics = shadow.metrics()
    assert metrics["summary"]["false_auto"] == 1
    assert metrics["summary"]["false_auto_rate"] == 1.0
    assert metrics["gates"]["false_auto_zero"] is False
    assert metrics["gates"]["recommendation"] == "keep_shadow"
    assert metrics["by_risk"]


def test_shadow_sampling_is_deterministic_and_can_be_disabled(tmp_path: Path) -> None:
    disabled = ShadowService(SQLiteShadowStore(tmp_path / "off.db"), enabled=False)
    assert disabled.should_shadow("session") is False

    enabled = ShadowService(
        SQLiteShadowStore(tmp_path / "on.db"), enabled=True, sample_rate=1.0
    )
    assert enabled.should_shadow("session") is True
    assert enabled.should_shadow("session") is True

    sampled = ShadowService(
        SQLiteShadowStore(tmp_path / "sampled.db"), enabled=True, sample_rate=0.5
    )
    first = sampled.should_shadow("stable-session", "olist-demo")
    assert first == sampled.should_shadow("stable-session", "olist-demo")


def test_shadow_recent_records_mask_order_ids(tmp_path: Path) -> None:
    shadow, guard, orders, analysis = _runtime(tmp_path)
    shadow.record_analysis(
        request_id="req-shadow-4",
        session_id="session-shadow-4",
        analysis=analysis,
        order_lookup=orders.get,
        policy_guard=guard,
    )
    records = shadow.recent(limit=10)
    assert records
    assert all(len(item["order_id"]) < 32 for item in records)
    assert all(item["observation_id"].startswith("shadow-") for item in records)
    assert "..." in records[0]["order_id"]
