"""Unit tests for deterministic complexity/risk decomposition."""

from __future__ import annotations

from app.after_sales.complexity import assess_request

A = "a" * 32
B = "b" * 32
C = "c" * 32


def _orders() -> dict[str, dict[str, object]]:
    return {
        A: {"status": "invoiced", "payment_value": 45.0, "delay_days": 0, "review_score": 4},
        B: {"status": "delivered", "payment_value": 500.0, "delay_days": 11, "review_score": 4},
        C: {"status": "delivered", "payment_value": 43.0, "delay_days": 11, "review_score": 4},
    }


def test_complex_request_splits_independent_sub_tasks() -> None:
    assessment = assess_request(
        f"订单 {A} 帮我取消，订单 {B} 晚到了我要退款",
        order_lookup=_orders().get,
    )

    assert assessment.tier == "complex"
    assert assessment.action_types == ["cancel_order", "refund_request"]
    assert assessment.order_ids == [A, B]
    assert assessment.write_units[0].order_id == A
    assert assessment.write_units[0].risk_level == "low"
    assert assessment.write_units[0].auto_eligible is True
    assert assessment.write_units[1].order_id == B
    assert assessment.write_units[1].risk_level == "high"
    assert assessment.write_units[1].auto_eligible is False
    assert assessment.summary()["sub_task_count"] == 2


def test_same_order_multiple_actions_are_not_deduped() -> None:
    assessment = assess_request(
        f"订单 {C} 晚到了我要退款，另外开发票",
        order_lookup=_orders().get,
    )

    assert assessment.tier == "complex"
    assert len(assessment.write_units) == 2
    assert [unit.action_type for unit in assessment.write_units] == [
        "refund_request",
        "invoice_request",
    ]
    assert {unit.order_id for unit in assessment.write_units} == {C}


def test_policy_question_is_not_turned_into_a_write() -> None:
    assessment = assess_request("帮我查退款政策", order_lookup=_orders().get)

    assert assessment.tier == "simple"
    assert assessment.write_units == []
    assert [unit.intent for unit in assessment.units] == ["policy"]


def test_missing_order_evidence_is_high_risk() -> None:
    assessment = assess_request("我要退款、取消订单、开发票，还要投诉", order_lookup=_orders().get)

    assert assessment.tier == "complex"
    assert len(assessment.write_units) == 4
    assert all(unit.risk_level == "high" for unit in assessment.write_units)
    assert all("order_id" in unit.missing_evidence for unit in assessment.write_units)


def test_address_evidence_fragments_are_merged_into_the_action() -> None:
    assessment = assess_request(
        f"订单 {A} 改地址，收件人张三，电话13800138000，"
        "新地址是上海市浦东新区世纪大道100号",
        order_lookup=_orders().get,
    )

    assert assessment.tier == "simple"
    unit = assessment.write_units[0]
    assert unit.action_type == "change_address"
    assert unit.evidence_complete is True
    assert "收件人" in unit.text
    assert "13800138000" in unit.text


def test_hard_risk_language_blocks_auto_eligibility() -> None:
    assessment = assess_request(
        f"订单 {C} 退款失败，我要投诉并起诉",
        order_lookup=_orders().get,
    )

    assert assessment.tier == "complex"
    assert all(unit.risk_level == "high" for unit in assessment.write_units)
    assert all(unit.auto_eligible is False for unit in assessment.write_units)


def test_read_plus_write_is_complex_but_counts_read_tasks() -> None:
    assessment = assess_request(
        f"查订单 {A} 状态，退款政策是什么，订单 {C} 取消",
        order_lookup=_orders().get,
    )

    assert assessment.tier == "complex"
    assert assessment.summary()["read_task_count"] == 2
    assert assessment.summary()["write_task_count"] == 1
    assert assessment.summary()["sub_task_count"] == 3
