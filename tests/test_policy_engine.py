from app.policy.engine import PolicyEngine


def test_low_value_delay_refund_is_auto():
    engine = PolicyEngine()
    decision = engine.evaluate({
        "action_type": "refund_request",
        "order_status": "delivered",
        "delay_days": 11,
        "amount": 43.0,
        "prior_compensation": False,
        "repeat_claim": False,
        "risk_level": "medium",
    })
    assert decision["decision"] == "auto"
    assert decision["policy_id"] == "demo_delay_refund_auto_v1"
    assert decision["demo_only"] is True


def test_high_value_delay_refund_goes_to_review():
    engine = PolicyEngine()
    decision = engine.evaluate({
        "action_type": "refund_request",
        "order_status": "delivered",
        "delay_days": 11,
        "amount": 118.86,
        "prior_compensation": False,
        "repeat_claim": False,
        "risk_level": "medium",
    })
    assert decision["decision"] == "review"
    assert decision["policy_id"] == "demo_delay_refund_review_v1"


def test_prior_compensation_blocks_auto_path():
    engine = PolicyEngine()
    decision = engine.evaluate({
        "action_type": "refund_request",
        "order_status": "delivered",
        "delay_days": 11,
        "amount": 43.0,
        "prior_compensation": True,
        "repeat_claim": False,
        "risk_level": "medium",
    })
    assert decision["decision"] == "review"


def test_preshipment_cancel_is_auto():
    engine = PolicyEngine()
    decision = engine.evaluate({
        "action_type": "cancel_order",
        "order_status": "invoiced",
        "amount": 32.01,
        "prior_compensation": False,
    })
    assert decision["decision"] == "auto"
    assert decision["policy_id"] == "demo_preship_cancel_auto_v1"
