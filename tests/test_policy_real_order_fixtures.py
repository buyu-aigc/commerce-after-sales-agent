from collections import Counter

from app.policy.engine import PolicyEngine
from evaluation.policy_real_order_bench import load_fixtures


def test_real_order_fixture_has_100_unique_orders_and_balanced_groups():
    rows = load_fixtures(__import__("pathlib").Path("evaluation/policy_real_order_fixtures_100.jsonl"))
    assert len(rows) == 100
    assert len({row["order_id"] for row in rows}) == 100
    groups = Counter(row["fixture_group"] for row in rows)
    assert groups == {
        "auto_low_delay": 30,
        "high_delay": 30,
        "no_delay": 20,
        "invoiced": 10,
        "canceled": 10,
    }


def test_real_order_fixture_policy_auto_count_is_30():
    rows = load_fixtures(__import__("pathlib").Path("evaluation/policy_real_order_fixtures_100.jsonl"))
    engine = PolicyEngine()
    auto = 0
    for row in rows:
        decision = engine.evaluate(
            {
                "action_type": "refund_request",
                "order_status": row["status"],
                "delay_days": row.get("delay_days") or 0,
                "amount": row.get("payment_value") or 0,
                "prior_compensation": False,
                "repeat_claim": False,
                "risk_level": "medium",
            }
        )
        auto += decision["decision"] == "auto"
    assert auto == 30
