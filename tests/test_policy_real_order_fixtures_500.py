from collections import Counter
from pathlib import Path

from app.policy.engine import PolicyEngine
from evaluation.policy_real_order_bench import load_fixtures


def test_500_real_order_fixture_groups_and_policy_auto():
    rows = load_fixtures(Path("evaluation/policy_real_order_fixtures_500.jsonl"))
    assert len(rows) == 500
    assert len({row["order_id"] for row in rows}) == 500
    groups = Counter(row["fixture_group"] for row in rows)
    assert groups == {
        "auto_low_delay": 150,
        "high_delay": 150,
        "no_delay": 100,
        "invoiced": 50,
        "canceled": 50,
    }
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
    assert auto == 150
