from collections import Counter
from pathlib import Path

from app.policy.engine import PolicyEngine
from evaluation.policy_real_order_bench import load_fixtures


def test_combined_fixture_has_600_unique_orders_and_hard_strata():
    rows = load_fixtures(Path("evaluation/policy_combined_fixtures.jsonl"))
    assert len(rows) == 600
    assert len({row["order_id"] for row in rows}) == 600
    groups = Counter(row["fixture_group"] for row in rows)
    assert groups == {
        "control_auto": 50,
        "repeat_claim": 50,
        "high_risk_user": 50,
        "refund_failure": 50,
        "high_delay": 200,
        "no_delay": 100,
        "invoiced": 50,
        "canceled": 50,
    }
    assert sum(bool(row.get("repeat_claim")) for row in rows) == 50
    assert sum(row.get("user_risk_level") == "high" for row in rows) == 50
    assert sum(bool(row.get("refund_failure")) for row in rows) == 50


def test_combined_policy_auto_only_control_stratum():
    rows = load_fixtures(Path("evaluation/policy_combined_fixtures.jsonl"))
    engine = PolicyEngine()
    auto = Counter()
    for row in rows:
        decision = engine.evaluate(
            {
                "action_type": "refund_request",
                "order_status": row["status"],
                "delay_days": row.get("delay_days") or 0,
                "amount": row.get("payment_value") or 0,
                "prior_compensation": False,
                "repeat_claim": bool(row.get("repeat_claim")),
                "risk_level": row.get("user_risk_level") or "",
                "refund_failure": bool(row.get("refund_failure")),
            }
        )
        if decision["decision"] == "auto":
            auto[row["fixture_group"]] += 1
    assert dict(auto) == {"control_auto": 50}
