import json
from pathlib import Path

from app.policy.engine import PolicyEngine


def test_policy_golden_cases():
    path = Path("evaluation/policy_golden_cases.jsonl")
    engine = PolicyEngine()
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        actual = engine.evaluate(case["facts"])
        mismatch = (
            actual["decision"] != case["expected_decision"]
            or actual["policy_id"] != case["expected_policy_id"]
        )
        if mismatch:
            failures.append((case["id"], actual["decision"], actual["policy_id"]))
    assert not failures, failures
