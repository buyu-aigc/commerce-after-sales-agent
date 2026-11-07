from pathlib import Path

from evaluation.policy_workflow_bench import run


def test_policy_workflow_routing_reduces_review_without_false_auto():
    report = run(Path("evaluation/policy_golden_cases.jsonl"))
    assert report["total"] == 20
    assert report["after_policy_check"]["auto"] == 6
    assert report["after_policy_check"]["review"] == 14
    assert report["false_auto"] == 0
    assert report["review_reduction_pp"] == 30.0
