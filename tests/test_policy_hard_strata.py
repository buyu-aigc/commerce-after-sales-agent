from evaluation.policy_hard_strata_bench import run


def test_hard_strata_never_auto_approve():
    report = run()
    assert report["total"] == 400
    assert report["false_auto"] == 0
    assert report["by_stratum"]["control"]["auto"] == 100
    assert report["by_stratum"]["repeat_claim"]["auto"] == 0
    assert report["by_stratum"]["high_risk_user"]["auto"] == 0
    assert report["by_stratum"]["refund_failure"]["auto"] == 0
