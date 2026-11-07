"""Regression gate for the 24-case complex partial-auto evaluation set."""

from __future__ import annotations

import asyncio

from evaluation.complex_partial_auto_eval import DEFAULT_CASES, load_cases, main_async


def test_complex_partial_auto_eval_meets_acceptance(tmp_path) -> None:
    cases = load_cases(DEFAULT_CASES)
    assert len(cases) >= 20
    assert len({case["id"] for case in cases}) == len(cases)

    report = asyncio.run(main_async(DEFAULT_CASES, tmp_path / "complex_report.json"))
    summary = report["summary"]

    assert summary["total_cases"] == len(cases)
    assert summary["complex_decomposed"] == len(cases)
    assert summary["high_risk_auto"] == 0
    assert summary["auto_without_policy_auto"] == 0
    assert summary["partial_auto_rate"] >= 0.8
    assert summary["low_risk_auto_rate"] >= 0.9
    assert summary["sub_task_policy_coverage"] >= 0.9
    assert summary["failing_cases"] == 0
    assert summary["appeals_or_rework"] == 0
