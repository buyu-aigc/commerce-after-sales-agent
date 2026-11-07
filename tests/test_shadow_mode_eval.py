"""Regression gate for the shadow-mode replay and its non-execution guarantees."""

from __future__ import annotations

from evaluation.shadow_mode_eval import DEFAULT_CASES, DEFAULT_LABELS, main_async


def test_shadow_replay_compares_labels_without_execution(tmp_path) -> None:
    report = main_async(DEFAULT_CASES, DEFAULT_LABELS, tmp_path / "shadow_report.json")
    summary = report["summary"]

    assert summary["suggestions"] == 48
    assert summary["decided"] == 41
    assert summary["high_risk_auto_suggestions"] == 0
    assert summary["false_auto"] == 0
    assert summary["auto_precision"] is None
    assert summary["business_approval_rate"] == 1.0
    assert summary["review_approved"] == 5
    assert summary["missed_auto"] == 0
    assert summary["review_rejected"] == 13
    assert report["execution_check"] == {
        "business_cases_created": 0,
        "order_events_created": 0,
    }
    assert report["gates"]["labels_synthetic"] is True
    assert report["gates"]["recommendation"] == "keep_shadow"
    assert len(report["cases"]) == 24
