"""Validate the author/expert-rubric label set used for the portfolio benchmark."""

from evaluation.expert_label_report import DEFAULT_CASES, DEFAULT_LABELS, build, load_jsonl


def test_expert_labels_cover_cases_without_high_risk_auto() -> None:
    labels = load_jsonl(DEFAULT_LABELS)
    cases = load_jsonl(DEFAULT_CASES)
    report = build(labels, cases)

    assert report["total_cases"] == 24
    assert report["total_labels"] == 48
    assert report["cases_with_labels"] == 24
    assert report["route_counts"] == {
        "auto_execute": 23,
        "human_review": 18,
        "clarify": 5,
        "reject": 2,
    }
    assert report["errors"] == []
    assert report["human_reviewed"] is False
