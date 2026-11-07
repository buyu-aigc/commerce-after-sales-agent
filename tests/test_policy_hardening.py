"""P0-2 hardening tests: schema, effective windows, guardrails, conflicts."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from app.policy.engine import PolicyEngine
from app.policy.schema import (
    PolicyPackError,
    canonical_pack_hash,
    detect_conflicts,
    validate_pack,
)

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "policies" / "demo_after_sales.json"
ARCHIVE = ROOT / "policies" / "archive" / "demo_after_sales_v1.0.0.json"


def _load(path: Path = PACK) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(pack: dict) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "pack.json"
    tmp.write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")
    return tmp


def _low_delay_facts(**overrides):
    facts = {
        "action_type": "refund_request",
        "order_status": "delivered",
        "delay_days": 11,
        "amount": 43.0,
        "prior_compensation": False,
        "repeat_claim": False,
        "risk_level": "medium",
        "refund_failure": False,
    }
    facts.update(overrides)
    return facts


def test_current_pack_is_valid_conflict_free_and_versioned():
    pack = _load()
    report = validate_pack(pack)
    assert report["ok"], report["errors"]
    assert report["version"] == "1.1.0"
    assert report["warnings"] == []
    assert report["pack_hash"] == canonical_pack_hash(pack)
    conflicts = detect_conflicts(pack)
    assert conflicts["errors"] == []
    assert conflicts["warnings"] == []


def test_archived_pack_remains_loadable_and_deterministic():
    engine = PolicyEngine(ARCHIVE)
    assert engine.version == "1.0.0"
    assert engine.validation["ok"]
    decision = engine.evaluate(_low_delay_facts())
    assert decision["decision"] == "auto"
    assert decision["policy_id"] == "demo_delay_refund_auto_v1"


def test_schema_rejects_unknown_condition_and_missing_cap():
    pack = _load()
    pack["policies"][0]["conditions"]["unknown_fact"] = True
    report = validate_pack(pack)
    assert not report["ok"]
    assert {error["code"] for error in report["errors"]} >= {"schema_error"}

    pack = _load()
    auto = next(item for item in pack["policies"] if item["decision"] == "auto")
    auto["max_amount"] = 0
    report = validate_pack(pack)
    assert "auto_policy_missing_amount_cap" in {error["code"] for error in report["errors"]}


def test_schema_rejects_unsafe_empty_default_and_duplicate_ids():
    pack = _load()
    default = next(item for item in pack["policies"] if not item["conditions"])
    default["decision"] = "auto"
    codes = {error["code"] for error in validate_pack(pack)["errors"]}
    assert "unsafe_default_policy" in codes

    pack = _load()
    pack["policies"][1]["policy_id"] = pack["policies"][0]["policy_id"]
    codes = {error["code"] for error in validate_pack(pack)["errors"]}
    assert "duplicate_policy_id" in codes


def test_engine_refuses_invalid_pack():
    pack = _load()
    pack["policies"][0]["decision"] = "sometimes"
    with pytest.raises(PolicyPackError):
        PolicyEngine(_write(pack))


def test_effective_window_is_right_open_and_inherited_from_pack():
    pack = _load()
    pack["effective_from"] = "2030-01-01T00:00:00Z"
    pack["effective_to"] = "2030-02-01T00:00:00Z"
    engine = PolicyEngine(_write(pack))
    before = engine.evaluate(_low_delay_facts(), as_of="2029-12-31T23:59:59Z")
    inside = engine.evaluate(_low_delay_facts(), as_of="2030-01-15T00:00:00Z")
    boundary = engine.evaluate(_low_delay_facts(), as_of="2030-02-01T00:00:00Z")
    assert before["decision"] == "review"
    assert before["policy_id"] == "demo_default_review_v1"
    assert inside["decision"] == "auto"
    assert boundary["decision"] == "review"


def test_per_policy_window_overrides_pack_window():
    pack = _load()
    auto = next(item for item in pack["policies"] if item["policy_id"] == "demo_delay_refund_auto_v1")
    auto["effective_to"] = "2030-01-01T00:00:00Z"
    engine = PolicyEngine(_write(pack))
    expired = engine.evaluate(_low_delay_facts(), as_of="2030-06-01T00:00:00Z")
    assert expired["decision"] == "review"
    assert expired["policy_id"] == "demo_delay_refund_review_v1"


def test_engine_downgrades_auto_when_amount_exceeds_cap():
    pack = _load()
    auto = next(item for item in pack["policies"] if item["policy_id"] == "demo_delay_refund_auto_v1")
    auto["max_amount"] = 50
    auto["conditions"]["amount_max"] = 100
    engine = PolicyEngine(_write(pack))
    decision = engine.evaluate(_low_delay_facts(amount=80))
    assert decision["decision"] == "review"
    assert decision["guardrail"] == ["amount_cap_exceeded"]
    assert "amount_cap_exceeded" in decision["reason"]


@pytest.mark.parametrize(
    "override",
    [
        {"repeat_claim": True},
        {"refund_failure": True},
        {"prior_compensation": True},
        {"risk_level": "high"},
    ],
)
def test_engine_hard_blocks_risky_auto_even_if_pack_matches(override):
    """Defense in depth: a bad pack cannot auto-approve a hard-risk fact."""
    pack = _load()
    auto = next(item for item in pack["policies"] if item["policy_id"] == "demo_delay_refund_auto_v1")
    auto["priority"] = 1
    auto["conditions"] = {"amount_max": 100}
    engine = PolicyEngine(_write(pack))
    decision = engine.evaluate(_low_delay_facts(**override))
    assert decision["decision"] == "review"
    assert decision["guardrail"]


def test_detects_same_priority_conflict_and_duplicate_rule():
    pack = _load()
    auto = next(item for item in pack["policies"] if item["policy_id"] == "demo_delay_refund_auto_v1")
    auto["priority"] = 30
    codes = {error["code"] for error in detect_conflicts(pack)["errors"]}
    assert "same_priority_conflict" in codes

    pack = _load()
    pack["policies"].append(copy.deepcopy(pack["policies"][0]))
    codes = {error["code"] for error in detect_conflicts(pack)["errors"]}
    assert "duplicate_rule" in codes


def test_detects_auto_policy_shadowing_a_narrower_review_rule():
    pack = _load()
    pack["policies"].extend(
        [
            {
                "policy_id": "wide_auto_v1",
                "priority": 1,
                "action_types": ["refund_request"],
                "conditions": {"amount_max": 500, "repeat_claim": False},
                "decision": "auto",
                "max_amount": 500,
                "reason": "drill wide auto",
            },
            {
                "policy_id": "narrow_review_v1",
                "priority": 50,
                "action_types": ["refund_request"],
                "conditions": {"amount_max": 100, "repeat_claim": False},
                "decision": "review",
                "max_amount": 0,
                "reason": "drill narrow review",
            },
        ]
    )
    report = detect_conflicts(pack)
    codes = {warning["code"] for warning in report["warnings"]}
    assert "auto_overrides_review" in codes
