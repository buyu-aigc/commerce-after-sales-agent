#!/usr/bin/env python3
"""Auto-approval spot check and rollback drill for the policy pack.

Two operator questions are answered here:

1. **Spot check** – for every frozen fixture the engine auto-approves, verify
   the amount ceiling and hard-risk guardrails actually held.
2. **Rollback drill** – can we switch back to the archived pack, get stable
   decisions, and switch forward again without drift?  Can the validator reject
   a deliberately unsafe pack and a priority collision?

The drill is deterministic and offline; it never calls a live service.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.policy.engine import PolicyEngine
from app.policy.schema import PolicyPackError, detect_conflicts, validate_pack
from evaluation.policy_versioned_replay import (
    DEFAULT_ARCHIVE,
    DEFAULT_AS_OF,
    DEFAULT_CURRENT,
    DEFAULT_FIXTURES,
    fixture_facts,
    load_fixtures,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "policy_hardening_drill_report.json"


def _decision_key(decision: dict[str, Any]) -> tuple[str, str]:
    return str(decision["decision"]), str(decision["policy_id"])


def _decisions(engine: PolicyEngine, rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    return {str(row["order_id"]): _decision_key(engine.evaluate(fixture_facts(row))) for row in rows}


def spot_check(engine: PolicyEngine, rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    failures: list[dict[str, Any]] = []
    auto_by_policy: Counter[str] = Counter()
    for row in rows:
        facts = fixture_facts(row)
        decision = engine.evaluate(facts)
        if decision["decision"] != "auto":
            continue
        checked += 1
        auto_by_policy[str(decision["policy_id"])] += 1
        reasons: list[str] = []
        amount = float(facts.get("amount") or 0)
        ceiling = float(decision.get("max_amount") or 0)
        if ceiling <= 0 or amount > ceiling:
            reasons.append("amount_cap_not_enforced")
        for key in ("repeat_claim", "refund_failure", "prior_compensation"):
            if bool(facts.get(key)):
                reasons.append(f"{key}_must_review")
        if str(facts.get("risk_level") or "") == "high":
            reasons.append("high_risk_must_review")
        if reasons:
            failures.append(
                {
                    "order_id": row["order_id"],
                    "policy_id": decision["policy_id"],
                    "reasons": reasons,
                    "amount": amount,
                    "max_amount": ceiling,
                    "fixture_group": row.get("fixture_group"),
                }
            )
    return {
        "checked": checked,
        "failures": failures,
        "auto_by_policy": dict(sorted(auto_by_policy.items())),
        "ok": not failures,
    }


def _unsafe_pack_drill(current: dict[str, Any]) -> dict[str, Any]:
    bad = copy.deepcopy(current)
    bad["version"] = "9.9.9"
    bad["policies"].insert(
        0,
        {
            "policy_id": "bad_auto_all_v1",
            "priority": 1,
            "action_types": ["refund_request"],
            "conditions": {"amount_max": 9999},
            "decision": "auto",
            "max_amount": 999999,
            "reason": "deliberately unsafe drill policy",
        },
    )
    report = validate_pack(bad)
    rejected = False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_pack.json"
        path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        try:
            PolicyEngine(path)
        except PolicyPackError:
            rejected = True
    return {
        "ok": not report["ok"] and rejected,
        "error_codes": sorted({error["code"] for error in report["errors"]}),
        "engine_rejected": rejected,
    }


def _conflict_pack_drill(current: dict[str, Any]) -> dict[str, Any]:
    conflicting = copy.deepcopy(current)
    for policy in conflicting["policies"]:
        if policy["policy_id"] == "demo_delay_refund_auto_v1":
            policy["priority"] = 30  # same priority as demo_delay_refund_review_v1
    report = detect_conflicts(conflicting)
    codes = sorted({error["code"] for error in report["errors"]})
    return {
        "ok": "same_priority_conflict" in codes,
        "error_codes": codes,
    }


def rollback_drill(
    *,
    pack_path: Path = DEFAULT_CURRENT,
    archive_path: Path = DEFAULT_ARCHIVE,
    fixtures_path: Path = DEFAULT_FIXTURES,
    as_of: str = DEFAULT_AS_OF,
) -> dict[str, Any]:
    rows = load_fixtures(fixtures_path)
    current = PolicyEngine(pack_path, as_of=as_of)
    archived = PolicyEngine(archive_path, as_of=as_of)
    current_decisions = _decisions(current, rows)
    archived_decisions = _decisions(archived, rows)

    with tempfile.TemporaryDirectory() as tmp:
        rollback_path = Path(tmp) / "demo_after_sales.json"
        shutil.copyfile(archive_path, rollback_path)
        rollback_engine = PolicyEngine(rollback_path, as_of=as_of)
        rollback_decisions = _decisions(rollback_engine, rows)
        rollback_hash = rollback_engine.pack_hash
        restored = PolicyEngine(pack_path, as_of=as_of)
        restored_decisions = _decisions(restored, rows)
        restored_hash = restored.pack_hash

    diff_rollback = [
        order_id
        for order_id in current_decisions
        if current_decisions[order_id] != rollback_decisions.get(order_id)
    ]
    diff_restore = [
        order_id
        for order_id in current_decisions
        if current_decisions[order_id] != restored_decisions.get(order_id)
    ]
    unsafe = _unsafe_pack_drill(current.pack)
    conflict = _conflict_pack_drill(current.pack)
    spot = spot_check(current, rows)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of,
        "fixtures": str(fixtures_path),
        "fixtures_count": len(rows),
        "spot_check": spot,
        "rollback": {
            "current_version": current.version,
            "rollback_version": archived.version,
            "current_hash": current.pack_hash,
            "rollback_hash": rollback_hash,
            "archive_hash": archived.pack_hash,
            "hash_matches_archive": rollback_hash == archived.pack_hash,
            "rollback_decisions_equal_archive": archived_decisions == rollback_decisions,
            "changed_on_rollback": len(diff_rollback),
            "changed_on_rollback_samples": diff_rollback[:10],
            "restored_hash": restored_hash,
            "restored_equal_current": restored_hash == current.pack_hash,
            "changed_on_restore": len(diff_restore),
        },
        "negative_drills": {
            "unsafe_pack": unsafe,
            "priority_conflict": conflict,
        },
        "current_validation": current.validation,
        "ok": bool(
            spot["ok"]
            and rollback_hash == archived.pack_hash
            and archived_decisions == rollback_decisions
            and restored_hash == current.pack_hash
            and not diff_restore
            and unsafe["ok"]
            and conflict["ok"]
            and current.validation["ok"]
            and not current.conflicts["errors"]
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    spot = report["spot_check"]
    rollback = report["rollback"]
    unsafe = report["negative_drills"]["unsafe_pack"]
    conflict = report["negative_drills"]["priority_conflict"]
    lines = [
        "# Policy Hardening Drill",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- as_of: {report['as_of']}",
        f"- fixtures: {report['fixtures_count']}",
        f"- result: **{'PASS' if report['ok'] else 'FAIL'}**",
        "",
        "## Auto-approval spot check",
        "",
        f"- auto decisions checked: {spot['checked']}",
        f"- failures: {len(spot['failures'])}",
        "",
        "| policy | auto count |",
        "|---|---:|",
    ]
    for policy_id, count in spot["auto_by_policy"].items():
        lines.append(f"| {policy_id} | {count} |")
    lines.extend(
        [
            "",
            "## Rollback drill",
            "",
            f"- current: {rollback['current_version']} (`{rollback['current_hash'][:12]}…`)",
            f"- rollback target: {rollback['rollback_version']} (`{rollback['archive_hash'][:12]}…`)",
            f"- rollback hash matches archive: {rollback['hash_matches_archive']}",
            f"- changed decisions on rollback: {rollback['changed_on_rollback']}",
            f"- restored hash equals current: {rollback['restored_equal_current']}",
            f"- changed decisions after restore: {rollback['changed_on_restore']}",
            "",
            "## Negative drills",
            "",
            "- unsafe pack rejected: "
            f"{unsafe['engine_rejected']} (codes: {', '.join(unsafe['error_codes'])})",
            f"- priority collision detected: {conflict['ok']} (codes: {', '.join(conflict['error_codes'])})",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = rollback_drill(
        pack_path=args.pack,
        archive_path=args.archive,
        fixtures_path=args.fixtures,
        as_of=args.as_of,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "spot_checked": report["spot_check"]["checked"],
                "spot_failures": len(report["spot_check"]["failures"]),
                "rollback_changed": report["rollback"]["changed_on_rollback"],
                "unsafe_rejected": report["negative_drills"]["unsafe_pack"]["engine_rejected"],
                "conflict_detected": report["negative_drills"]["priority_conflict"]["ok"],
            },
            ensure_ascii=False,
        )
    )
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
