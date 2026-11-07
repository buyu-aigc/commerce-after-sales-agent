#!/usr/bin/env python3
"""Versioned replay of frozen after-sales fixtures against policy packs.

This is the rollout gate for a policy change: every candidate pack is replayed
against the same 600 frozen order rows, with a fixed ``as_of``, and compared
with the archived version.  A candidate that auto-approves a non-control
stratum, or loses expected auto approvals, fails the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.policy.engine import PolicyEngine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = ROOT / "policies" / "demo_after_sales.json"
DEFAULT_ARCHIVE = ROOT / "policies" / "archive" / "demo_after_sales_v1.0.0.json"
DEFAULT_FIXTURES = ROOT / "evaluation" / "policy_combined_fixtures.jsonl"
DEFAULT_GOLDEN = ROOT / "evaluation" / "policy_golden_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation" / "policy_versioned_replay_report.json"
DEFAULT_AS_OF = "2026-10-19T00:00:00Z"
EXPECTED_AUTO_GROUPS = frozenset({"control_auto", "auto_low_delay"})


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_golden(path: Path) -> list[dict[str, Any]]:
    return load_fixtures(path)


def fixture_facts(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": "refund_request",
        "order_status": row["status"],
        "delay_days": row.get("delay_days") or 0,
        "amount": row.get("payment_value") or 0,
        "prior_compensation": False,
        "repeat_claim": bool(row.get("repeat_claim")),
        "risk_level": row.get("user_risk_level") or "",
        "refund_failure": bool(row.get("refund_failure")),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_pack(
    pack_path: Path,
    rows: list[dict[str, Any]],
    *,
    as_of: str,
    golden: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    engine = PolicyEngine(pack_path, as_of=as_of)
    by_group: dict[str, Counter[str]] = {}
    policy_hits: Counter[str] = Counter()
    rows_out: list[dict[str, Any]] = []
    false_auto: list[str] = []
    missed_auto: list[str] = []
    auto = review = 0
    for row in rows:
        decision = engine.evaluate(fixture_facts(row))
        group = str(row.get("fixture_group") or "unknown")
        by_group.setdefault(group, Counter())
        by_group[group]["total"] += 1
        policy_hits[str(decision["policy_id"])] += 1
        is_auto = decision["decision"] == "auto"
        if is_auto:
            auto += 1
            by_group[group]["auto"] += 1
            if group not in EXPECTED_AUTO_GROUPS:
                false_auto.append(str(row["order_id"]))
        else:
            review += 1
            by_group[group]["review"] += 1
            if group in EXPECTED_AUTO_GROUPS:
                missed_auto.append(str(row["order_id"]))
        rows_out.append(
            {
                "order_id": row["order_id"],
                "group": group,
                "decision": decision["decision"],
                "policy_id": decision["policy_id"],
                "amount": float(row.get("payment_value") or 0),
            }
        )
    golden_mismatches: list[dict[str, Any]] = []
    for case in golden or []:
        actual = engine.evaluate(dict(case["facts"]))
        if (
            actual["decision"] != case["expected_decision"]
            or actual["policy_id"] != case["expected_policy_id"]
        ):
            golden_mismatches.append(
                {
                    "id": case.get("id"),
                    "expected": [case["expected_decision"], case["expected_policy_id"]],
                    "actual": [actual["decision"], actual["policy_id"]],
                }
            )
    return {
        "version": engine.version,
        "pack": str(pack_path),
        "pack_hash": engine.pack_hash,
        "as_of": as_of,
        "total": len(rows),
        "auto": auto,
        "review": review,
        "auto_rate": round(auto / len(rows), 4) if rows else 0,
        "review_rate": round(review / len(rows), 4) if rows else 0,
        "false_auto": len(false_auto),
        "false_auto_orders": false_auto[:20],
        "missed_auto": len(missed_auto),
        "missed_auto_orders": missed_auto[:20],
        "by_group": {
            group: {
                "total": counts["total"],
                "auto": counts["auto"],
                "review": counts["review"],
            }
            for group, counts in sorted(by_group.items())
        },
        "policy_hits": dict(sorted(policy_hits.items())),
        "golden_mismatches": golden_mismatches,
        "validation_ok": engine.validation["ok"],
        "conflict_errors": engine.conflicts["errors"],
        "conflict_warnings": engine.conflicts["warnings"],
        "rows": rows_out,
    }


def compare_versions(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    before = {row["order_id"]: row for row in baseline["rows"]}
    newly_auto: list[dict[str, Any]] = []
    auto_removed: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for row in candidate["rows"]:
        previous = before.get(row["order_id"])
        if previous is None:
            continue
        if previous["decision"] != row["decision"]:
            changed.append(
                {
                    "order_id": row["order_id"],
                    "group": row["group"],
                    "from": previous["decision"],
                    "to": row["decision"],
                }
            )
            if row["decision"] == "auto":
                newly_auto.append({"order_id": row["order_id"], "group": row["group"]})
            else:
                auto_removed.append({"order_id": row["order_id"], "group": row["group"]})
    regressions = [item for item in newly_auto if item["group"] not in EXPECTED_AUTO_GROUPS]
    improvements = [item for item in newly_auto if item["group"] in EXPECTED_AUTO_GROUPS]
    return {
        "baseline_version": baseline["version"],
        "candidate_version": candidate["version"],
        "changed_decisions": len(changed),
        "changed_samples": changed[:20],
        "newly_auto": len(newly_auto),
        "newly_auto_samples": newly_auto[:20],
        "auto_removed": len(auto_removed),
        "auto_removed_samples": auto_removed[:20],
        "regressions": regressions,
        "improvements": improvements,
    }


def run(
    *,
    packs: Iterable[Path],
    fixtures_path: Path = DEFAULT_FIXTURES,
    golden_path: Path = DEFAULT_GOLDEN,
    as_of: str = DEFAULT_AS_OF,
) -> dict[str, Any]:
    rows = load_fixtures(fixtures_path)
    golden = load_golden(golden_path)
    versions = [replay_pack(pack, rows, as_of=as_of, golden=golden) for pack in packs]
    baseline, candidate = versions[0], versions[-1]
    comparison = compare_versions(baseline, candidate)
    ok = all(
        version["validation_ok"]
        and version["false_auto"] == 0
        and version["missed_auto"] == 0
        and not version["golden_mismatches"]
        for version in versions
    ) and not comparison["regressions"]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of,
        "fixtures": str(fixtures_path),
        "fixtures_count": len(rows),
        "fixtures_sha256": _sha256(fixtures_path),
        "golden": str(golden_path),
        "golden_count": len(golden),
        "expected_auto_groups": sorted(EXPECTED_AUTO_GROUPS),
        "versions": versions,
        "comparison": comparison,
        "ok": ok,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Policy Versioned Replay",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- as_of: {report['as_of']} (fixed for reproducibility)",
        f"- fixtures: {report['fixtures_count']} rows, sha256 `{report['fixtures_sha256'][:16]}…`",
        f"- golden cases: {report['golden_count']}",
        f"- result: **{'PASS' if report['ok'] else 'FAIL'}**",
        "",
        "| version | pack hash | auto | review | false auto | missed auto | golden mismatch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for version in report["versions"]:
        lines.append(
            f"| {version['version']} | `{version['pack_hash'][:12]}…` | {version['auto']} | "
            f"{version['review']} | {version['false_auto']} | {version['missed_auto']} | "
            f"{len(version['golden_mismatches'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-stratum auto",
            "",
            "| version | " + " | ".join(report["versions"][-1]["by_group"]) + " |",
            "|---|" + "---:|" * len(report["versions"][-1]["by_group"]),
        ]
    )
    for version in report["versions"]:
        cells = [
            f"{version['by_group'][group]['auto']}/{version['by_group'][group]['total']}"
            for group in version["by_group"]
        ]
        lines.append(f"| {version['version']} | " + " | ".join(cells) + " |")
    comparison = report["comparison"]
    lines.extend(
        [
            "",
            "## Version comparison",
            "",
            f"- baseline: {comparison['baseline_version']}",
            f"- candidate: {comparison['candidate_version']}",
            f"- changed decisions: {comparison['changed_decisions']}",
            f"- newly auto: {comparison['newly_auto']}",
            f"- auto removed: {comparison['auto_removed']}",
            f"- regressions: {len(comparison['regressions'])}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, action="append", dest="packs")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    packs = args.packs or [DEFAULT_ARCHIVE, DEFAULT_CURRENT]
    report = run(packs=packs, fixtures_path=args.fixtures, golden_path=args.golden, as_of=args.as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "versions": [
                    {
                        "version": version["version"],
                        "auto": version["auto"],
                        "review": version["review"],
                        "false_auto": version["false_auto"],
                        "missed_auto": version["missed_auto"],
                    }
                    for version in report["versions"]
                ],
                "regressions": len(report["comparison"]["regressions"]),
            },
            ensure_ascii=False,
        )
    )
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
