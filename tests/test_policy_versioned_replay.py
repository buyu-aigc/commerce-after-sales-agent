"""P0-2 rollout gates: 600-row versioned replay and rollback drill."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.policy_hardening_drill import rollback_drill
from evaluation.policy_versioned_replay import (
    DEFAULT_ARCHIVE,
    DEFAULT_AS_OF,
    DEFAULT_CURRENT,
    DEFAULT_FIXTURES,
    DEFAULT_GOLDEN,
    run,
)


def test_600_row_versioned_replay_passes_both_versions(tmp_path):
    report = run(
        packs=[DEFAULT_ARCHIVE, DEFAULT_CURRENT],
        fixtures_path=DEFAULT_FIXTURES,
        golden_path=DEFAULT_GOLDEN,
        as_of=DEFAULT_AS_OF,
    )
    assert report["fixtures_count"] == 600
    assert report["ok"], json.dumps(report["comparison"], ensure_ascii=False)
    versions = {version["version"]: version for version in report["versions"]}
    assert set(versions) == {"1.0.0", "1.1.0"}
    for version in versions.values():
        assert version["false_auto"] == 0
        assert version["missed_auto"] == 0
        assert version["golden_mismatches"] == []
        assert version["auto"] == 50
        assert version["review"] == 550
        assert version["by_group"]["control_auto"]["auto"] == 50
        for group, counts in version["by_group"].items():
            if group != "control_auto":
                assert counts["auto"] == 0, (group, counts)
    comparison = report["comparison"]
    assert comparison["changed_decisions"] == 0
    assert comparison["regressions"] == []
    assert len(comparison["improvements"]) == 0


def test_replay_output_is_deterministic(tmp_path):
    output = tmp_path / "replay.json"
    first = run(packs=[DEFAULT_ARCHIVE, DEFAULT_CURRENT])
    second = run(packs=[DEFAULT_ARCHIVE, DEFAULT_CURRENT])
    assert [version["pack_hash"] for version in first["versions"]] == [
        version["pack_hash"] for version in second["versions"]
    ]
    assert first["comparison"] == second["comparison"]
    assert first["fixtures_sha256"] == second["fixtures_sha256"]
    output.write_text(json.dumps(first), encoding="utf-8")
    assert Path(output).exists()


def test_rollback_drill_passes_and_rejects_unsafe_pack():
    report = rollback_drill(
        pack_path=DEFAULT_CURRENT,
        archive_path=DEFAULT_ARCHIVE,
        fixtures_path=DEFAULT_FIXTURES,
        as_of=DEFAULT_AS_OF,
    )
    assert report["ok"], json.dumps(
        {
            "spot": report["spot_check"],
            "rollback": report["rollback"],
            "negative": report["negative_drills"],
        },
        ensure_ascii=False,
    )
    assert report["spot_check"]["checked"] == 50
    assert report["spot_check"]["failures"] == []
    assert report["spot_check"]["auto_by_policy"] == {"demo_delay_refund_auto_v1": 50}
    assert report["rollback"]["hash_matches_archive"] is True
    assert report["rollback"]["restored_equal_current"] is True
    assert report["rollback"]["changed_on_rollback"] == 0
    assert report["negative_drills"]["unsafe_pack"]["engine_rejected"] is True
    assert report["negative_drills"]["priority_conflict"]["ok"] is True
