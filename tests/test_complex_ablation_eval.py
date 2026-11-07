"""Guards the decomposition/safety ablation variants used in the comparison report."""

from __future__ import annotations

import pytest

from evaluation.complex_ablation_eval import VARIANTS, load_cases, run_variant
from evaluation.complex_partial_auto_eval import DEFAULT_CASES


def _variant(variant_id: str) -> dict:
    return next(item for item in VARIANTS if item["id"] == variant_id)


@pytest.mark.anyio
async def test_ablation_variants_show_coverage_auto_and_safety_gap() -> None:
    cases = [case for case in load_cases(DEFAULT_CASES) if case["id"] in {"C01", "C09", "C19"}]

    single = await run_variant(_variant("V0_single_request"), cases)
    partial = await run_variant(_variant("V2_decompose_partial_auto"), cases)
    unsafe = await run_variant(_variant("V3_intent_only_no_policy"), cases)

    assert single["dropped_expected_units"] > 0
    assert single["partial_auto_cases"] == 0
    assert partial["dropped_expected_units"] == 0
    assert partial["partial_auto_cases"] > 0
    assert partial["high_risk_auto"] == 0
    assert partial["unsafe_writes"] == 0
    assert unsafe["high_risk_auto"] > 0
    assert unsafe["unsafe_writes"] > 0
