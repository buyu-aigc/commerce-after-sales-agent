import tempfile
from pathlib import Path

import pytest

from evaluation.policy_real_order_bench import main_async


@pytest.mark.anyio
async def test_real_order_graph_bench():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "report.json"
        report = await main_async(Path("evaluation/policy_real_order_fixtures.jsonl"), output)
    assert report["without_policy"]["review"] == 15
    assert report["with_policy"]["review"] == 9
    assert report["policy_auto"] == 6
    assert report["false_auto"] == 0
