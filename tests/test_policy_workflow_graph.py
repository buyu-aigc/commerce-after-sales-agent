import pytest

from evaluation.policy_workflow_graph import make_orders, run_scenario


@pytest.mark.anyio
async def test_real_graph_policy_routing_reduces_review():
    specs = make_orders()
    without = await run_scenario(False, specs)
    with_policy = await run_scenario(True, specs)
    assert without["review"] == 20
    assert with_policy["auto"] == 6
    assert with_policy["review"] == 14
    assert with_policy["auto_rate"] == 0.3
