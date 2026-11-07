from unittest.mock import AsyncMock

import pytest

from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph


def test_policy_pending_routes_to_policy_node():
    assert AgentGraph._route_after_after_sales({"policy_pending": True}) == "policy"


@pytest.mark.anyio
async def test_apply_policy_auto_completes_task():
    actions = object.__new__(AgentActions)
    actions._execute_escalation_draft = AsyncMock(return_value=("done", True, {}))
    state = {
        "completed_tasks": [],
        "current_task_index": 0,
        "escalation_draft": {"action_type": "refund_request"},
        "policy_decision": {"policy_id": "demo_delay_refund_auto_v1"},
        "answer_parts": ["准备处理"],
        "trajectory_events": [],
    }
    result = await actions.apply_policy_auto(state)
    assert result["policy_pending"] is False
    assert result["completed_tasks"][0]["status"] == "completed"
    assert "符合自动处理条件" in result["final_answer"]
    assert result["trajectory_events"][-1]["status"] == "completed"
