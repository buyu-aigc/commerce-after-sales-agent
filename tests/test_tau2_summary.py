import pytest

from scripts.summarize_tau2_results import summarize


def test_tau2_summary_computes_pass_k_and_failures() -> None:
    simulations = [
        {
            "task_id": "task_a",
            "termination_reason": "user_stop",
            "duration": 10.0,
            "agent_cost": 0.01,
            "user_cost": 0.002,
            "reward_info": {
                "reward": 1.0,
                "db_check": {"db_match": True},
                "action_checks": [
                    {"tool_type": "read", "action_reward": 1.0},
                    {"tool_type": "write", "action_reward": 1.0},
                ],
                "nl_assertions": [{"met": True}],
            },
        },
        {
            "task_id": "task_a",
            "termination_reason": "user_stop",
            "duration": 20.0,
            "agent_cost": 0.02,
            "user_cost": 0.003,
            "reward_info": {
                "reward": 0.0,
                "db_check": {"db_match": False},
                "action_checks": [
                    {"tool_type": "read", "action_reward": 0.0},
                    {"tool_type": "write", "action_reward": 1.0},
                ],
                "nl_assertions": [{"met": False}],
            },
        },
        {
            "task_id": "task_b",
            "termination_reason": "agent_stop",
            "duration": 30.0,
            "agent_cost": 0.03,
            "reward_info": {"reward": 0.0},
        },
        {
            "task_id": "task_b",
            "termination_reason": "agent_stop",
            "duration": 40.0,
            "agent_cost": 0.04,
            "reward_info": {"reward": 0.0},
        },
        {
            "task_id": "task_c",
            "termination_reason": "infrastructure_error",
            "reward_info": None,
        },
    ]

    result = summarize(simulations)

    assert result["total_simulations"] == 5
    assert result["evaluated_simulations"] == 4
    assert result["total_tasks"] == 2
    assert result["avg_reward"] == 0.25
    assert result["pass_hat_ks"]["pass^1"] == 0.25
    assert result["pass_hat_ks"]["pass^2"] == 0.0
    assert result["avg_agent_cost"] == 0.025
    assert result["agent_cost_coverage"] == {"count": 4, "total": 4}
    assert result["avg_user_cost"] == 0.0025
    assert result["user_cost_coverage"] == {"count": 2, "total": 4}
    assert result["avg_total_cost"] == pytest.approx(0.0175)
    assert result["total_cost_coverage"] == {"count": 2, "total": 4}
    assert result["db_match"] == {"correct": 1, "total": 2}
    assert result["action_match"] == {
        "read": {"correct": 1, "total": 2},
        "write": {"correct": 2, "total": 2},
    }
    assert result["nl_assertions"] == {"correct": 1, "total": 2}
    assert result["termination_counts"] == {
        "agent_stop": 2,
        "infrastructure_error": 1,
        "user_stop": 2,
    }
    assert result["failed_task_ids"] == ["task_b"]
