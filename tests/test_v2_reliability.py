"""Crash boundary and eligibility regressions; real SQLite/services/graphs."""

import json
import sqlite3
import time
from unittest.mock import patch

import pytest

from app import support_conversations as conversations
from app.after_sales.shadow import SUGGEST_AUTO, SUGGEST_REVIEW, _metrics_from_rows
from tests import test_v2_conversation_case_decoupling as acceptance
from tests.test_v2_conversation_case_decoupling import (
    REVIEW_HEADERS,
    A,
    _create_case,
)

v2_runtime = acceptance.v2_runtime
client = acceptance.client


def _notices(case_id):
    return [
        m
        for m in conversations.messages_with_case("reliability")
        if m.get("case_id") == case_id and m.get("event_key")
    ]


def test_case_and_init_job_rollback_together(v2_runtime, monkeypatch):
    service = v2_runtime["case_service"]

    def fail(*args):
        raise RuntimeError("job insert failure")

    monkeypatch.setattr(service, "_insert_job", fail)
    with pytest.raises(RuntimeError, match="job insert failure"):
        service.submit_for_review(
            action_type="refund_request",
            order_id=A,
            message_text="atomic",
            session_id="reliability",
            user_id="demo-customer",
            expires_at=time.time() + 900,
            initialize_workflow=True,
        )
    assert not service.list_by_session("reliability")


@pytest.mark.anyio
async def test_orphan_init_is_repaired(v2_runtime):
    service = v2_runtime["case_service"]
    result = service.submit_for_review(
        action_type="refund_request",
        order_id=A,
        message_text="orphan",
        session_id="reliability",
        user_id="demo-customer",
        expires_at=time.time() + 900,
        case_payload={"draft": {"action_type": "refund_request", "order_id": A, "message_text": "orphan"}},
    )
    case_id = result["result_id"]
    assert service.get_workflow_job(f"init:{case_id}:1") is None
    await v2_runtime["workflow"].recover_jobs()
    assert service.get_workflow_job(f"init:{case_id}:1")["status"] == "done"
    state = await v2_runtime["workflow"]._case_graph.aget_state(
        {"configurable": {"thread_id": service.get(case_id)["workflow_thread_id"]}}
    )
    assert state.next


@pytest.mark.anyio
@pytest.mark.parametrize("decision", ["approve", "reject", "timeout"])
async def test_persisted_decisions_recover_graph_and_notify_once(client, v2_runtime, decision):
    case_id = await _create_case(client, "reliability", "refund_request")
    service = v2_runtime["case_service"]
    assert service.decide_case(case_id, decision, "crashed-reviewer")["ok"]
    await v2_runtime["workflow"].recover_jobs()
    record = service.get(case_id)
    assert (
        record["status"]
        == {"approve": "executed", "reject": "rejected", "timeout": "timeout_canceled"}[decision]
    )
    snapshot = await v2_runtime["workflow"]._case_graph.aget_state(
        {"configurable": {"thread_id": record["workflow_thread_id"]}}
    )
    assert not snapshot.next
    assert service.count_case_events(case_id) == (1 if decision == "approve" else 0)
    assert len(_notices(case_id)) == 1
    await v2_runtime["workflow"].recover_jobs()
    assert len(_notices(case_id)) == 1


@pytest.mark.anyio
async def test_notification_failure_and_ack_crash_are_retryable(client, v2_runtime):
    case_id = await _create_case(client, "reliability", "refund_request")
    workflow = v2_runtime["workflow"]
    service = v2_runtime["case_service"]
    with patch.object(conversations, "append_message", side_effect=RuntimeError("ledger down")):
        result = await workflow.resume_case(case_id, "approve", "reviewer")
    assert result["ok"]
    assert service.pending_notifications()
    assert not _notices(case_id)
    # Ledger commit succeeds but acknowledgement in cases.db is lost.
    with patch.object(service, "finish_notification", side_effect=RuntimeError("ack lost")):
        await workflow.recover_jobs()
    assert len(_notices(case_id)) == 1
    assert service.pending_notifications()
    await workflow.recover_jobs()
    assert len(_notices(case_id)) == 1
    assert not service.pending_notifications()
    assert service.count_case_events(case_id) == 1


@pytest.mark.anyio
async def test_completed_execution_with_unacknowledged_job_is_recovered(client, v2_runtime):
    case_id = await _create_case(client, "reliability", "refund_request")
    workflow = v2_runtime["workflow"]
    service = v2_runtime["case_service"]
    original = workflow._case_graph.ainvoke

    async def crash(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("crash after checkpoint before acknowledgement")

    with patch.object(workflow._case_graph, "ainvoke", crash):
        result = await workflow.resume_case(case_id, "approve", "reviewer")
    assert not result["ok"]
    assert service.get(case_id)["status"] == "executed"
    with sqlite3.connect(service.path) as conn:
        conn.execute("UPDATE case_workflow_jobs SET lease_until = NULL WHERE job_type = 'resume'")
    await workflow.recover_jobs()
    assert service.get_workflow_job(f"resume:{case_id}:1")["status"] == "done"
    assert service.count_case_events(case_id) == 1
    assert len(_notices(case_id)) == 1


@pytest.mark.anyio
async def test_appeal_new_attempt_rejects_stale_approval(client, v2_runtime):
    case_id = await _create_case(client, "reliability", "refund_request")
    service = v2_runtime["case_service"]
    workflow = v2_runtime["workflow"]
    old = service.get(case_id)
    rejected = await client.post(
        f"/review/cases/{case_id}/reject", headers=REVIEW_HEADERS, json={"expected_attempt": 1}
    )
    assert rejected.status_code == 200
    other = await client.post(
        f"/customer/cases/{case_id}/appeal",
        headers={"X-Demo-Customer": "customer-smoke"},
        json={"reason": "new evidence"},
    )
    assert other.status_code in {403, 404}
    appeal = await client.post(f"/customer/cases/{case_id}/appeal", json={"reason": "new evidence"})
    assert appeal.status_code == 200, appeal.text
    new = service.get(case_id)
    assert new["workflow_attempt"] == 2
    assert new["workflow_thread_id"] != old["workflow_thread_id"]
    assert new["review_decision"] is None
    payload = json.loads(new["case_payload"])
    assert payload["appeal_reason"] == "new evidence"
    packet = await client.get(f"/review/cases/{case_id}", headers=REVIEW_HEADERS)
    assert packet.json()["appeal_reason"] == "new evidence"
    assert packet.json()["review_history"][0]["decision"] == "reject"
    assert payload["review_history"][0]["decision"] == "reject"
    for body in ({"expected_attempt": 1}, {}):
        stale = await client.post(f"/review/cases/{case_id}/approve", headers=REVIEW_HEADERS, json=body)
        assert stale.status_code == 409
    snapshot = await workflow._case_graph.aget_state(
        {"configurable": {"thread_id": new["workflow_thread_id"]}}
    )
    assert snapshot.next
    approved = await client.post(
        f"/review/cases/{case_id}/approve", headers=REVIEW_HEADERS, json={"expected_attempt": 2}
    )
    assert approved.status_code == 200, approved.text
    repeat = await client.post(
        f"/review/cases/{case_id}/approve", headers=REVIEW_HEADERS, json={"expected_attempt": 2}
    )
    assert repeat.status_code == 200
    assert service.count_case_events(case_id) == 1
    assert service.get(case_id)["status"] == "executed"
    assert len(_notices(case_id)) == 2  # rejected attempt 1, executed attempt 2


def _row(index, **overrides):
    return dict(
        request_id=f"r{index}",
        suggested_outcome=SUGGEST_AUTO,
        human_decision="approve",
        risk_level="low",
        automation_eligible=1,
        label_source="real_blind_review",
        eligibility_reviewer="independent-reviewer",
        **overrides,
    )


def test_shadow_review_approval_is_not_automation_label():
    rows = [
        dict(
            request_id=f"r{i}", suggested_outcome=SUGGEST_REVIEW, human_decision="approve", risk_level="high"
        )
        for i in range(30)
    ]
    result = _metrics_from_rows(rows)
    assert result["summary"]["review_approved"] == 30
    assert result["summary"]["missed_auto"] == 0
    assert result["summary"]["missed_auto_rate"] is None
    assert result["summary"]["auto_precision"] is None
    assert result["gates"]["recommendation"] == "keep_shadow"


def test_shadow_gate_requires_independent_labels_and_requests():
    rows = [_row(i) for i in range(300)]
    assert _metrics_from_rows(rows)["gates"]["recommendation"] == "review_for_rollout"
    for row in rows:
        row["request_id"] = "one-correlated-request"
    assert _metrics_from_rows(rows)["gates"]["recommendation"] == "keep_shadow"
    rows = [_row(i) for i in range(300)]
    for row in rows:
        row["label_source"] = "synthetic"
    assert _metrics_from_rows(rows)["gates"]["recommendation"] == "keep_shadow"
    rows = [_row(i) for i in range(300)]
    rows[0]["automation_eligible"] = 0
    assert _metrics_from_rows(rows)["gates"]["recommendation"] == "keep_shadow"


@pytest.mark.anyio
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_decision_job_survives_sqlite_checkpointer_restart(client, v2_runtime, tmp_path, decision):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.case_workflow_service import CaseWorkflowService
    from app.olist.service import SQLiteCaseService

    workflow = v2_runtime["workflow"]
    checkpoint = tmp_path / "durable-checkpoints.db"
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint)) as saver:
        workflow.configure(saver)
        case_id = await _create_case(client, "reliability", "refund_request")
        assert v2_runtime["case_service"].decide_case(case_id, decision, "crashed-reviewer")["ok"]
    service = SQLiteCaseService(v2_runtime["case_service"].path)
    recovered = CaseWorkflowService(service, v2_runtime["actions"])
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint)) as saver:
        recovered.configure(saver)
        await recovered.recover_jobs()
        assert service.get(case_id)["status"] == ("executed" if decision == "approve" else "rejected")
        assert service.count_case_events(case_id) == (1 if decision == "approve" else 0)
        assert len(_notices(case_id)) == 1
        await recovered.recover_jobs()
        assert len(_notices(case_id)) == 1


@pytest.mark.anyio
async def test_expired_pending_case_is_finalized_by_recovery(client, v2_runtime):
    case_id = await _create_case(client, "reliability", "refund_request")
    service = v2_runtime["case_service"]
    with sqlite3.connect(service.path) as conn:
        conn.execute("UPDATE cases SET expires_at = ? WHERE case_id = ?", (time.time() - 1, case_id))
    await v2_runtime["workflow"].recover_jobs()
    assert service.get(case_id)["status"] == "timeout_canceled"
    assert service.count_case_events(case_id) == 0
    assert len(_notices(case_id)) == 1


def test_automation_label_is_persisted_separately_and_not_overwritten(tmp_path):
    from tests.test_shadow_mode import _runtime

    shadow, guard, orders, analysis = _runtime(tmp_path)
    suggestions = shadow.record_analysis(
        request_id="labels",
        session_id="labels",
        analysis=analysis,
        order_lookup=orders.get,
        policy_guard=guard,
    )
    auto = next(row for row in suggestions if row["suggested_outcome"] == SUGGEST_AUTO)
    shadow.link_case(
        case_id="case-label",
        request_id="labels",
        sub_task_index=auto["sub_task_index"],
        action_type=auto["action_type"],
    )
    shadow.record_human_decision(case_id="case-label", decision="approve", reviewer_id="business-reviewer")
    assert shadow.metrics()["summary"]["auto_precision"] is None
    assert (
        shadow.store.record_automation_label(
            observation_id=auto["observation_id"],
            eligible=False,
            source="real_blind_review",
            reviewer_id="eligibility-reviewer",
        )
        == 1
    )
    assert (
        shadow.store.record_automation_label(
            observation_id=auto["observation_id"],
            eligible=True,
            source="real_blind_review",
            reviewer_id="second-reviewer",
        )
        == 0
    )
    summary = shadow.metrics()["summary"]
    assert summary["auto_approved"] == 1
    assert summary["unsafe_auto_labels"] == 1
    assert summary["auto_precision"] == 0
    assert shadow.metrics()["gates"]["recommendation"] == "keep_shadow"


@pytest.mark.anyio
async def test_v2_pending_case_cannot_bypass_review_claim(client, v2_runtime):
    case_id = await _create_case(client, "reliability", "refund_request")
    service = v2_runtime["case_service"]
    assert service.get(case_id)["workflow_version"] == 2
    with pytest.raises(ValueError, match="case_not_approved"):
        service.execute_action("refund_request", A, "bypass", case_id=case_id)
    assert service.count_case_events(case_id) == 0


def test_missed_auto_rate_uses_all_eligible_requests_and_requires_complete_labels():
    rows = [_row(0), _row(1), _row(2)]
    rows[1]["suggested_outcome"] = SUGGEST_REVIEW
    rows[2]["suggested_outcome"] = SUGGEST_REVIEW
    rows[2]["automation_eligible"] = 0
    result = _metrics_from_rows(rows)["summary"]
    assert result["missed_auto"] == 1
    assert result["missed_auto_rate"] == 0.5  # FN / (TP + FN), not FN / all review.
    rows[2]["automation_eligible"] = None
    assert _metrics_from_rows(rows)["summary"]["missed_auto_rate"] is None
