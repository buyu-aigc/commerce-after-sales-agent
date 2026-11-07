"""Writes wait for the customer's click on a confirmation card (CUSTOMER_WRITE_CONFIRMATION=1).

The click, not the model's reading of the wording, authorizes a write. Tokens are single use, bound to
the customer's session, expire, and re-run only the stored task without another model call.
"""

from unittest.mock import patch

import pytest

from app.intent.turn_contract import TaskProposal, TurnProposal, compile_turn
from tests import test_v2_conversation_case_decoupling as acceptance

v2_runtime = acceptance.v2_runtime
client = acceptance.client
A, B = acceptance.A, acceptance.B


def planner_returning(*make_tasks):
    calls = {"n": 0}

    async def planner(self, **kw):
        calls["n"] += 1
        indices = {o["order_id"]: i + 1 for i, o in enumerate(kw["orders"])}
        return compile_turn(
            TurnProposal(tasks=[make(indices, kw["message"]) for make in make_tasks]),
            message=kw["message"],
            orders=kw["orders"],
            cases=kw["cases"],
            allowed=kw["allowed"],
            user_id=kw["user_id"],
            session_id=kw["session_id"],
        )

    return planner, calls


def refund_a(indices, message):
    return TaskProposal(
        kind="application",
        order_index=indices[A],
        action="refund_request",
        disposition="requested",
        evidence_quote=message,
    )


def read_b(indices, message):
    return TaskProposal(kind="order_read", order_index=indices[B], evidence_quote=message)


@pytest.fixture
def confirmation_mode(monkeypatch):
    import app.main as main
    from app.llm.customer_presenter import present

    monkeypatch.setenv("CUSTOMER_PLANNING_MODE", "contract")
    monkeypatch.setenv("CUSTOMER_WRITE_CONFIRMATION", "1")
    # Card mechanics on every write; the tiered default is covered in test_tiered_confirmation.py.
    monkeypatch.setenv("CUSTOMER_CONFIRMATION_SCOPE", "all")
    monkeypatch.setattr(main, "present", present)


async def chat(client, planner, session, message, **extra):
    with patch("app.main.CustomerTurnPlanner.plan", planner), acceptance._guard()[0], acceptance._guard()[1]:
        return await client.post("/customer/chat", json={"session_id": session, "message": message, **extra})


def card(response):
    actions = response.json()["ui_actions"]
    confirm = next(a for a in actions if a["id"] == "confirm_application")
    decline = next(a for a in actions if a["id"] == "decline_application")
    return confirm, decline


def clicked(action, kind="confirm_application"):
    return {"requested_action": kind, "confirmation_id": action["confirmation_id"]}


@pytest.mark.anyio
async def test_write_waits_for_click_and_click_runs_only_the_stored_task(
    client, v2_runtime, confirmation_mode
):
    planner, calls = planner_returning(refund_a)
    first = await chat(client, planner, "confirm-flow", "已经收到但晚到十一天，申请退款")
    assert first.status_code == 200, first.text
    assert not v2_runtime["case_service"].list_by_session("confirm-flow")
    confirm, _ = card(first)
    assert confirm["order_id"] == A and confirm["action_type"] == "refund_request"
    assert len(confirm["confirmation_id"]) == 64
    second = await chat(client, planner, "confirm-flow", confirm["message"], **clicked(confirm))
    assert second.status_code == 200, second.text
    assert calls["n"] == 1  # confirming never asks the model again
    cases = v2_runtime["case_service"].list_by_session("confirm-flow")
    assert len(cases) == 1 and cases[0]["order_id"] == A and cases[0]["action_type"] == "refund_request"
    assert cases[0]["status"] == "pending_review"
    assert v2_runtime["case_service"].count_case_events(cases[0]["case_id"]) == 0


@pytest.mark.anyio
async def test_token_is_single_use(client, v2_runtime, confirmation_mode):
    planner, _ = planner_returning(refund_a)
    confirm, _ = card(await chat(client, planner, "single-use", "申请退款"))
    await chat(client, planner, "single-use", confirm["message"], **clicked(confirm))
    again = await chat(client, planner, "single-use", confirm["message"], **clicked(confirm))
    assert "已失效" in again.json()["answer"]
    assert len(v2_runtime["case_service"].list_by_session("single-use")) == 1


@pytest.mark.anyio
async def test_decline_submits_nothing_and_voids_the_token(client, v2_runtime, confirmation_mode):
    planner, _ = planner_returning(refund_a)
    confirm, decline = card(await chat(client, planner, "decline", "申请退款"))
    declined = await chat(
        client, planner, "decline", decline["message"], **clicked(decline, "decline_application")
    )
    assert "先不提交" in declined.json()["answer"]
    late = await chat(client, planner, "decline", confirm["message"], **clicked(confirm))
    assert "已失效" in late.json()["answer"]
    assert not v2_runtime["case_service"].list_by_session("decline")


@pytest.mark.anyio
@pytest.mark.parametrize("token", ["0" * 64, None])
async def test_forged_or_missing_token_submits_nothing(client, v2_runtime, confirmation_mode, token):
    planner, _ = planner_returning(refund_a)
    await chat(client, planner, "forged", "申请退款")
    extra = {"requested_action": "confirm_application", **({"confirmation_id": token} if token else {})}
    response = await chat(client, planner, "forged", "确认提交退款申请", **extra)
    assert response.status_code == 200 and "已失效" in response.json()["answer"]
    assert not v2_runtime["case_service"].list_by_session("forged")


@pytest.mark.anyio
async def test_token_is_bound_to_its_session(client, v2_runtime, confirmation_mode):
    planner, _ = planner_returning(refund_a)
    confirm, _ = card(await chat(client, planner, "owner-session", "申请退款"))
    other = await chat(client, planner, "other-session", confirm["message"], **clicked(confirm))
    assert "已失效" in other.json()["answer"]
    assert not v2_runtime["case_service"].list_by_session("other-session")
    assert not v2_runtime["case_service"].list_by_session("owner-session")


@pytest.mark.anyio
async def test_expired_token_submits_nothing(client, v2_runtime, confirmation_mode):
    import app.main as main

    planner, _ = planner_returning(refund_a)
    confirm, _ = card(await chat(client, planner, "expired", "申请退款"))
    config = {"configurable": {"thread_id": "expired"}}
    snapshot = await main.conversation_agent.aget_state(config)
    pending = dict(snapshot.values["pending_confirmations"])
    pending[confirm["confirmation_id"]] = {**pending[confirm["confirmation_id"]], "expires_at": 0}
    await main.conversation_agent.aupdate_state(
        config, {"pending_confirmations": pending}, as_node="finalize_answer"
    )
    late = await chat(client, planner, "expired", confirm["message"], **clicked(confirm))
    assert "已失效" in late.json()["answer"]
    assert not v2_runtime["case_service"].list_by_session("expired")


@pytest.mark.anyio
async def test_reads_in_the_same_turn_are_answered_before_confirmation(client, v2_runtime, confirmation_mode):
    planner, _ = planner_returning(refund_a, read_b)
    first = await chat(client, planner, "mixed", "申请退款并核对另一笔状态")
    body = first.json()
    assert "当前状态" in body["answer"] and "请确认" in body["answer"]
    assert not v2_runtime["case_service"].list_by_session("mixed")
    assert card(first)[0]["order_id"] == A


@pytest.mark.anyio
async def test_ambiguous_object_offers_only_the_candidate_orders(client, v2_runtime, confirmation_mode):
    def ambiguous(indices, message):
        return TaskProposal(
            kind="order_read",
            ambiguity="ambiguous_order",
            candidate_indexes=[indices[A], indices[B]],
            evidence_quote=message,
        )

    planner, _ = planner_returning(ambiguous)
    response = await chat(client, planner, "options", "查一下那笔")
    options = response.json()["ui_actions"]
    assert {o["order_id"] for o in options} == {A, B}
    assert all(o["id"] == "track_order" for o in options)


@pytest.mark.anyio
async def test_write_the_workflow_would_reject_gets_no_card_and_no_case(
    client, v2_runtime, confirmation_mode
):
    def cancel_delivered_a(indices, message):
        return TaskProposal(
            kind="application",
            order_index=indices[A],
            action="cancel_order",
            disposition="requested",
            evidence_quote=message,
        )

    planner, _ = planner_returning(cancel_delivered_a)
    response = await chat(client, planner, "ineligible", "帮我取消这笔订单")
    assert response.status_code == 200, response.text
    assert not [a for a in response.json()["ui_actions"] if a["id"] == "confirm_application"]
    assert not v2_runtime["case_service"].list_by_session("ineligible")


@pytest.mark.anyio
async def test_confirmation_action_is_rejected_outside_contract_mode(client, monkeypatch):
    monkeypatch.setenv("CUSTOMER_PLANNING_MODE", "legacy")
    response = await client.post(
        "/customer/chat",
        json={
            "session_id": "legacy",
            "message": "确认",
            "requested_action": "confirm_application",
            "confirmation_id": "0" * 64,
        },
    )
    assert response.status_code == 422


C = "360787554f41824600bdcba9687cd4f0"  # high-value, pre-shipment, same category as A


def refund_c(indices, message):
    return TaskProposal(
        kind="application",
        order_index=indices[C],
        action="refund_request",
        disposition="requested",
        evidence_quote=message,
    )


def cancel_b(indices, message):
    return TaskProposal(
        kind="application",
        order_index=indices[B],
        action="cancel_order",
        disposition="requested",
        evidence_quote=message,
    )


def reselect(response):
    return next(a for a in response.json()["ui_actions"] if a["id"] == "reselect_application")


@pytest.mark.anyio
async def test_wrong_order_card_offers_the_same_action_on_the_order_meant(
    client, v2_runtime, confirmation_mode
):
    planner, calls = planner_returning(refund_c)
    first = await chat(client, planner, "reselect", "健康美容那笔申请退款")
    wrong, _ = card(first)
    assert wrong["order_id"] == C
    other = reselect(first)
    # Same token as the card; no order_id, so the page does not switch to the rejected order.
    assert other["confirmation_id"] == wrong["confirmation_id"] and not other.get("order_id")
    offered = await chat(
        client, planner, "reselect", other["message"], **clicked(other, "reselect_application")
    )
    assert offered.status_code == 200, offered.text
    assert calls["n"] == 1  # picking another order never asks the model again
    confirm, _ = card(offered)
    assert confirm["order_id"] == A and confirm["action_type"] == "refund_request"
    assert not v2_runtime["case_service"].list_by_session("reselect")
    stale = await chat(client, planner, "reselect", wrong["message"], **clicked(wrong))
    assert "已失效" in stale.json()["answer"]
    await chat(client, planner, "reselect", confirm["message"], **clicked(confirm))
    cases = v2_runtime["case_service"].list_by_session("reselect")
    assert [(c["order_id"], c["action_type"]) for c in cases] == [(A, "refund_request")]


@pytest.mark.anyio
async def test_reselect_falls_back_to_other_orders_when_no_same_category_order_qualifies(
    client, v2_runtime, confirmation_mode
):
    planner, _ = planner_returning(cancel_b)
    first = await chat(client, planner, "reselect-any", "家居那笔帮我取消")
    assert card(first)[0]["order_id"] == B
    offered = await chat(
        client, planner, "reselect-any", "不是这笔订单", **clicked(reselect(first), "reselect_application")
    )
    offers = {a["order_id"] for a in offered.json()["ui_actions"] if a["id"] == "confirm_application"}
    assert offers == {C}  # the only other order a cancellation would be submitted for
    assert not v2_runtime["case_service"].list_by_session("reselect-any")


@pytest.mark.anyio
async def test_reselect_without_another_eligible_order_submits_nothing(
    client, v2_runtime, confirmation_mode, monkeypatch
):
    import app.main as main

    planner, _ = planner_returning(refund_c)
    first = await chat(client, planner, "reselect-none", "申请退款")
    eligible = main._would_submit
    monkeypatch.setattr(main, "_would_submit", lambda task: task.get("order_id") == C and eligible(task))
    offered = await chat(
        client, planner, "reselect-none", "不是这笔订单", **clicked(reselect(first), "reselect_application")
    )
    assert "没有找到其他可以提交退款申请的订单" in offered.json()["answer"]
    assert not [a for a in offered.json()["ui_actions"] if a.get("confirmation_id")]
    assert not v2_runtime["case_service"].list_by_session("reselect-none")


@pytest.mark.anyio
async def test_a_notification_arriving_during_a_click_does_not_replace_the_turn(
    client, v2_runtime, confirmation_mode, monkeypatch
):
    """A case executed by the previous click may notify the session while the next click is recorded."""
    import app.main as main

    planner, _ = planner_returning(refund_a)
    confirm, _ = card(await chat(client, planner, "notify-race", "申请退款"))
    record = main.conversations.append_message

    def notification_lands_right_after(session_id, sender, content, **kwargs):
        result = record(session_id, sender, content, **kwargs)
        if sender == "customer":
            record(session_id, "staff", "售后申请已处理完成。")
        return result

    monkeypatch.setattr(main.conversations, "append_message", notification_lands_right_after)
    response = await chat(client, planner, "notify-race", confirm["message"], **clicked(confirm))
    assert response.status_code == 200, response.text
    cases = v2_runtime["case_service"].list_by_session("notify-race")
    assert [(c["order_id"], c["status"]) for c in cases] == [(A, "pending_review")]
