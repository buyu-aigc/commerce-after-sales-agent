"""Tiered confirmation (default scope "auto_only"): a click only where a write takes effect on its own.

A write the workflow is certain to hand to a reviewer is submitted straight away (a person checks
it anyway, and it can still be withdrawn). A write that would execute automatically, and every
withdrawal, still waits for the customer's click. Choosing another order always offers cards.
"""

import pytest

from tests import test_v2_conversation_case_decoupling as acceptance
from tests.test_case_withdrawal import planner_for, say, withdraw
from tests.test_write_confirmation import (
    A,
    B,
    C,
    cancel_b,
    card,
    chat,
    clicked,
    planner_returning,
    refund_a,
    reselect,
)

v2_runtime = acceptance.v2_runtime
client = acceptance.client


@pytest.fixture
def tiered_mode(monkeypatch):
    import app.main as main
    from app.llm.customer_presenter import present

    monkeypatch.setenv("CUSTOMER_PLANNING_MODE", "contract")
    monkeypatch.setenv("CUSTOMER_WRITE_CONFIRMATION", "1")
    monkeypatch.delenv("CUSTOMER_CONFIRMATION_SCOPE", raising=False)
    monkeypatch.setattr(main, "present", present)


def confirm_offers(response):
    return [a for a in response.json()["ui_actions"] if a["id"] == "confirm_application"]


@pytest.mark.anyio
async def test_review_route_write_is_submitted_without_a_card(client, v2_runtime, tiered_mode):
    planner, calls = planner_returning(refund_a)  # delivered, 11 days late, R$118.86: reviewed
    response = await chat(client, planner, "tiered-review", "晚到十一天，申请退款")
    assert response.status_code == 200, response.text
    assert not confirm_offers(response)
    cases = v2_runtime["case_service"].list_by_session("tiered-review")
    assert [(c["order_id"], c["status"]) for c in cases] == [(A, "pending_review")]
    assert calls["n"] == 1


@pytest.mark.anyio
async def test_auto_route_write_still_waits_for_the_click(client, v2_runtime, tiered_mode):
    planner, _ = planner_returning(cancel_b)  # pre-shipment, R$32.01: executes automatically
    first = await chat(client, planner, "tiered-auto", "家居那笔帮我取消")
    confirm, _ = card(first)
    assert confirm["order_id"] == B
    assert not v2_runtime["case_service"].list_by_session("tiered-auto")
    await chat(client, planner, "tiered-auto", confirm["message"], **clicked(confirm))
    cases = v2_runtime["case_service"].list_by_session("tiered-auto")
    assert [(c["order_id"], c["status"]) for c in cases] == [(B, "executed")]


@pytest.mark.anyio
async def test_mixed_turn_submits_the_reviewed_write_and_cards_the_automatic_one(
    client, v2_runtime, tiered_mode
):
    planner, _ = planner_returning(refund_a, cancel_b)
    response = await chat(client, planner, "tiered-mixed", "退款那笔，再取消家居那笔")
    assert [a["order_id"] for a in confirm_offers(response)] == [B]
    cases = v2_runtime["case_service"].list_by_session("tiered-mixed")
    assert [(c["order_id"], c["status"]) for c in cases] == [(A, "pending_review")]


@pytest.mark.anyio
async def test_withdrawal_still_needs_a_click(client, v2_runtime, tiered_mode):
    def refund(indices, message):
        return [refund_a(indices, message)]

    planner = planner_for([refund, withdraw()])
    await say(client, planner, "tiered-withdraw", "晚到十一天，申请退款")
    case = v2_runtime["case_service"].list_by_session("tiered-withdraw")[0]
    assert case["status"] == "pending_review"
    second = await say(client, planner, "tiered-withdraw", "撤回刚才的退款申请")
    offer = next(a for a in second.json()["ui_actions"] if a["id"] == "confirm_application")
    assert offer["action_type"] == "withdraw_case"
    assert v2_runtime["case_service"].get(case["case_id"])["status"] == "pending_review"
    await say(
        client,
        planner,
        "tiered-withdraw",
        offer["message"],
        requested_action="confirm_application",
        confirmation_id=offer["confirmation_id"],
    )
    assert v2_runtime["case_service"].get(case["case_id"])["status"] == "withdrawn"


@pytest.mark.anyio
async def test_choosing_another_order_offers_cards_even_for_reviewed_routes(client, v2_runtime, tiered_mode):
    planner, _ = planner_returning(cancel_b)
    first = await chat(client, planner, "tiered-reselect", "家居那笔帮我取消")
    offered = await chat(
        client, planner, "tiered-reselect", "不是这笔订单", **clicked(reselect(first), "reselect_application")
    )
    assert {a["order_id"] for a in confirm_offers(offered)} == {C}  # R$1783.44 cancel: reviewed, still a card
    assert not v2_runtime["case_service"].list_by_session("tiered-reselect")


def test_unknown_routes_keep_the_card(v2_runtime):
    import app.main as main

    assert main._goes_to_review({"order_id": "missing-order", "action_type": "refund_request"}) is False
    assert main._goes_to_review({"intent": "case_withdraw", "case_id": "c1"}) is False
    assert main._goes_to_review({"order_id": B, "action_type": "cancel_order"}) is False  # automatic
    assert main._goes_to_review({"order_id": A, "action_type": "refund_request"}) is True  # reviewed


def test_scope_all_restores_a_card_on_every_write(monkeypatch):
    import app.main as main

    monkeypatch.setenv("CUSTOMER_CONFIRMATION_SCOPE", "all")
    assert main._confirmation_scope() == "all"
    monkeypatch.setenv("CUSTOMER_CONFIRMATION_SCOPE", "something-else")
    assert main._confirmation_scope() == "auto_only"
