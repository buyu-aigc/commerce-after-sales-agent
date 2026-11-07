"""A customer can withdraw a case nobody has decided yet; afterwards it can never be approved or executed.

Withdrawal is an atomic, conditional status change (same condition as a review decision), so a
withdrawal and an approval cannot both win. It never runs through the case-creating workflow.
"""

from unittest.mock import patch

import pytest

from app.intent.turn_contract import TaskProposal, TurnProposal, compile_turn, execution_tasks
from tests import test_v2_conversation_case_decoupling as acceptance

v2_runtime = acceptance.v2_runtime
client = acceptance.client
A = acceptance.A


def planner_for(turns):
    state = {"n": 0}

    async def planner(self, **kw):
        make = turns[min(state["n"], len(turns) - 1)]
        state["n"] += 1
        indices = {o["order_id"]: i + 1 for i, o in enumerate(kw["orders"])}
        return compile_turn(
            TurnProposal(tasks=make(indices, kw["message"])),
            message=kw["message"],
            orders=kw["orders"],
            cases=kw["cases"],
            allowed=kw["allowed"],
            user_id=kw["user_id"],
            session_id=kw["session_id"],
        )

    return planner


def refund(indices, message):
    return [
        TaskProposal(
            kind="application",
            order_index=indices[A],
            action="refund_request",
            disposition="requested",
            evidence_quote=message,
        )
    ]


def withdraw(disposition="requested"):
    def make(indices, message):
        return [TaskProposal(kind="case_withdraw", case_index=1, disposition=disposition, evidence_quote=message)]

    return make


@pytest.fixture
def contract_mode(monkeypatch):
    import app.main as main
    from app.llm.customer_presenter import present

    monkeypatch.setenv("CUSTOMER_PLANNING_MODE", "contract")
    monkeypatch.setattr(main, "present", present)


async def say(client, planner, session, message, **extra):
    with patch("app.main.CustomerTurnPlanner.plan", planner), acceptance._guard()[0], acceptance._guard()[1]:
        return await client.post("/customer/chat", json={"session_id": session, "message": message, **extra})


async def approve(client, case_id):
    return await client.post(
        f"/review/cases/{case_id}/approve", headers=acceptance.REVIEW_HEADERS, json={"expected_attempt": 1}
    )


@pytest.mark.anyio
async def test_withdrawn_case_can_never_be_approved_or_executed(client, v2_runtime, contract_mode):
    planner = planner_for([refund, withdraw()])
    await say(client, planner, "h33", "晚到十一天，申请退款")
    case = v2_runtime["case_service"].list_by_session("h33")[0]
    assert case["status"] == "pending_review"
    reply = await say(client, planner, "h33", "我不想退了，撤回刚才的退款申请")
    assert "已撤回" in reply.json()["answer"]
    assert v2_runtime["case_service"].get(case["case_id"])["status"] == "withdrawn"
    assert (await approve(client, case["case_id"])).status_code == 409
    assert v2_runtime["case_service"].count_case_events(case["case_id"]) == 0


@pytest.mark.anyio
async def test_decided_case_cannot_be_withdrawn(client, v2_runtime, contract_mode):
    planner = planner_for([refund, withdraw()])
    await say(client, planner, "decided", "晚到十一天，申请退款")
    case = v2_runtime["case_service"].list_by_session("decided")[0]
    assert (await approve(client, case["case_id"])).status_code == 200
    reply = await say(client, planner, "decided", "撤回刚才的退款申请")
    assert "无法撤回" in reply.json()["answer"]
    assert v2_runtime["case_service"].get(case["case_id"])["status"] == "executed"
    assert v2_runtime["case_service"].count_case_events(case["case_id"]) == 1


@pytest.mark.anyio
async def test_asking_whether_one_can_withdraw_changes_nothing(client, v2_runtime, contract_mode):
    planner = planner_for([refund, withdraw("question")])
    await say(client, planner, "ask-only", "晚到十一天，申请退款")
    await say(client, planner, "ask-only", "申请提交以后还能撤回吗？")
    assert v2_runtime["case_service"].list_by_session("ask-only")[0]["status"] == "pending_review"


@pytest.mark.anyio
async def test_withdrawal_waits_for_confirmation_when_cards_are_on(client, v2_runtime, contract_mode, monkeypatch):
    monkeypatch.setenv("CUSTOMER_WRITE_CONFIRMATION", "1")
    monkeypatch.setenv("CUSTOMER_CONFIRMATION_SCOPE", "all")
    planner = planner_for([refund, withdraw()])
    first = await say(client, planner, "h33-cards", "晚到十一天，申请退款")
    card = next(a for a in first.json()["ui_actions"] if a["id"] == "confirm_application")
    await say(client, planner, "h33-cards", card["message"], requested_action="confirm_application",
              confirmation_id=card["confirmation_id"])
    case = v2_runtime["case_service"].list_by_session("h33-cards")[0]
    second = await say(client, planner, "h33-cards", "撤回刚才的退款申请")
    assert v2_runtime["case_service"].get(case["case_id"])["status"] == "pending_review"
    card = next(a for a in second.json()["ui_actions"] if a["id"] == "confirm_application")
    assert card["action_type"] == "withdraw_case"
    # A withdrawal names a case, not an order: there is no other order to pick.
    assert not [a for a in second.json()["ui_actions"] if a["id"] == "reselect_application"]
    await say(client, planner, "h33-cards", card["message"], requested_action="confirm_application",
              confirmation_id=card["confirmation_id"])
    assert v2_runtime["case_service"].get(case["case_id"])["status"] == "withdrawn"
    assert (await approve(client, case["case_id"])).status_code == 409


CASE = {"case_id": "c1", "order_id": A, "user_id": "owner", "session_id": "s", "status": "pending_review"}


def compile_withdraw(message="撤回刚才的申请", *, quote=None, case=CASE, **kw):
    return compile_turn(
        TurnProposal(
            tasks=[
                TaskProposal(
                    kind="case_withdraw",
                    case_index=1,
                    disposition=kw.pop("disposition", "requested"),
                    evidence_quote=message if quote is None else quote,
                )
            ]
        ),
        message=message,
        orders=[{"order_id": A}],
        cases=[case],
        allowed={A},
        user_id="owner",
        session_id="s",
        **kw,
    )


def test_explicit_evidenced_withdrawal_compiles_to_a_case_scoped_write():
    task = compile_withdraw()["tasks"][0]
    assert task["intent"] == "case_withdraw" and task["side_effect"] and task["case_id"] == "c1"


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"quote": "替我撤回"}, "unverifiable_evidence"),
        ({"case": {**CASE, "session_id": "other"}}, "case_owner_required"),
        ({"selection_only": True}, "object_confirmation_not_authorization"),
        ({"origin": "unavailable"}, "fallback_no_new_application"),
        ({"disposition": "question"}, "not_authorized"),
    ],
)
def test_withdrawal_needs_evidence_ownership_and_an_explicit_request(kwargs, reason):
    plan = compile_withdraw(**kwargs)
    assert not plan["tasks"][0]["side_effect"]
    assert plan["decisions"][0]["validation"] == reason


def test_case_creating_workflow_never_executes_a_withdrawal():
    with pytest.raises(ValueError):
        execution_tasks(compile_withdraw(), user_id="owner", session_id="s", allowed={A})
