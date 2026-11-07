"""Selection-based order binding and tolerant parsing of the turn contract.

The model chooses among numbered, owned orders; the compiler only checks that a
choice exists and is owned. Verbatim customer wording is required for writes,
not for reads. Malformed output restricts one task, never the whole turn.
"""

import json
from unittest.mock import AsyncMock

import pytest

from app.intent.turn_contract import (
    CustomerTurnPlanner,
    TaskProposal,
    TurnProposal,
    compile_turn,
    parse_turn_proposal,
)

ORDERS = [
    {"order_id": "a", "category": "beauty", "amount": 118.86, "delay_days": 11, "actions": [{"id": "x"}]},
    {"order_id": "b", "category": "home", "amount": 32.01, "delay_days": 0, "actions": []},
]
ALLOWED = {"a", "b"}


def compile_one(task, message="原话", history=None):
    return compile_turn(
        TurnProposal(tasks=[task]),
        message=message,
        orders=ORDERS,
        cases=[],
        allowed=ALLOWED,
        user_id="owner",
        session_id="s",
        history=history,
    )


def raw(payload):
    return type("Raw", (), {"content": json.dumps(payload, ensure_ascii=False)})()


async def plan_with(payload):
    model = AsyncMock()
    model.ainvoke.return_value = raw(payload)
    plan = await CustomerTurnPlanner(model).plan(
        message="原话",
        orders=ORDERS,
        cases=[],
        allowed=ALLOWED,
        user_id="owner",
        session_id="s",
        history=[],
    )
    return plan, model


def test_selected_index_binds_owned_order_without_wording_proof():
    plan = compile_one(TaskProposal(kind="order_read", order_index=1, evidence_quote="模型改写过的描述"))
    task = plan["tasks"][0]
    assert task["intent"] == "order_status" and task["order_id"] == "a"
    # Unverified wording never reaches the answer; the customer's message does.
    assert task["text"] == "原话"


def test_read_task_keeps_verified_wording_for_answering():
    plan = compile_one(TaskProposal(kind="policy", evidence_quote="退款条件"), message="请问退款条件是什么")
    assert plan["tasks"][0]["intent"] == "policy" and plan["tasks"][0]["text"] == "退款条件"


def test_foreign_or_missing_index_is_never_bound():
    plan = compile_turn(
        TurnProposal(
            tasks=[
                TaskProposal(kind="order_read", order_index=2),
                TaskProposal(kind="order_read", order_index=5),
            ]
        ),
        message="原话",
        orders=ORDERS,
        cases=[],
        allowed={"a"},
        user_id="owner",
        session_id="s",
    )
    assert [t["intent"] for t in plan["tasks"]] == ["clarify", "clarify"]
    assert [d["validation"] for d in plan["decisions"]] == ["order_owner_required", "invalid_order_reference"]


@pytest.mark.parametrize("quote", ["我要申请退款", "我要 申请退款", "我要申请退款！", "“我要申请退款”"])
def test_application_evidence_ignores_spacing_width_and_edge_punctuation(quote):
    plan = compile_one(
        TaskProposal(
            kind="application",
            action="refund_request",
            disposition="requested",
            order_index=1,
            evidence_quote=quote,
        ),
        message="东西晚到了，我要申请退款！",
    )
    assert plan["tasks"][0]["side_effect"]


def test_application_evidence_keeps_clause_boundaries():
    plan = compile_one(
        TaskProposal(
            kind="application",
            action="refund_request",
            disposition="requested",
            order_index=1,
            evidence_quote="取消母婴那笔帮我退款",
        ),
        message="先别取消。母婴那笔帮我退款",
    )
    assert not plan["tasks"][0]["side_effect"]
    assert plan["decisions"][0]["validation"] == "unverifiable_evidence"


def test_application_still_needs_customer_wording():
    plan = compile_one(
        TaskProposal(
            kind="application",
            action="cancel_order",
            disposition="requested",
            order_index=2,
            evidence_quote="请取消",
        ),
        message="家居那笔什么时候发货",
    )
    assert not plan["tasks"][0]["side_effect"]


@pytest.mark.anyio
async def test_one_malformed_task_does_not_discard_the_turn():
    plan, _ = await plan_with(
        {
            "tasks": [
                {"kind": "order_read", "order_index": 2, "evidence_quote": "原话"},
                {"kind": "refund", "order_index": 1, "disposition": "requested"},
            ]
        }
    )
    assert plan["origin"] == "model"
    assert [t["intent"] for t in plan["tasks"]] == ["order_status", "clarify"]
    assert not any(t["side_effect"] for t in plan["tasks"])
    assert plan["diagnostics"]["task_format_errors"][0]["position"] == 1


@pytest.mark.anyio
async def test_extra_keys_and_unknown_fields_are_ignored_not_fatal():
    plan, _ = await plan_with(
        {
            "tasks": [
                {
                    "kind": "order_read",
                    "order_index": "1",
                    "evidence_quote": "原话",
                    "required_fields": ["status", "shipping_eta"],
                    "order_reference": {"constraints": [{"field": "delay_days"}]},
                    "note": "extra",
                }
            ]
        }
    )
    assert plan["origin"] == "model"
    assert plan["tasks"][0]["intent"] == "order_status" and plan["tasks"][0]["order_id"] == "a"
    assert plan["tasks"][0]["required_fields"] == ["status"]


@pytest.mark.anyio
async def test_output_without_tasks_is_unavailable_and_never_writes():
    plan, _ = await plan_with({"answer": "好的"})
    assert plan["origin"] == "unavailable"
    assert all(not t["side_effect"] for t in plan["tasks"])


@pytest.mark.anyio
async def test_planner_payload_omits_ui_actions():
    _, model = await plan_with({"tasks": [{"kind": "discover", "evidence_quote": "原话"}]})
    payload = json.loads(model.ainvoke.call_args.args[0][1].content)
    assert all("actions" not in order for order in payload["orders"])
    assert payload["orders"][0]["delay_days"] == 11


@pytest.mark.anyio
async def test_planner_sees_customer_facing_category_and_status_names():
    model = AsyncMock()
    model.ainvoke.return_value = raw({"tasks": [{"kind": "discover"}]})
    await CustomerTurnPlanner(model).plan(
        message="原话",
        orders=[{"order_id": "b", "category": "bed_bath_table", "status": "invoiced"}],
        cases=[],
        allowed={"b"},
        user_id="owner",
        session_id="s",
        history=[],
    )
    order = json.loads(model.ainvoke.call_args.args[0][1].content)["orders"][0]
    assert order["category"] == "bed_bath_table" and order["category_label"] == "家居用品"
    assert order["status"] == "invoiced" and order["status_label"] == "待发货"


def test_parse_accepts_common_wrappers():
    proposal, errors = parse_turn_proposal(raw([{"tasks": [{"kind": "discover"}]}]))
    assert [t.kind for t in proposal.tasks] == ["discover"] and not errors


@pytest.mark.anyio
async def test_planner_lists_orders_newest_first_with_plain_shipping_facts_and_binds_by_that_numbering():
    orders = [
        {
            "order_id": "old",
            "category": "bed_bath_table",
            "status": "shipped",
            "purchased_at": "2017-01-02 10:00:00",
        },
        {
            "order_id": "new",
            "category": "bed_bath_table",
            "status": "invoiced",
            "purchased_at": "2017-03-04 09:00:00",
        },
    ]
    model = AsyncMock()
    model.ainvoke.return_value = raw({"tasks": [{"kind": "order_read", "order_index": 1}]})
    plan = await CustomerTurnPlanner(model).plan(
        message="原话",
        orders=orders,
        cases=[],
        allowed={"old", "new"},
        user_id="owner",
        session_id="s",
        history=[],
    )
    seen = json.loads(model.ainvoke.call_args.args[0][1].content)["orders"]
    assert [(o["index"], o["order_id"]) for o in seen] == [(1, "new"), (2, "old")]
    # Same category: only plain shipping facts tell these two apart.
    assert [(o["shipped"], o["delivered"]) for o in seen] == [(False, False), (True, False)]
    assert seen[0]["purchased_on"] == "2017-03-04"
    assert all("in_transit" not in o and "needs_attention" not in o for o in seen)
    assert plan["tasks"][0]["order_id"] == "new"


@pytest.mark.anyio
async def test_request_list_is_recorded_and_never_fatal():
    plan, _ = await plan_with({"requests": ["查一下状态", 3, "顺便退款"], "tasks": [{"kind": "discover"}]})
    assert plan["diagnostics"]["request_count"] == 2
    plan, _ = await plan_with({"requests": "不是列表", "tasks": [{"kind": "discover"}]})
    assert plan["diagnostics"]["request_count"] == 0 and plan["tasks"]


@pytest.mark.parametrize(
    "quote", ["三十多的家居...帮我取消", "三十多的家居…帮我取消", "三十多的家居……这两笔都帮我取消"]
)
def test_application_evidence_accepts_in_order_fragments_of_a_shared_clause(quote):
    plan = compile_one(
        TaskProposal(
            kind="application",
            action="cancel_order",
            disposition="requested",
            order_index=2,
            evidence_quote=quote,
        ),
        message="三十多的家居、三百多的母婴，这两笔都帮我取消。",
    )
    assert plan["tasks"][0]["side_effect"]


@pytest.mark.parametrize("quote", ["帮我取消...三十多的家居", "三十多的家居...帮我退款"])
def test_elided_evidence_must_be_the_customers_words_in_their_order(quote):
    plan = compile_one(
        TaskProposal(
            kind="application",
            action="cancel_order",
            disposition="requested",
            order_index=2,
            evidence_quote=quote,
        ),
        message="三十多的家居、三百多的母婴，这两笔都帮我取消。",
    )
    assert not plan["tasks"][0]["side_effect"]
    assert plan["decisions"][0]["validation"] == "unverifiable_evidence"


@pytest.mark.parametrize(
    "deferred_candidates, request_survives", [([1, 3], True), ([1, 2], False), ([], False)]
)
def test_deferral_only_conflicts_with_a_request_it_could_refer_to(deferred_candidates, request_survives):
    orders = [{"order_id": o, "category": c} for o, c in [("h", "home"), ("g", "baby"), ("b", "home")]]
    message = "家居那笔先别取消，母婴那笔帮我取消"
    deferred = TaskProposal(
        kind="application",
        action="cancel_order",
        disposition="deferred",
        ambiguity="ambiguous_order" if deferred_candidates else "unknown_order",
        candidate_indexes=deferred_candidates,
        evidence_quote="家居那笔先别取消",
    )
    requested = TaskProposal(
        kind="application",
        action="cancel_order",
        disposition="requested",
        order_index=2,
        evidence_quote="母婴那笔帮我取消",
    )
    plan = compile_turn(
        TurnProposal(tasks=[deferred, requested]),
        message=message,
        orders=orders,
        cases=[],
        allowed={"h", "g", "b"},
        user_id="owner",
        session_id="s",
    )
    assert bool(plan["tasks"][1].get("side_effect")) is request_survives
    assert not plan["tasks"][0].get("side_effect")
