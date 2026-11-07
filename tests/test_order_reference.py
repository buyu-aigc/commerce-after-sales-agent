"""Fact-binding properties, independent of natural-language test utterances."""

import pytest

from app.intent.order_reference import OrderConstraint, OrderReference, resolve_order_reference
from app.intent.turn_contract import TaskProposal, TurnProposal, compile_turn, execution_tasks

ORDERS = [
    {
        "order_id": "a",
        "category": "home",
        "amount": "20.10",
        "status": "shipped",
        "purchased_at": "2026-01-02 12:00:00",
    },
    {
        "order_id": "b",
        "category": "home",
        "amount": "80.00",
        "status": "delivered",
        "purchased_at": "2026-01-03",
    },
    {"order_id": "c", "category": "baby", "amount": "20.10", "status": "shipped"},
]


def ref(*conditions):
    return OrderReference(
        constraints=[
            OrderConstraint(field=f, operator=op, value=v, evidence_quote="fact") for f, op, v in conditions
        ]
    )


def resolve(reference, orders=ORDERS, allowed=None, evidence=None):
    return resolve_order_reference(
        reference,
        orders=orders,
        allowed={"a", "b", "c"} if allowed is None else allowed,
        evidence=["fact"] if evidence is None else evidence,
    )


def compile_task(task, orders=ORDERS):
    return compile_turn(
        TurnProposal(tasks=[task]),
        message="fact",
        orders=orders,
        cases=[],
        allowed={"a", "b", "c"},
        user_id="owner",
        session_id="s",
    )


def test_category_alone_keeps_real_ambiguity():
    assert resolve(ref(("category", "eq", "home"))) == {
        "status": "ambiguous_order",
        "candidate_ids": ["a", "b"],
    }


@pytest.mark.parametrize("orders", [ORDERS, list(reversed(ORDERS)), ORDERS[1:] + ORDERS[:1]])
def test_conjunctive_matching_is_order_independent(orders):
    assert resolve(ref(("category", "eq", "home"), ("amount", "eq", "20.1")), orders)["candidate_ids"] == [
        "a"
    ]


def test_date_range_uses_dates_not_lexical_timestamp():
    assert resolve(ref(("purchased_at", "gte", "2026-01-02"), ("purchased_at", "lte", "2026-01-02")))[
        "candidate_ids"
    ] == ["a"]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "twenty"])
def test_invalid_amount_is_not_a_selection(value):
    assert resolve(ref(("amount", "eq", value)))["status"] == "invalid_reference_value"


@pytest.mark.parametrize("value", ["2026-02-30", "yesterday", "2026-01-02junk"])
def test_invalid_date_is_not_a_selection(value):
    assert resolve(ref(("purchased_at", "eq", value)))["status"] == "invalid_reference_value"


def test_missing_fact_and_no_match_do_not_fall_back_to_index():
    # A model that reports no matching order cannot be overridden by an index.
    plan = compile_task(
        TaskProposal(kind="order_read", order_index=1, evidence_quote="fact", ambiguity="unknown_order")
    )
    assert plan["tasks"][0]["intent"] == "clarify"
    assert plan["decisions"][0]["validation"] == "unknown_order"
    assert resolve(ref(("delivered_at", "eq", "2026-01-02")))["candidate_ids"] == []


def test_foreign_order_is_never_a_candidate():
    assert resolve(ref(("category", "eq", "baby")), allowed={"a", "b"})["candidate_ids"] == []


def test_empty_allowed_scope_is_respected():
    assert resolve(ref(("category", "eq", "baby")), allowed=set())["candidate_ids"] == []


def test_reference_evidence_must_be_actual_customer_text():
    assert (
        resolve(ref(("category", "eq", "baby")), evidence=["different sibling"])["status"]
        == "unverifiable_reference_evidence"
    )


def test_literal_text_cannot_use_range_comparisons():
    assert resolve(ref(("category", "gte", "baby")))["status"] == "invalid_reference_operator"


def test_ambiguous_candidates_keep_only_owned_existing_orders():
    plan = compile_turn(
        TurnProposal(
            tasks=[
                TaskProposal(
                    kind="order_read",
                    evidence_quote="fact",
                    ambiguity="ambiguous_order",
                    candidate_indexes=[1, 2, 9],
                )
            ]
        ),
        message="fact",
        orders=ORDERS,
        cases=[],
        allowed={"a", "c"},
        user_id="owner",
        session_id="s",
    )
    assert plan["tasks"][0]["intent"] == "clarify"
    assert plan["decisions"][0]["order_resolution"] == {"status": "ambiguous_order", "candidate_ids": ["a"]}
    assert plan["tasks"][0]["candidate_order_ids"] == ["a"]


def test_unique_binding_does_not_resolve_missing_intent():
    plan = compile_task(
        TaskProposal(kind="order_read", order_index=3, evidence_quote="fact", ambiguity="missing_intent")
    )
    assert plan["tasks"][0]["intent"] == "clarify"


def test_declared_ambiguity_with_index_restricts_execution():
    plan = compile_task(
        TaskProposal(
            kind="application",
            disposition="requested",
            action="refund_request",
            order_index=1,
            evidence_quote="fact",
            ambiguity="ambiguous_order",
            candidate_indexes=[1, 2],
        )
    )
    assert plan["decisions"][0]["validation"] == "ambiguous_order"
    assert not plan["tasks"][0]["side_effect"]


@pytest.mark.parametrize(
    "disposition", ["question", "conditional", "deferred", "withdrawn", "uncertain", "background"]
)
def test_resolved_order_never_overrides_authorization(disposition):
    plan = compile_task(
        TaskProposal(
            kind="application",
            action="invoice_request",
            disposition=disposition,
            evidence_quote="fact",
            order_index=3,
        )
    )
    assert not plan["tasks"][0]["side_effect"]


def test_unverified_supporting_quote_cannot_bind_order():
    plan = compile_task(
        TaskProposal(
            kind="application",
            action="invoice_request",
            disposition="requested",
            evidence_quote="fact",
            supporting_quotes=["fabricated"],
            order_index=3,
        )
    )
    assert plan["decisions"][0]["validation"] == "unverifiable_supporting_evidence"


@pytest.mark.parametrize("other", ["baby", "home"])
def test_authorization_conflicts_compare_resolved_objects(other):
    tasks = [
        TaskProposal(
            kind="application",
            action="invoice_request",
            disposition=d,
            evidence_quote="fact",
            order_index=index,
        )
        for d, index in [("requested", 3), ("deferred", 3 if other == "baby" else 1)]
    ]
    plan = compile_turn(
        TurnProposal(tasks=tasks),
        message="fact",
        orders=ORDERS,
        cases=[],
        allowed={"a", "b", "c"},
        user_id="owner",
        session_id="s",
    )
    assert plan["tasks"][0]["side_effect"] == (other == "home")


def test_constraints_do_not_reclassify_policy_into_status():
    plan = compile_task(TaskProposal(kind="policy", evidence_quote="fact", order_index=3))
    assert plan["tasks"][0]["intent"] == "policy"


def test_duplicate_snapshot_rows_are_one_order_not_false_ambiguity():
    assert resolve(ref(("category", "eq", "baby")), ORDERS + [ORDERS[-1]])["status"] == "unique"


@pytest.mark.parametrize("source", ["current", "customer_history", "assistant_history", "absent"])
def test_independently_quoted_constraint_uses_customer_sources_without_duplicate_fields(source):
    # Supporting reasons may come from any customer turn, never from the assistant.
    message = "fact; request" if source == "current" else "request"
    history = (
        [{"sender": "customer" if source == "customer_history" else "assistant", "content": "fact"}]
        if source in {"customer_history", "assistant_history"}
        else []
    )
    task = TaskProposal(
        kind="application",
        action="refund_request",
        disposition="requested",
        evidence_quote="request",
        supporting_quotes=["fact"],
        order_index=3,
    )
    plan = compile_turn(
        TurnProposal(tasks=[task]),
        message=message,
        history=history,
        orders=ORDERS,
        cases=[],
        allowed={"a", "b", "c"},
        user_id="owner",
        session_id="s",
    )
    assert plan["tasks"][0]["side_effect"] == (source in {"current", "customer_history"})
    assert plan["decisions"][0]["validation"] == (
        "accepted" if source in {"current", "customer_history"} else "unverifiable_supporting_evidence"
    )


def test_order_evidence_cannot_replace_current_application_evidence():
    task = TaskProposal(
        kind="application",
        action="refund_request",
        disposition="requested",
        evidence_quote="fabricated request",
        order_index=3,
    )
    plan = compile_task(task)
    assert plan["decisions"][0]["validation"] == "unverifiable_evidence"
    assert not plan["tasks"][0]["side_effect"]


def selection_plan(*, pending=None, action="refund_request", history=None, selected=1):
    return compile_turn(
        TurnProposal(
            tasks=[
                TaskProposal(
                    kind="application",
                    action=action,
                    disposition="requested",
                    order_index=selected,
                    evidence_quote="confirmation",
                )
            ]
        ),
        message="confirmation",
        orders=ORDERS,
        cases=[],
        allowed={"a", "b", "c"},
        user_id="owner",
        session_id="s",
        selection_only=True,
        pending=pending,
        history=[{"sender": "customer", "content": "fact"}] if history is None else history,
    )


def pending_application(**kwargs):
    return compile_task(
        TaskProposal(
            kind="application",
            action="refund_request",
            disposition="requested",
            evidence_quote="fact",
            **kwargs,
        )
    )


@pytest.mark.parametrize(
    "action",
    [
        "refund_request",
        "cancel_order",
        "invoice_request",
        "change_address",
        "complaint_escalation",
        "open_support_case",
    ],
)
def test_object_selection_alone_cannot_create_any_application(action):
    plan = selection_plan(action=action)
    assert not plan["tasks"][0]["side_effect"]
    assert plan["decisions"][0]["validation"] == "object_confirmation_not_authorization"


def test_object_selection_can_resume_matching_pending_application():
    plan = selection_plan(pending=pending_application())
    assert plan["tasks"][0]["side_effect"]
    assert plan["tasks"][0]["authorization_source"] == "pending_application"
    assert (
        execution_tasks(
            plan, user_id="owner", session_id="s", allowed={"a", "b", "c"}, message="confirmation"
        )[0]["order_id"]
        == "a"
    )


def test_selection_does_not_turn_previous_discovery_into_application():
    pending = compile_task(TaskProposal(kind="discover", evidence_quote="fact"))
    assert not selection_plan(pending=pending)["tasks"][0]["side_effect"]


@pytest.mark.parametrize(
    "disposition", ["deferred", "withdrawn", "conditional", "background", "uncertain", "question"]
)
def test_pending_non_authorization_cannot_be_promoted_by_selection(disposition):
    pending = compile_task(
        TaskProposal(
            kind="application", action="refund_request", disposition=disposition, evidence_quote="fact"
        )
    )
    assert not selection_plan(pending=pending)["tasks"][0]["side_effect"]


@pytest.mark.parametrize(
    "field,value",
    [("user_id", "other"), ("session_id", "other"), ("origin", "unavailable"), ("message_sha256", "stale")],
)
def test_pending_application_is_scoped_and_tied_to_last_customer_turn(field, value):
    pending = pending_application()
    pending[field] = value
    assert not selection_plan(pending=pending)["tasks"][0]["side_effect"]


def test_selection_cannot_reuse_stale_or_already_handled_application():
    assert not selection_plan(
        pending=pending_application(),
        history=[
            {"sender": "customer", "content": "fact"},
            {"sender": "customer", "content": "new instruction"},
        ],
    )["tasks"][0]["side_effect"]
    assert not selection_plan(pending=pending_application(order_index=1))["tasks"][0]["side_effect"]


def test_selection_cannot_change_pending_action():
    assert not selection_plan(pending=pending_application(), action="invoice_request")["tasks"][0][
        "side_effect"
    ]


def test_selection_respects_pending_candidate_set():
    pending = pending_application(ambiguity="ambiguous_order", candidate_indexes=[1, 2])
    assert selection_plan(pending=pending, selected=1)["tasks"][0]["side_effect"]
    assert not selection_plan(pending=pending, selected=3)["tasks"][0]["side_effect"]


def test_graph_boundary_rejects_selection_without_pending_authorization_source():
    plan = selection_plan(pending=pending_application())
    plan["tasks"][0]["authorization_source"] = "current_turn"
    with pytest.raises(ValueError, match="turn_task_not_authorized"):
        execution_tasks(plan, user_id="owner", session_id="s", allowed={"a", "b", "c"})
