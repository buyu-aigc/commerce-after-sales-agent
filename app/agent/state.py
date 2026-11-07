from typing import NotRequired, TypedDict


class AgentState(TypedDict):
    """Serializable state that flows through LangGraph nodes.

    Runtime dependencies live in DI. This state stores only durable business
    data that can be checkpointed and resumed after HITL interrupts.
    """

    session_id: str
    conversation_id: NotRequired[str]
    decouple_cases: NotRequired[bool]
    case_id: NotRequired[str]
    workflow_thread_id: NotRequired[str]
    workflow_attempt: NotRequired[int]
    delegated_pending: NotRequired[dict[str, object]]
    tenant_id: NotRequired[str]
    user_id: NotRequired[str]
    role: NotRequired[str]
    auth_scopes: NotRequired[list[str]]
    channel: NotRequired[str]
    page_context: NotRequired[dict[str, object]]
    requested_action: NotRequired[str]
    validated_turn_plan: NotRequired[dict[str, object] | None]
    # Writes waiting for the customer's click on a confirmation card, by single-use token.
    pending_confirmations: NotRequired[dict[str, object]]
    allowed_order_ids: NotRequired[list[str]]
    messages: list[dict[str, str]]
    route_intent: str
    request_analysis: NotRequired[dict[str, object]]
    shadow_mode: NotRequired[bool]
    shadow_suggestions: NotRequired[list[dict[str, object]]]
    task_plan: list[dict[str, object]]
    completed_tasks: list[dict[str, object]]
    current_task_index: NotRequired[int]
    current_task: NotRequired[dict[str, object]]
    current_task_text: NotRequired[str]
    current_slots: NotRequired[dict[str, object]]
    current_context: NotRequired[dict[str, object]]
    dialog_state: NotRequired[dict[str, object]]
    active_order_id: NotRequired[str]
    customer_order_context: NotRequired[dict[str, object]]
    filled_slots: NotRequired[dict[str, object]]
    missing_slots: NotRequired[list[str]]
    handoff_reasons: NotRequired[list[str]]
    turn_count: NotRequired[int]
    failure_count: NotRequired[int]
    answer_parts: NotRequired[list[str]]
    customer_read_results: NotRequired[list[dict[str, object]]]
    workflow_complete: NotRequired[bool]
    trajectory_events: list[dict[str, object]]
    artifacts: dict[str, object]
    retrieved_insights: list[dict[str, object]]
    retrieved_policy: list[dict[str, object]]
    retrieved_support_docs: list[dict[str, object]]
    after_sales_cases: list[dict[str, object]]
    escalation_draft: NotRequired[dict[str, object]]
    pending_side_effect: NotRequired[dict[str, object]]
    risk_level: NotRequired[str]
    repeat_claim: NotRequired[bool]
    refund_failure: NotRequired[bool]
    policy_pending: NotRequired[bool]
    policy_decision: NotRequired[dict[str, object]]
    user_response: NotRequired[str]
    final_answer: str
