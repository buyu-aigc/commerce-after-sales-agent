from pydantic import BaseModel, Field


class CustomerPageContext(BaseModel):
    """Small, server-validated UI context carried with a customer request."""

    surface: str = Field(default="assistant", min_length=1, max_length=64)
    selected_order_id: str | None = Field(default=None, max_length=64)


class CustomerAction(BaseModel):
    id: str
    label: str
    style: str = "secondary"
    order_id: str | None = None
    message: str | None = None
    # Set on confirmation cards: the single-use token and the write it would submit.
    confirmation_id: str | None = None
    action_type: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    tenant_id: str = Field(default="olist-demo", min_length=1, max_length=64)
    user_id: str = Field(default="demo-customer", min_length=1, max_length=128)
    role: str = Field(default="support_agent", min_length=1, max_length=64)
    auth_scopes: list[str] | None = None
    channel: str = Field(default="internal_debug", min_length=1, max_length=64)
    page_context: CustomerPageContext = Field(default_factory=CustomerPageContext)
    requested_action: str | None = Field(default=None, max_length=64)
    confirmation_id: str | None = Field(default=None, min_length=64, max_length=64)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = []
    ui_actions: list[CustomerAction] = Field(default_factory=list)
    case_id: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    request_analysis: dict = Field(default_factory=dict)
    sub_cases: list[dict] = Field(default_factory=list)


class TraceSummaryResponse(BaseModel):
    total: int
    status_counts: dict[str, int]
    route_intent_counts: dict[str, int]
    avg_latency_ms: float
    p50_latency_ms: float = 0.0
    p95_latency_ms: float


class CaseMetricsResponse(BaseModel):
    total_cases: int
    auto_resolution_rate: float
    hitl_rate: float
    wrong_write_blocked: int
    wrong_write_block_rate: float
    policy_hit_rate: float
    tool_error_rate: float
    p50_latency_ms: float = 0.0
    p95_latency_ms: float
    cost_per_case: float
    cost_sample_count: int


class TraceReplayResponse(BaseModel):
    session_id: str
    traces: list[dict]


class RuntimeObservabilityResponse(BaseModel):
    generated_at: str
    tools: dict
    policy: dict
    recovery: dict
    traces: dict
    cases: dict


class RuntimeStatusResponse(BaseModel):
    mode: str
    model: str
    base_url: str
    runtime_backend: str = "memory"
    rate_limit_per_minute: int = 1000
    shadow_mode: bool = False
    shadow_sample_rate: float = 0.0


class ReviewSessionResponse(BaseModel):
    session_id: str
    has_pending: bool
    pending_side_effect: dict = Field(default_factory=dict)
    escalation_draft: dict = Field(default_factory=dict)
    after_sales_cases: list[dict] = Field(default_factory=list)
    completed_tasks: list[dict] = Field(default_factory=list)
    trajectory_events: list[dict] = Field(default_factory=list)
    customer_safe_summary: str = ""
    order_facts: dict = Field(default_factory=dict)


class ReviewActionRequest(BaseModel):
    expected_attempt: int | None = Field(default=None, ge=1)
    reviewer_id: str = Field(default="demo-reviewer", min_length=1, max_length=128)
    tenant_id: str = Field(default="olist-demo", min_length=1, max_length=64)
    role: str = Field(default="after_sales_operator", min_length=1, max_length=64)
    auth_scopes: list[str] | None = None
    reason: str = Field(default="", max_length=1000)


class CustomerCaseResponse(BaseModel):
    case_id: str
    status: str
    action_type: str
    order_id: str
    message_text: str
    created_at: str
    updated_at: str | None = None
    expires_at: float | None = None
    appeal_count: int = 0
    request_id: str = ""
    request_tier: str = ""
    risk_level: str = ""
    policy_decision: str = ""
    auto_executed: bool = False
    sub_task_index: int | None = None
    sub_task_total: int | None = None
    display_status: str = ""


class CustomerAppealRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=1000)


class ReviewCaseListResponse(BaseModel):
    cases: list[dict]


class DemoScenarioRequest(BaseModel):
    """A fixed, disposable walkthrough used only by the public demo surface."""

    scenario_id: str = Field(min_length=1, max_length=64)


class DemoReviewRequest(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
