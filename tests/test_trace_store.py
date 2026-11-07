from pathlib import Path

from app.trace_store import case_metrics, list_session_traces, record_trace, trace_summary


def test_record_trace_writes_sqlite_row(tmp_path, monkeypatch) -> None:
    import app.trace_store as trace_store

    monkeypatch.setattr(trace_store, "TRACE_DB", tmp_path / "traces.db")
    trace_id = record_trace(
        session_id="s1",
        user_message="hello",
        result={
            "route_intent": "policy",
            "final_answer": "ok",
            "retrieved_policy": ["Human Approval Policy"],
            "trajectory_events": [{"node": "plan_tasks", "status": "completed"}],
        },
        latency_ms=12.3,
        status="ok",
    )
    assert trace_id
    assert Path(trace_store.TRACE_DB).exists()
    traces = list_session_traces("s1")
    assert traces[0]["trace_id"] == trace_id
    assert "trajectory_json" in traces[0]
    summary = trace_summary()
    assert summary["total"] == 1
    assert summary["status_counts"] == {"ok": 1}


def test_record_trace_redacts_sensitive_fields(tmp_path, monkeypatch) -> None:
    import app.trace_store as trace_store

    order_id = "203096f03d82e0dffbc41ebc2e2bcfb7"
    monkeypatch.setattr(trace_store, "TRACE_DB", tmp_path / "traces.db")
    record_trace(
        session_id="s-redact",
        user_message=f"手机号 18336782812，邮箱 user@example.com，订单 {order_id}",
        result={
            "route_intent": "escalation",
            "final_answer": f"已处理订单 {order_id}，联系 user@example.com",
            "trajectory_events": [{"details": {"order_id": order_id}}],
            "after_sales_cases": [{"order_id": order_id, "customer_reply": "联系 18336782812"}],
        },
        latency_ms=10.0,
        status="ok",
    )

    trace = list_session_traces("s-redact")[0]
    serialized = " ".join(str(value) for value in trace.values())
    assert order_id not in serialized
    assert "18336782812" not in serialized
    assert "user@example.com" not in serialized
    assert "203096f0...cfb7" in serialized
    assert "183****2812" in serialized
    assert "u***@example.com" in serialized


def test_case_metrics_aggregates_after_sales_cases(tmp_path, monkeypatch) -> None:
    import app.trace_store as trace_store

    monkeypatch.setattr(trace_store, "TRACE_DB", tmp_path / "traces.db")
    record_trace(
        session_id="s1",
        user_message="申请退款",
        result={
            "route_intent": "escalation",
            "final_answer": "pending hitl",
            "trajectory_events": [
                {
                    "node": "execute_task_plan",
                    "status": "awaiting_confirmation",
                    "details": {"tool": "assess_after_sales_case"},
                }
            ],
            "after_sales_cases": [
                {
                    "action_type": "refund_request",
                    "policy_refs": ["Compensation Boundary Policy"],
                    "decision": {
                        "outcome": "needs_human_review",
                        "requires_human": True,
                    },
                    "verification": {"required_next_step": "hitl"},
                }
            ],
        },
        latency_ms=1200.0,
        status="ok",
    )
    record_trace(
        session_id="s2",
        user_message="取消已送达订单",
        result={
            "route_intent": "escalation",
            "final_answer": "reject",
            "trajectory_events": [
                {
                    "node": "execute_task_plan",
                    "status": "completed",
                    "details": {"tool": "assess_after_sales_case"},
                }
            ],
            "after_sales_cases": [
                {
                    "action_type": "cancel_order",
                    "policy_refs": ["Address Change Policy"],
                    "decision": {"outcome": "reject", "requires_human": False},
                    "verification": {"required_next_step": "stop"},
                }
            ],
        },
        latency_ms=800.0,
        status="ok",
    )

    metrics = case_metrics()

    assert metrics["total_cases"] == 2
    assert metrics["hitl_rate"] == 0.5
    assert metrics["auto_resolution_rate"] == 0.5
    assert metrics["wrong_write_blocked"] == 1
    assert metrics["policy_hit_rate"] == 1.0
    assert metrics["tool_error_rate"] == 0.0
