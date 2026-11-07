"""V2 coordination between a conversation and independent review workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langgraph.types import Command

from app import observability
from app import support_conversations as conversations
from app.agent.actions import AgentActions
from app.agent.graph import AgentGraph
from app.agent.state import AgentState
from app.olist.service import SQLiteCaseService

logger = logging.getLogger("agent.case_workflow")


class CaseWorkflowService:
    """Create, initialize, resume and recover one workflow per after-sales Case.

    The service intentionally keeps the conversation graph and the case graph
    separate. A Case owns its own LangGraph thread and its own durable job row;
    the conversation only receives a short, idempotent notification.
    """

    def __init__(self, case_service: SQLiteCaseService, actions: AgentActions) -> None:
        self._case_service = case_service
        self._actions = actions
        self._case_graph = None
        self._legacy_agent = None

    def configure(self, checkpointer, legacy_agent=None) -> None:
        self._case_graph = AgentGraph(self._actions).build_case(checkpointer)
        self._legacy_agent = legacy_agent

    @property
    def configured(self) -> bool:
        return self._case_graph is not None

    async def dispatch_case(
        self,
        *,
        action_type: str,
        order_id: str,
        message_text: str,
        state: AgentState,
        draft: dict[str, Any],
        expires_at: float,
        timeout_seconds: int,
        handoff_reasons: list[str],
    ) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError("case_workflow_not_configured")
        t_dispatch = time.perf_counter()
        request_analysis = dict(state.get("request_analysis") or {})
        payload = {
            "draft": dict(draft),
            "sub_task": draft.get("sub_task"),
            "request_id": request_analysis.get("request_id"),
            "request_tier": request_analysis.get("tier"),
            "question": _latest_question(state),
            "completed_tasks": list(state.get("completed_tasks", [])),
            "allowed_order_ids": list(state.get("allowed_order_ids", [])),
            "auth_scopes": list(state.get("auth_scopes", [])),
            "channel": str(state.get("channel", "customer_self_service")),
            "timeout_seconds": int(timeout_seconds),
            "handoff_reasons": list(handoff_reasons),
        }
        t_persist = time.perf_counter()
        submitted = self._case_service.submit_for_review(
            action_type=action_type,
            order_id=order_id,
            message_text=message_text,
            session_id=str(state.get("conversation_id") or state.get("session_id", "unknown")),
            user_id=str(state.get("user_id", "demo-customer")),
            expires_at=expires_at,
            case_payload=payload,
            conversation_id=str(state.get("conversation_id") or state.get("session_id", "unknown")),
            tenant_id=str(state.get("tenant_id", "olist-demo")),
            initialize_workflow=True,
            idempotency_scope=(
                f"{state.get('tenant_id', 'olist-demo')}:{state.get('user_id', 'demo-customer')}"
            ),
        )
        case_id = str(submitted["result_id"])
        record = self._case_service.get(case_id)
        if record is None:
            raise RuntimeError(f"case_not_found:{case_id}")
        thread_id = str(record.get("workflow_thread_id") or f"case:{case_id}:attempt:1")
        if not record.get("workflow_thread_id"):
            record = (
                self._case_service.update_workflow(case_id, thread_id, workflow_version=1, workflow_attempt=1)
                or record
            )
        t_ensure = time.perf_counter()
        try:
            await self.ensure_workflow(case_id)
        except Exception as exc:  # keep the conversation responsive; job remains retryable
            logger.exception("case workflow initialization failed for %s: %s", case_id, exc)
            self._case_service.mark_case_result(
                case_id, "initialization_failed", reviewer_id="system", receipt={"error": str(exc)}
            )
        t_done = time.perf_counter()
        logger.info(
            "CASE_TIMING dispatch case=%s persist_ms=%.2f ensure_ms=%.2f total_ms=%.2f",
            case_id,
            (t_persist - t_dispatch) * 1000,
            (t_done - t_ensure) * 1000,
            (t_done - t_dispatch) * 1000,
        )
        latest = self._case_service.get(case_id) or record
        return latest

    async def ensure_workflow(self, case_id: str) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError("case_workflow_not_configured")
        t_start = time.perf_counter()
        record = self._case_service.get(case_id)
        if record is None:
            raise RuntimeError(f"case_not_found:{case_id}")
        if str(record.get("status")) not in {
            "pending_review",
            "appealed_pending_review",
            "initialization_failed",
        }:
            return record
        thread_id = str(record.get("workflow_thread_id") or "")
        if not thread_id:
            thread_id = f"case:{case_id}:attempt:{int(record.get('workflow_attempt') or 1)}"
            record = self._case_service.update_workflow(case_id, thread_id) or record
        job_key = f"init:{case_id}:{int(record.get('workflow_attempt') or 1)}"
        self._case_service.create_workflow_job(
            case_id,
            workflow_thread_id=thread_id,
            workflow_attempt=int(record.get("workflow_attempt") or 1),
        )
        job = self._case_service.get_workflow_job(job_key)
        if job and job["status"] == "done":
            return record
        if not self._case_service.claim_workflow_job(job_key, lease_seconds=120):
            return record
        config = {"configurable": {"thread_id": thread_id}}
        t_read = time.perf_counter()
        snapshot = await self._case_graph.aget_state(config)
        t_invoke = time.perf_counter()
        if not snapshot.next and not snapshot.values:
            try:
                await self._case_graph.ainvoke(self._initial_state(record), config)
            except Exception as exc:
                self._case_service.fail_workflow_job(job_key, str(exc))
                raise
        t_done = time.perf_counter()
        logger.info(
            "CASE_TIMING ensure case=%s read_ms=%.2f init_ms=%.2f total_ms=%.2f",
            case_id,
            (t_invoke - t_read) * 1000,
            (t_done - t_invoke) * 1000,
            (t_done - t_start) * 1000,
        )
        job_key = f"init:{case_id}:{int(record.get('workflow_attempt') or 1)}"
        self._case_service.finish_workflow_job(job_key)
        if str(record.get("status")) == "initialization_failed":
            self._case_service.mark_case_result(case_id, "pending_review", reviewer_id="system")
        return self._case_service.get(case_id) or record

    async def resume_case(
        self,
        case_id: str,
        decision: str,
        reviewer_id: str,
        *,
        expected_attempt: int | None = None,
    ) -> dict[str, Any]:
        if decision not in {"approve", "reject", "timeout"}:
            raise ValueError(f"unsupported_decision:{decision}")
        t_start = time.perf_counter()
        record = self._case_service.get(case_id)
        if record is None:
            return {"ok": False, "error": "not_found"}
        attempt = int(record.get("workflow_attempt") or 1)
        if expected_attempt is not None and expected_attempt != attempt:
            return {"ok": False, "error": "decision_conflict", "record": record}
        status = str(record.get("status"))
        job_key = f"resume:{case_id}:{attempt}"
        job = self._case_service.get_workflow_job(job_key)
        terminal = {"executed", "rejected", "timeout_canceled", "execution_failed"}
        if status in terminal and (job is None or job["status"] == "done"):
            stored = str(record.get("review_decision") or "")
            if stored == decision or stored == "timeout":
                self._case_service.enqueue_notification(case_id)
                await self.deliver_notifications()
                return {"ok": True, "already": True, "record": record, "result": None}
            return {"ok": False, "error": "case_already_decided", "record": record}
        if status in {"pending_review", "appealed_pending_review", "initialization_failed"}:
            # Ensure even a freshly committed/orphan init job can be reviewed.
            thread = str(record.get("workflow_thread_id") or "")
            if thread.startswith("case:"):
                await self.ensure_workflow(case_id)
                init_job = self._case_service.get_workflow_job(f"init:{case_id}:{attempt}")
                if not init_job or init_job["status"] != "done":
                    return {"ok": False, "error": "workflow_initializing", "record": record}
            expiry = record.get("expires_at")
            timed_out = isinstance(expiry, (int, float)) and time.time() >= float(expiry)
            if timed_out:
                decision = "timeout"
                reviewer_id = "system_timeout"
            decided = self._case_service.decide_case(case_id, decision, reviewer_id, expected_attempt=attempt)
            if not decided.get("ok"):
                return {"ok": False, "error": "decision_conflict", "record": decided.get("record")}
            record = decided["record"]
        elif str(record.get("review_decision")) != decision:
            return {"ok": False, "error": "decision_conflict", "record": record}
        # All decisions, including reject/timeout, need durable graph finalization.
        if not self._case_service.claim_workflow_job(job_key, lease_seconds=120):
            current = self._case_service.get(case_id) or record
            return {"ok": True, "already": True, "record": current, "result": None}
        decision_value = {"approve": "__review_approve__", "reject": "no", "timeout": "__hitl_timeout__"}[
            str(record.get("review_decision") or decision)
        ]
        thread_id = str(record.get("workflow_thread_id") or "")
        legacy = int(record.get("workflow_version") or 1) == 1 and (
            not thread_id or thread_id == record.get("session_id")
        )
        if legacy and self._legacy_agent is not None:
            graph = self._legacy_agent
            thread_id = thread_id or str(record.get("session_id"))
        else:
            graph = self._case_graph
            thread_id = thread_id or f"case:{case_id}:attempt:{attempt}"
        if graph is None:
            return {"ok": False, "error": "workflow_not_configured", "record": record}
        config = {"configurable": {"thread_id": thread_id}}
        t_decided = time.perf_counter()
        try:
            snapshot = await graph.aget_state(config)
            if snapshot.next:
                result = await graph.ainvoke(Command(resume=decision_value), config)
            elif snapshot.values:
                result = dict(snapshot.values)
            else:
                raise RuntimeError("case_checkpoint_missing")
        except Exception as exc:
            logger.exception("case resume failed for %s: %s", case_id, exc)
            self._case_service.fail_workflow_job(job_key, str(exc))
            return {"ok": False, "error": "workflow_failed", "record": self._case_service.get(case_id)}
        t_done = time.perf_counter()
        logger.info(
            "CASE_TIMING resume case=%s decide_ms=%.2f graph_ms=%.2f total_ms=%.2f",
            case_id,
            (t_decided - t_start) * 1000,
            (t_done - t_decided) * 1000,
            (t_done - t_start) * 1000,
        )
        self._case_service.finish_workflow_job(job_key)
        updated = self._case_service.get(case_id) or record
        self._case_service.enqueue_notification(case_id)
        await self.deliver_notifications()
        return {"ok": True, "already": False, "record": updated, "result": result}

    def review_packet(self, case_id: str) -> dict[str, Any] | None:
        record = self._case_service.get(case_id)
        if record is None:
            return None
        payload = _case_payload(record)
        draft = dict(payload.get("draft") or {})
        return {
            "case_id": case_id,
            "session_id": record.get("session_id"),
            "status": record.get("status"),
            "has_pending": str(record.get("status")) in {"pending_review", "appealed_pending_review"},
            "action_type": record.get("action_type"),
            "order_id": record.get("order_id"),
            "expires_at": record.get("expires_at"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "reviewed_at": record.get("reviewed_at"),
            "workflow_thread_id": record.get("workflow_thread_id"),
            "workflow_attempt": record.get("workflow_attempt"),
            "appeal_reason": record.get("appeal_reason"),
            "review_history": payload.get("review_history", []),
            "pending_side_effect": {
                "type": record.get("action_type"),
                "requires_confirmation": True,
                "review_case_id": case_id,
                "workflow_thread_id": record.get("workflow_thread_id"),
                "expires_at": record.get("expires_at"),
                "task_index": draft.get("task_index", 0),
                "handoff_reasons": payload.get("handoff_reasons", []),
            },
            "escalation_draft": draft,
            "after_sales_cases": [draft.get("after_sales_case")] if draft.get("after_sales_case") else [],
            "completed_tasks": payload.get("completed_tasks", []),
            "trajectory_events": [],
        }

    async def recover_jobs(self) -> None:
        if not self.configured:
            return
        observability.record_recovery_run()
        self._case_service.repair_workflow_jobs()
        for record in self._case_service.expired_workflow_cases():
            await self.resume_case(
                record["case_id"],
                "timeout",
                "system_timeout",
                expected_attempt=int(record.get("workflow_attempt") or 1),
            )
        for job in self._case_service.pending_workflow_jobs():
            job_type = str(job.get("job_type") or "init")
            case_id = str(job.get("case_id") or "")
            success = False
            try:
                if job_type == "resume":
                    record = self._case_service.get(case_id)
                    if record is None:
                        self._case_service.fail_workflow_job(str(job["job_key"]), "case_not_found")
                        continue
                    decision = str(record.get("review_decision") or "approve")
                    outcome = await self.resume_case(
                        case_id,
                        decision,
                        str(record.get("reviewer_id") or "recovery"),
                        expected_attempt=int(job.get("workflow_attempt") or 1),
                    )
                    if not outcome.get("ok"):
                        self._case_service.fail_workflow_job(
                            str(job["job_key"]), str(outcome.get("error") or "resume_failed")
                        )
                    else:
                        success = True
                else:
                    await self.ensure_workflow(case_id)
                    success = True
            except Exception as exc:
                logger.exception("case job recovery failed for %s: %s", case_id, exc)
                self._case_service.fail_workflow_job(str(job["job_key"]), str(exc))
            finally:
                observability.record_recovery(job_type, case_id, success=success)
        await self.deliver_notifications()

    async def deliver_notifications(self) -> None:
        """At-least-once delivery with a unique ledger event key across DBs."""
        for item in self._case_service.pending_notifications():
            try:
                # Do not publish a result until its review checkpoint is finalized.
                record = self._case_service.get(item["case_id"]) or {}
                attempt = int(item["event_key"].split(":attempt:")[1].split(":")[0])
                job = self._case_service.get_workflow_job(f"resume:{item['case_id']}:{attempt}")
                if job and job["status"] != "done":
                    continue
                if not record:
                    continue
                await asyncio.to_thread(
                    conversations.append_message,
                    item["session_id"],
                    "staff",
                    item["content"],
                    case_id=item["case_id"],
                    event_key=item["event_key"],
                )
                self._case_service.finish_notification(item["event_key"])
            except Exception:
                logger.exception("case notification remains queued: %s", item["event_key"])

    async def migrate_legacy(self, *, apply: bool = False) -> dict[str, Any]:
        """Bind old session-thread pending cases after checking their checkpoint."""
        report: dict[str, Any] = {"checked": 0, "bound": [], "conflicts": [], "skipped": [], "apply": apply}
        pending = self._case_service.list_pending(limit=500)
        for record in pending:
            case_id = str(record.get("case_id"))
            if record.get("workflow_thread_id"):
                report["skipped"].append({"case_id": case_id, "reason": "already_bound"})
                continue
            if not self._legacy_agent:
                report["skipped"].append({"case_id": case_id, "reason": "legacy_graph_unavailable"})
                continue
            report["checked"] += 1
            session_id = str(record.get("session_id") or "")
            snapshot = await self._legacy_agent.aget_state({"configurable": {"thread_id": session_id}})
            draft = dict((snapshot.values or {}).get("escalation_draft") or {})
            checkpoint_case = str(draft.get("review_case_id") or "")
            if not snapshot.next or checkpoint_case != case_id:
                report["conflicts"].append(
                    {
                        "case_id": case_id,
                        "session_id": session_id,
                        "reason": "checkpoint_mismatch",
                        "checkpoint_case": checkpoint_case,
                        "has_interrupt": bool(snapshot.next),
                    }
                )
                continue
            siblings = [
                item
                for item in pending
                if item.get("session_id") == session_id and not item.get("workflow_thread_id")
            ]
            if len(siblings) > 1:
                report["conflicts"].append(
                    {"case_id": case_id, "session_id": session_id, "reason": "multiple_pending_cases"}
                )
                continue
            if apply:
                self._case_service.update_workflow(
                    case_id, session_id, workflow_version=1, workflow_attempt=1
                )
            report["bound"].append({"case_id": case_id, "session_id": session_id})
        return report

    def _initial_state(self, record: dict[str, Any]) -> AgentState:
        payload = _case_payload(record)
        draft = dict(payload.get("draft") or {})
        case_id = str(record.get("case_id"))
        attempt = int(record.get("workflow_attempt") or 1)
        expires_at = record.get("expires_at")
        pending = {
            "type": str(record.get("action_type") or draft.get("action_type") or "open_support_case"),
            "requires_confirmation": True,
            "task_intent": "escalation",
            "task_index": int(draft.get("task_index") or 0),
            "review_case_id": case_id,
            "created_at": time.time(),
            "expires_at": expires_at,
            "timeout_seconds": int(payload.get("timeout_seconds") or 900),
            "handoff_reasons": list(payload.get("handoff_reasons") or []),
        }
        return {
            "session_id": str(record.get("session_id") or ""),
            "conversation_id": str(record.get("conversation_id") or record.get("session_id") or ""),
            "decouple_cases": False,
            "case_id": case_id,
            "workflow_thread_id": str(record.get("workflow_thread_id") or ""),
            "workflow_attempt": attempt,
            "tenant_id": str(record.get("tenant_id") or "olist-demo"),
            "user_id": str(record.get("user_id") or "demo-customer"),
            "role": "customer",
            "auth_scopes": list(payload.get("auth_scopes") or []),
            "channel": str(payload.get("channel") or "customer_self_service"),
            "page_context": {},
            "requested_action": "",
            "allowed_order_ids": list(payload.get("allowed_order_ids") or []),
            "messages": [{"role": "user", "content": str(payload.get("question") or "")}],
            "route_intent": "escalation",
            "task_plan": [],
            "completed_tasks": list(payload.get("completed_tasks") or []),
            "current_task_index": int(draft.get("task_index") or 0),
            "current_task": {"intent": "escalation", "side_effect": True},
            "current_task_text": str(payload.get("question") or ""),
            "current_slots": {
                "order_id": str(record.get("order_id") or ""),
                "action_type": str(record.get("action_type") or ""),
            },
            "current_context": {},
            "dialog_state": {},
            "active_order_id": str(record.get("order_id") or ""),
            "filled_slots": {"order_id": str(record.get("order_id") or "")},
            "missing_slots": [],
            "handoff_reasons": list(payload.get("handoff_reasons") or []),
            "turn_count": 1,
            "failure_count": 0,
            "answer_parts": [],
            "workflow_complete": False,
            "trajectory_events": [],
            "artifacts": {},
            "retrieved_insights": [],
            "retrieved_policy": [],
            "retrieved_support_docs": [],
            "after_sales_cases": [draft.get("after_sales_case")] if draft.get("after_sales_case") else [],
            "escalation_draft": draft,
            "pending_side_effect": pending,
            "user_response": "",
            "final_answer": "",
        }


def _latest_question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if str(message.get("role")) == "user":
            return str(message.get("content", ""))
    return ""


def _case_payload(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("case_payload")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
