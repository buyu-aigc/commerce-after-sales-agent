from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.olist.catalog import category_risks_by_name, full_orders_by_id, load_dataset, load_order_facts_index
from app.olist.retrieval import adaptive_category_retrieval


@dataclass(frozen=True)
class OrderStatusView:
    order_id: str
    status: str
    customer_state: str
    purchase_timestamp: str
    estimated_delivery_date: str
    delivered_customer_date: str
    delay_days: int | None
    payment_value: float
    review_score: int | None
    category_summary: str
    refund_status: str | None = None
    address_change_status: str | None = None
    invoice_status: str | None = None


class OrderProjectionStore(Protocol):
    """Read the sandbox state produced by approved after-sales actions."""

    def get_order_projection(self, order_id: str) -> dict[str, Any]:
        """Return the latest derived state for an order."""


class OlistService:
    def __init__(self, projection_store: OrderProjectionStore | None = None) -> None:
        self._by_id = full_orders_by_id()
        self._category_risks = category_risks_by_name()
        self._projection_store = projection_store

    def set_projection_store(self, projection_store: OrderProjectionStore) -> None:
        """Attach the durable after-sales projection after DI has been wired."""
        self._projection_store = projection_store

    def metadata(self) -> dict:
        metadata = dict(load_dataset()["metadata"])
        metadata["full_order_count"] = load_order_facts_index().get("metadata", {}).get("order_count")
        metadata["category_risk_count"] = len(self._category_risks)
        return metadata

    def find_order_id(self, text: str) -> str | None:
        match = re.search(r"[0-9a-f]{32}", text.lower())
        return match.group(0) if match else None

    def get_order_status(self, order_id: str) -> OrderStatusView | None:
        order = self._by_id.get(order_id)
        if order is None:
            return None
        projection = self._projection_store.get_order_projection(order_id) if self._projection_store else {}
        categories = [p["category"] for p in order["products"]]
        return OrderStatusView(
            order_id=order["order_id"],
            status=str(projection.get("order_status") or order["status"]),
            customer_state=order["customer_state"],
            purchase_timestamp=order["purchase_timestamp"],
            estimated_delivery_date=order["estimated_delivery_date"],
            delivered_customer_date=order["delivered_customer_date"],
            delay_days=order["delay_days"],
            payment_value=order["payment_value"],
            review_score=order["review_score"],
            category_summary="、".join(sorted(set(categories))),
            refund_status=_as_optional_text(projection.get("refund_status")),
            address_change_status=_as_optional_text(projection.get("address_change_status")),
            invoice_status=_as_optional_text(projection.get("invoice_status")),
        )

    def category_insights(self, query: str) -> list[dict[str, object]]:
        hits = adaptive_category_retrieval(query, sorted(self._category_risks))
        matched_categories = [hit.category for hit in hits if hit.category in self._category_risks]
        if matched_categories:
            categories = matched_categories[:4]
        else:
            categories = sorted(
                self._category_risks,
                key=lambda name: int(self._category_risks[name]["order_count"]),
                reverse=True,
            )[:4]

        insights: list[dict[str, object]] = []
        for cat in categories:
            risk = self._category_risks[cat]
            insights.append(
                {
                    "name": cat,
                    "price": float(risk["avg_payment_value"]),
                    "description": (
                        f"{cat} category has {risk['order_count']} full-data orders, "
                        f"delay_rate={risk['delay_rate']:.2%}, "
                        f"low_review_rate={risk['low_review_rate']:.2%}, "
                        f"cancellation_rate={risk['cancellation_rate']:.2%}."
                    ),
                    "rules": (
                        "Use order_status for exact order facts; use escalation for delayed "
                        "or low-review follow-up."
                    ),
                    "sample_order_ids": risk["sample_order_ids"][:3],
                }
            )
        return insights

    def after_sales_priority_report(
        self,
        query: str = "",
        top_categories: int = 5,
        top_orders: int = 8,
    ) -> dict[str, object]:
        category_rows = []
        for category, risk in self._category_risks.items():
            order_count = int(risk["order_count"])
            risk_score = _ops_risk_score(risk)
            category_rows.append(
                {
                    "category": category,
                    "risk_score": risk_score,
                    "order_count": order_count,
                    "delay_rate": risk["delay_rate"],
                    "low_review_rate": risk["low_review_rate"],
                    "cancellation_rate": risk["cancellation_rate"],
                    "avg_delay_days": risk.get("avg_delay_days", 0),
                    "sample_order_ids": risk["sample_order_ids"][:3],
                    "recommended_action": _category_action(risk),
                }
            )
        high_risk_categories = sorted(
            category_rows,
            key=lambda item: (float(item["risk_score"]), int(item["order_count"])),
            reverse=True,
        )[:top_categories]

        order_rows = []
        for order in self._by_id.values():
            priority_score, reasons = _order_priority(order)
            if priority_score <= 0:
                continue
            order_rows.append(
                {
                    "order_id": order["order_id"],
                    "priority_score": priority_score,
                    "status": order["status"],
                    "category_summary": "、".join(sorted({p["category"] for p in order["products"]})),
                    "delay_days": order["delay_days"],
                    "review_score": order["review_score"],
                    "payment_value": order["payment_value"],
                    "reasons": reasons,
                    "recommended_action": _order_action(order),
                }
            )
        priority_orders = sorted(
            order_rows,
            key=lambda item: (float(item["priority_score"]), float(item["payment_value"] or 0)),
            reverse=True,
        )[:top_orders]

        return {
            "query": query,
            "summary": (
                f"Identified {len(high_risk_categories)} high-risk categories and "
                f"{len(priority_orders)} priority after-sales orders from Olist facts."
            ),
            "high_risk_categories": high_risk_categories,
            "priority_orders": priority_orders,
            "decision_rules": [
                "delay_rate, low_review_rate, cancellation_rate, and order_count drive category priority",
                "delayed/canceled/low-review/high-value orders are ranked for after-sales follow-up",
                "recommended actions are read-only suggestions; refunds/cancellations still require HITL",
            ],
        }

    def escalation_draft(self, order_id: str) -> dict | None:
        status = self.get_order_status(order_id)
        if status is None:
            return None

        reasons = []
        if status.delay_days is not None and status.delay_days > 0:
            reasons.append(f"delivery delayed by {status.delay_days} day(s)")
        if status.review_score is not None and status.review_score <= 2:
            reasons.append(f"low review score {status.review_score}")
        if status.status == "canceled":
            reasons.append("order was canceled")
        if not reasons:
            reasons.append("customer follow-up requested")

        message = (
            f"您好，我们正在跟进您的订单 {order_id[:8]}。"
            f"当前状态为 {status.status}，类目：{status.category_summary}。"
            f"本次跟进原因：{'; '.join(reasons)}。"
            "我们会核查物流与售后记录，并在确认后提供补偿或后续处理方案。"
        )
        return {
            "order_id": order_id,
            "reason": "; ".join(reasons),
            "message_text": message,
        }


class InMemoryCaseService:
    def __init__(self) -> None:
        self._cases: dict[str, dict] = {}
        self._idempotency_index: dict[str, str] = {}
        self._order_projections: dict[str, dict[str, Any]] = {}
        self._order_events: list[dict[str, Any]] = []

    def open_case(self, order_id: str, message_text: str) -> str:
        result = self.execute_action(
            action_type="open_support_case",
            order_id=order_id,
            message_text=message_text,
        )
        return str(result["result_id"])

    def execute_action(
        self,
        action_type: str,
        order_id: str,
        message_text: str,
        *,
        case_id: str | None = None,
    ) -> dict[str, object]:
        idempotency_key = _idempotency_key(action_type, order_id, message_text)
        if idempotency_key in self._idempotency_index:
            case_id = self._idempotency_index[idempotency_key]
            record = self._cases[case_id]
            was_executed = record.get("status") == "executed"
            event: dict[str, Any] | None = None
            if not was_executed:
                record["status"] = "executed"
                event = self._apply_order_event(action_type, order_id, case_id)
                record["order_event"] = event
            return {
                "result_id": case_id,
                "duplicate": was_executed,
                "idempotency_key": idempotency_key,
                "record": record,
                "order_event": event,
            }

        digest = hashlib.sha1(idempotency_key.encode()).hexdigest()[:10]
        case_id = f"{_case_prefix(action_type)}-{digest}"
        self._idempotency_index[idempotency_key] = case_id
        self._cases[case_id] = {
            "case_id": case_id,
            "action_type": action_type,
            "order_id": order_id,
            "message_text": message_text,
            "status": "executed",
            "idempotency_key": idempotency_key,
        }
        event = self._apply_order_event(action_type, order_id, case_id)
        self._cases[case_id]["order_event"] = event
        return {
            "result_id": case_id,
            "duplicate": False,
            "idempotency_key": idempotency_key,
            "record": self._cases[case_id],
            "order_event": event,
        }

    def reset(self) -> None:
        self._cases.clear()
        self._idempotency_index.clear()
        self._order_projections.clear()
        self._order_events.clear()

    def get(self, case_id: str) -> dict | None:
        return self._cases.get(case_id)

    def get_by_session(self, session_id: str) -> dict | None:
        for record in self._cases.values():
            if record.get("session_id") == session_id:
                return record
        return None

    def submit_for_review(
        self,
        *,
        action_type: str,
        order_id: str,
        message_text: str,
        session_id: str,
        user_id: str,
        expires_at: float | None,
        case_payload: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        tenant_id: str = "olist-demo",
        idempotency_scope: str | None = None,
    ) -> dict[str, object]:
        idempotency_key = _idempotency_key(action_type, order_id, message_text)
        if idempotency_key in self._idempotency_index:
            case_id = self._idempotency_index[idempotency_key]
            return {
                "result_id": case_id,
                "duplicate": True,
                "idempotency_key": idempotency_key,
                "record": self._cases[case_id],
            }
        digest = hashlib.sha1(idempotency_key.encode()).hexdigest()[:10]
        case_id = f"{_case_prefix(action_type)}-{digest}"
        self._idempotency_index[idempotency_key] = case_id
        self._cases[case_id] = {
            "case_id": case_id,
            "action_type": action_type,
            "order_id": order_id,
            "message_text": message_text,
            "status": "pending_review",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "user_id": user_id,
            "expires_at": expires_at,
            "appeal_count": 0,
            "case_payload": json.dumps(case_payload or {}, ensure_ascii=True, sort_keys=True),
        }
        return {
            "result_id": case_id,
            "duplicate": False,
            "idempotency_key": idempotency_key,
            "record": self._cases[case_id],
        }

    def mark_case_result(
        self,
        case_id: str,
        status: str,
        reviewer_id: str = "",
        receipt: dict[str, Any] | None = None,
    ) -> dict | None:
        record = self._cases.get(case_id)
        if record is None:
            return None
        if record.get("status") == status:
            return record
        terminal = {"executed", "rejected", "timeout_canceled", "execution_failed", "withdrawn"}
        if record.get("status") in terminal:
            return record
        record["status"] = status
        record["reviewer_id"] = reviewer_id
        record["execution_receipt"] = json.dumps(receipt or {}, ensure_ascii=True, sort_keys=True)
        record["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return record

    def withdraw_case(self, case_id: str, *, user_id: str, session_id: str) -> dict[str, object]:
        """Same contract as the SQLite store: only an undecided case of this customer can be withdrawn."""
        record = self._cases.get(case_id)
        if record is None:
            return {"ok": False, "error": "not_found"}
        if record.get("user_id") != user_id or record.get("session_id") != session_id:
            return {"ok": False, "error": "owner_required"}
        if record.get("status") == "withdrawn":
            return {"ok": True, "already": True, "record": record}
        if record.get("status") not in {"pending_review", "appealed_pending_review"} or record.get("review_decision"):
            return {"ok": False, "conflict": True, "record": record}
        record["status"] = "withdrawn"
        record["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return {"ok": True, "already": False, "record": record}

    def create_auto_case(
        self,
        *,
        action_type: str,
        order_id: str,
        message_text: str,
        session_id: str,
        user_id: str,
        tenant_id: str = "olist-demo",
        case_payload: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        submitted = self.submit_for_review(
            action_type=action_type,
            order_id=order_id,
            message_text=message_text,
            session_id=session_id,
            user_id=user_id,
            expires_at=None,
            case_payload=case_payload,
        )
        case_id = str(submitted["result_id"])
        self.mark_case_result(case_id, "approving", reviewer_id="system_auto", receipt={"auto": True})
        return {
            "case_id": case_id,
            "duplicate": bool(submitted.get("duplicate")),
            "record": self._cases.get(case_id),
        }

    def mark_review_result(self, session_id: str, status: str, reviewer_id: str = "") -> dict | None:
        record = self.get_by_session(session_id)
        if record is None:
            return None
        record["status"] = status
        record["reviewer_id"] = reviewer_id
        return record

    def appeal_case(self, case_id: str, user_id: str, reason: str) -> dict | None:
        record = self._cases.get(case_id)
        if record is None or record.get("user_id") != user_id:
            return None
        if record.get("status") not in {"rejected", "timeout_canceled"}:
            return record
        record["status"] = "appealed_pending_review"
        record["appeal_reason"] = reason
        record["appeal_count"] = int(record.get("appeal_count") or 0) + 1
        return record

    def list_pending(self, limit: int = 50) -> list[dict]:
        return [
            dict(record)
            for record in self._cases.values()
            if record.get("status") in {"pending_review", "appealed_pending_review"}
        ][:limit]

    def get_order_projection(self, order_id: str) -> dict[str, Any]:
        return dict(self._order_projections.get(order_id, {}))

    def count_order_events(self, order_id: str) -> int:
        return sum(1 for event in self._order_events if event.get("order_id") == order_id)

    def _apply_order_event(self, action_type: str, order_id: str, case_id: str) -> dict[str, Any]:
        projection = self._order_projections.setdefault(order_id, {"order_id": order_id})
        patch = _order_state_patch(action_type)
        projection.update(patch)
        event = {
            "event_id": f"evt-{len(self._order_events) + 1}",
            "case_id": case_id,
            "order_id": order_id,
            "action_type": action_type,
            "event_type": "after_sales_action_executed",
            "state_patch": patch,
        }
        self._order_events.append(event)
        return event


class SQLiteCaseService:
    """Persistent side-effect case store with business-granularity idempotency."""

    def __init__(self, path: str | Path = "data/cases.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def open_case(self, order_id: str, message_text: str) -> str:
        result = self.execute_action(
            action_type="open_support_case",
            order_id=order_id,
            message_text=message_text,
        )
        return str(result["result_id"])

    def execute_action(
        self,
        action_type: str,
        order_id: str,
        message_text: str,
        *,
        case_id: str | None = None,
    ) -> dict[str, object]:
        if case_id:
            return self._execute_bound_case(case_id, action_type, order_id, message_text)
        idempotency_key = _idempotency_key(action_type, order_id, message_text)
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            existing = conn.execute(
                "SELECT * FROM cases WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                record = dict(existing)
                order_event: dict[str, Any] | None = None
                if record["status"] != "executed":
                    conn.execute(
                        """
                        UPDATE cases
                        SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE case_id = ?
                        """,
                        ("executed", record["case_id"]),
                    )
                    record = dict(
                        conn.execute(
                            "SELECT * FROM cases WHERE case_id = ?",
                            (record["case_id"],),
                        ).fetchone()
                    )
                    order_event = self._append_order_event(
                        conn,
                        action_type=action_type,
                        order_id=order_id,
                        case_id=str(record["case_id"]),
                    )
                return {
                    "result_id": record["case_id"],
                    "duplicate": existing["status"] == "executed",
                    "idempotency_key": idempotency_key,
                    "record": record,
                    "order_event": order_event,
                }

            digest = hashlib.sha1(idempotency_key.encode()).hexdigest()[:10]
            case_id = f"{_case_prefix(action_type)}-{digest}"
            conn.execute(
                """
                INSERT INTO cases (
                    case_id, idempotency_key, action_type, order_id,
                    message_text, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (case_id, idempotency_key, action_type, order_id, message_text, "executed"),
            )
            record = dict(
                conn.execute(
                    "SELECT * FROM cases WHERE case_id = ?",
                    (case_id,),
                ).fetchone()
            )
            order_event = self._append_order_event(
                conn,
                action_type=action_type,
                order_id=order_id,
                case_id=case_id,
            )
        return {
            "result_id": case_id,
            "duplicate": False,
            "idempotency_key": idempotency_key,
            "record": record,
            "order_event": order_event,
        }

    def reset(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM case_notifications")
            conn.execute("DELETE FROM case_workflow_jobs")
            conn.execute("DELETE FROM cases")
            conn.execute("DELETE FROM order_events")
            conn.execute("DELETE FROM order_projections")

    def get(self, case_id: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            return dict(row) if row else None

    def get_by_session(self, session_id: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM cases WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def submit_for_review(
        self,
        *,
        action_type: str,
        order_id: str,
        message_text: str,
        session_id: str,
        user_id: str,
        expires_at: float | None,
        case_payload: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        tenant_id: str = "olist-demo",
        idempotency_scope: str | None = None,
        initialize_workflow: bool = False,
    ) -> dict[str, object]:
        idempotency_key = _idempotency_key(action_type, order_id, message_text, scope=idempotency_scope)
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            existing = conn.execute(
                "SELECT * FROM cases WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                record = dict(existing)
                return {
                    "result_id": record["case_id"],
                    "duplicate": True,
                    "idempotency_key": idempotency_key,
                    "record": record,
                }

            digest = hashlib.sha1(idempotency_key.encode()).hexdigest()[:10]
            case_id = f"{_case_prefix(action_type)}-{digest}"
            workflow_thread_id = f"case:{case_id}:attempt:1"
            conn.execute(
                """
                INSERT INTO cases (
                    case_id, idempotency_key, action_type, order_id, message_text,
                    status, created_at, updated_at, session_id, user_id, expires_at,
                    appeal_count, case_payload, conversation_id, tenant_id,
                    workflow_thread_id, workflow_version, workflow_attempt
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    ?, ?, ?, 0, ?, ?, ?, ?, ?, 1
                )
                """,
                (
                    case_id,
                    idempotency_key,
                    action_type,
                    order_id,
                    message_text,
                    "pending_review",
                    session_id,
                    user_id,
                    expires_at,
                    json.dumps(case_payload or {}, ensure_ascii=True, sort_keys=True),
                    conversation_id or session_id,
                    tenant_id,
                    workflow_thread_id,
                    2 if initialize_workflow else 1,
                ),
            )
            record = dict(conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
            if initialize_workflow:
                self._insert_job(conn, record, "init")
        return {
            "result_id": case_id,
            "duplicate": False,
            "idempotency_key": idempotency_key,
            "record": record,
        }

    def create_auto_case(
        self,
        *,
        action_type: str,
        order_id: str,
        message_text: str,
        session_id: str,
        user_id: str,
        tenant_id: str = "olist-demo",
        case_payload: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        submitted = self.submit_for_review(
            action_type=action_type,
            order_id=order_id,
            message_text=message_text,
            session_id=session_id,
            user_id=user_id,
            expires_at=None,
            case_payload=case_payload,
            conversation_id=session_id,
            tenant_id=tenant_id,
        )
        case_id = str(submitted["result_id"])
        self.mark_case_result(case_id, "approving", reviewer_id="system_auto", receipt={"auto": True})
        return {
            "case_id": case_id,
            "duplicate": bool(submitted.get("duplicate")),
            "record": self.get(case_id),
        }

    def list_by_session(
        self,
        session_id: str,
        *,
        statuses: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                rows = conn.execute(
                    f"SELECT * FROM cases WHERE session_id = ? AND status IN ({placeholders}) "
                    "ORDER BY created_at ASC, case_id ASC LIMIT ?",
                    (session_id, *statuses, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM cases WHERE session_id = ? ORDER BY created_at ASC, case_id ASC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def decide_case(
        self,
        case_id: str,
        decision: str,
        reviewer_id: str = "",
        *,
        expected_attempt: int | None = None,
    ) -> dict[str, object]:
        """Atomically claim a review decision. Only one approve/reject wins."""
        if decision not in {"approve", "reject", "timeout"}:
            raise ValueError(f"Unsupported decision: {decision}")
        new_status = {"approve": "approving", "reject": "rejected", "timeout": "timeout_canceled"}[decision]
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                return {"ok": False, "error": "not_found"}
            record = dict(row)
            if expected_attempt is not None and int(record.get("workflow_attempt") or 1) != expected_attempt:
                return {"ok": False, "conflict": True, "record": record}
            if record.get("review_decision") == decision and record.get("status") in {
                "approving",
                "executed",
                "rejected",
                "execution_failed",
                "timeout_canceled",
            }:
                return {"ok": True, "already": True, "record": record}
            if record.get("status") not in {"pending_review", "appealed_pending_review"}:
                return {"ok": False, "conflict": True, "record": record}
            cursor = conn.execute(
                """
                UPDATE cases
                SET review_decision = ?, status = ?, reviewer_id = ?,
                    reviewed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE case_id = ?
                  AND status IN ('pending_review', 'appealed_pending_review')
                  AND (review_decision IS NULL OR review_decision = '')
                """,
                (decision, new_status, reviewer_id, case_id),
            )
            if cursor.rowcount == 0:
                current = dict(conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
                if current.get("review_decision") == decision:
                    return {"ok": True, "already": True, "record": current}
                return {"ok": False, "conflict": True, "record": current}
            current = dict(conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
            self._insert_job(conn, current, "resume")
            self._enqueue_notification(conn, current)
        return {"ok": True, "already": False, "record": current}

    def withdraw_case(self, case_id: str, *, user_id: str, session_id: str) -> dict[str, object]:
        """Customer withdrawal of a case nobody has decided yet.

        Same atomic condition as decide_case: a review decision and a withdrawal cannot
        both win, and a withdrawn case can no longer be approved or executed.
        """
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                return {"ok": False, "error": "not_found"}
            record = dict(row)
            if record.get("user_id") != user_id or record.get("session_id") != session_id:
                return {"ok": False, "error": "owner_required"}
            if record.get("status") == "withdrawn":
                return {"ok": True, "already": True, "record": record}
            cursor = conn.execute(
                """
                UPDATE cases
                SET status = 'withdrawn', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE case_id = ?
                  AND status IN ('pending_review', 'appealed_pending_review')
                  AND (review_decision IS NULL OR review_decision = '')
                """,
                (case_id,),
            )
            current = dict(conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
            if cursor.rowcount == 0:
                return {"ok": False, "conflict": True, "record": current}
            self._enqueue_notification(conn, current)
        return {"ok": True, "already": False, "record": current}

    def mark_case_result(
        self,
        case_id: str,
        status: str,
        reviewer_id: str = "",
        receipt: dict[str, Any] | None = None,
    ) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                return None
            record = dict(row)
            if record.get("status") == status:
                if receipt is not None:
                    conn.execute(
                        "UPDATE cases SET execution_receipt = ? WHERE case_id = ?",
                        (json.dumps(receipt, sort_keys=True), case_id),
                    )
                self._enqueue_notification(conn, record)
                return dict(conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
            terminal = {"executed", "rejected", "timeout_canceled", "execution_failed", "withdrawn"}
            if record.get("status") in terminal:
                return record
            conn.execute(
                """
                UPDATE cases
                SET status = ?, reviewer_id = ?, execution_receipt = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE case_id = ?
                """,
                (
                    status,
                    reviewer_id,
                    json.dumps(receipt or {}, ensure_ascii=True, sort_keys=True),
                    case_id,
                ),
            )
            updated = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            self._enqueue_notification(conn, dict(updated))
        return dict(updated) if updated else None

    def update_workflow(
        self,
        case_id: str,
        workflow_thread_id: str,
        *,
        workflow_version: int = 1,
        workflow_attempt: int = 1,
    ) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                UPDATE cases
                SET workflow_thread_id = ?, workflow_version = ?, workflow_attempt = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE case_id = ?
                """,
                (workflow_thread_id, workflow_version, workflow_attempt, case_id),
            )
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _insert_job(conn, record: dict, job_type: str) -> None:
        attempt = int(record.get("workflow_attempt") or 1)
        conn.execute(
            """INSERT OR IGNORE INTO case_workflow_jobs
            (job_key, case_id, workflow_thread_id, workflow_attempt, job_type,
             status, attempts, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'queued', 0,
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            (
                f"{job_type}:{record['case_id']}:{attempt}",
                record["case_id"],
                record["workflow_thread_id"],
                attempt,
                job_type,
            ),
        )

    @staticmethod
    def _enqueue_notification(conn, record: dict) -> None:
        status = str(record.get("status"))
        labels = {
            "executed": "售后申请已处理完成。",
            "rejected": "售后申请审核未通过，可补充凭证后申诉。",
            "timeout_canceled": "售后申请审核已超时，可补充凭证后申诉。",
            "execution_failed": "售后申请执行未完成，请联系工作人员核查。",
        }
        if status not in labels or not record.get("session_id"):
            return
        if status == "executed" and record.get("action_type") == "refund_request":
            labels[status] = "退款申请已提交，实际退款结果仍需后续确认。"
        key = f"case:{record['case_id']}:attempt:{int(record.get('workflow_attempt') or 1)}:{status}"
        conn.execute(
            """INSERT OR IGNORE INTO case_notifications
            (event_key, case_id, session_id, content, status)
            VALUES (?, ?, ?, ?, 'queued')""",
            (key, record["case_id"], record["session_id"], labels[status]),
        )

    def enqueue_notification(self, case_id: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row:
                self._enqueue_notification(conn, dict(row))

    def pending_notifications(self) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in conn.execute("SELECT * FROM case_notifications WHERE status = 'queued' LIMIT 100")
            ]

    def finish_notification(self, event_key: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE case_notifications SET status = 'done' WHERE event_key = ?", (event_key,))

    def repair_workflow_jobs(self) -> None:
        """Repair pre-fix orphan V2 rows; never guess a legacy session checkpoint."""
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM cases WHERE workflow_thread_id LIKE 'case:%' "
                "AND status IN ('pending_review', 'appealed_pending_review', 'initialization_failed')"
            ).fetchall()
            for row in rows:
                self._insert_job(conn, dict(row), "init")

    def expired_workflow_cases(self) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM cases WHERE workflow_thread_id LIKE 'case:%' "
                    "AND status IN ('pending_review', 'appealed_pending_review') "
                    "AND expires_at IS NOT NULL AND expires_at <= ? LIMIT 50",
                    (time.time(),),
                )
            ]

    def create_workflow_job(
        self,
        case_id: str,
        *,
        workflow_thread_id: str,
        workflow_attempt: int = 1,
        job_key: str | None = None,
        job_type: str = "init",
    ) -> None:
        key = job_key or f"{job_type}:{case_id}:{workflow_attempt}"
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO case_workflow_jobs (
                    job_key, case_id, workflow_thread_id, workflow_attempt,
                    job_type, status, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', 0,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (key, case_id, workflow_thread_id, workflow_attempt, job_type),
            )

    def claim_workflow_job(self, job_key: str, lease_seconds: float = 60.0) -> bool:
        now = time.time()
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute(
                """
                UPDATE case_workflow_jobs
                SET status = 'running', lease_until = ?, attempts = attempts + 1,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE job_key = ?
                  AND status IN ('queued', 'running')
                  AND (lease_until IS NULL OR lease_until <= ?)
                """,
                (now + max(lease_seconds, 1), job_key, now),
            )
            return cursor.rowcount > 0

    def get_workflow_job(self, job_key: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM case_workflow_jobs WHERE job_key = ?", (job_key,)).fetchone()
        return dict(row) if row else None

    def pending_workflow_jobs(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM case_workflow_jobs
                WHERE status IN ('queued', 'running')
                  AND (lease_until IS NULL OR lease_until <= ?)
                ORDER BY updated_at ASC, job_key ASC LIMIT ?
                """,
                (time.time(), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def finish_workflow_job(self, job_key: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE case_workflow_jobs SET status = 'done', lease_until = NULL, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE job_key = ?",
                (job_key,),
            )

    def fail_workflow_job(self, job_key: str, error: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                UPDATE case_workflow_jobs
                SET status = 'queued', attempts = attempts + 1, last_error = ?,
                    lease_until = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE job_key = ?
                """,
                (error[:1000], time.time() + 5, job_key),
            )

    def count_case_events(self, case_id: str) -> int:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM order_events WHERE case_id = ?", (case_id,)).fetchone()
        return int(row[0]) if row else 0

    def _execute_bound_case(
        self,
        case_id: str,
        action_type: str,
        order_id: str,
        message_text: str,
    ) -> dict[str, object]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                raise ValueError(f"case_not_found:{case_id}")
            record = dict(row)
            if str(record.get("action_type")) != action_type:
                raise ValueError("case_action_mismatch")
            if str(record.get("order_id")) != order_id:
                raise ValueError("case_order_mismatch")
            if str(record.get("status")) == "executed":
                return {
                    "result_id": case_id,
                    "duplicate": True,
                    "idempotency_key": record.get("idempotency_key"),
                    "record": record,
                    "order_event": None,
                }
            legacy_pending = (
                int(record.get("workflow_version") or 1) == 1
                and record.get("status") == "pending_review"
                and not record.get("review_decision")
            )
            # V1's governed finalize node has no Case CAS step. Keep that
            # compatibility path; new independent V2 workflows require a claim.
            if record.get("status") != "approving" and not legacy_pending:
                raise ValueError("case_not_approved")
            conn.execute(
                """
                UPDATE cases
                SET status = 'executed', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE case_id = ?
                """,
                (case_id,),
            )
            updated = dict(conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
            self._enqueue_notification(conn, updated)
            event = self._append_order_event(
                conn,
                action_type=action_type,
                order_id=order_id,
                case_id=case_id,
            )
        return {
            "result_id": case_id,
            "duplicate": False,
            "idempotency_key": updated.get("idempotency_key"),
            "record": updated,
            "order_event": event,
        }

    def mark_review_result(self, session_id: str, status: str, reviewer_id: str = "") -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT case_id FROM cases WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE cases
                SET status = ?, reviewer_id = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE case_id = ?
                """,
                (status, reviewer_id, row["case_id"]),
            )
            updated = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?",
                (row["case_id"],),
            ).fetchone()
            return dict(updated) if updated else None

    def appeal_case(self, case_id: str, user_id: str, reason: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None or row["user_id"] != user_id:
                return None
            if row["status"] not in {"rejected", "timeout_canceled"}:
                return dict(row)
            record = dict(row)
            old_job = conn.execute(
                "SELECT status FROM case_workflow_jobs WHERE job_key = ?",
                (f"resume:{case_id}:{int(record.get('workflow_attempt') or 1)}",),
            ).fetchone()
            if old_job and old_job["status"] != "done":
                raise ValueError("previous_review_finalizing")
            attempt = int(record.get("workflow_attempt") or 1) + 1
            payload = json.loads(record.get("case_payload") or "{}")
            history = list(payload.get("review_history") or [])
            history.append(
                {
                    "attempt": attempt - 1,
                    "decision": record.get("review_decision"),
                    "reviewer_id": record.get("reviewer_id"),
                    "reviewed_at": record.get("reviewed_at"),
                    "reason": record.get("appeal_reason"),
                    "thread_id": record.get("workflow_thread_id"),
                }
            )
            payload.update(review_history=history, appeal_reason=reason)
            draft = dict(payload.get("draft") or {})
            draft["appeal_reason"] = reason
            payload["draft"] = draft
            timeout = max(1, int(payload.get("timeout_seconds") or 900))
            conn.execute(
                """UPDATE cases SET status = 'appealed_pending_review', appeal_reason = ?,
                appeal_count = COALESCE(appeal_count, 0) + 1, workflow_attempt = ?,
                workflow_thread_id = ?, workflow_version = 2, review_decision = NULL,
                reviewer_id = NULL, reviewed_at = NULL, execution_receipt = NULL,
                expires_at = ?, case_payload = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE case_id = ?""",
                (
                    reason,
                    attempt,
                    f"case:{case_id}:attempt:{attempt}",
                    time.time() + timeout,
                    json.dumps(payload, ensure_ascii=True, sort_keys=True),
                    case_id,
                ),
            )
            updated = dict(conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
            self._insert_job(conn, updated, "init")
            return updated

    def list_pending(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM cases
                WHERE status IN ('pending_review', 'appealed_pending_review')
                ORDER BY updated_at ASC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_review_queue(self, limit: int = 200) -> list[dict]:
        """Reviewer queue: pending plus stuck/approving cases."""
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM cases
                WHERE status IN ('pending_review', 'appealed_pending_review', 'approving')
                ORDER BY
                    CASE status WHEN 'approving' THEN 0 ELSE 1 END,
                    updated_at ASC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_order_projection(self, order_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT projection_json FROM order_projections WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(str(row["projection_json"]))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def count_order_events(self, order_id: str) -> int:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM order_events WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    action_type TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(cases)").fetchall()}
            additions = {
                "updated_at": "TEXT",
                "session_id": "TEXT",
                "user_id": "TEXT",
                "reviewer_id": "TEXT",
                "expires_at": "REAL",
                "appeal_count": "INTEGER DEFAULT 0",
                "appeal_reason": "TEXT",
                "case_payload": "TEXT",
                "conversation_id": "TEXT",
                "tenant_id": "TEXT",
                "workflow_thread_id": "TEXT",
                "workflow_version": "INTEGER DEFAULT 1",
                "workflow_attempt": "INTEGER DEFAULT 1",
                "review_decision": "TEXT",
                "reviewed_at": "TEXT",
                "execution_receipt": "TEXT",
            }
            for column, ddl in additions.items():
                if column not in columns:
                    conn.execute(f"ALTER TABLE cases ADD COLUMN {column} {ddl}")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_events (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    state_patch_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_projections (
                    order_id TEXT PRIMARY KEY,
                    projection_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS case_workflow_jobs (
                    job_key TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    workflow_thread_id TEXT NOT NULL,
                    workflow_attempt INTEGER NOT NULL DEFAULT 1,
                    job_type TEXT NOT NULL DEFAULT 'init',
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_until REAL,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS case_notifications (
                    event_key TEXT PRIMARY KEY, case_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL
                )"""
            )
            job_columns = {row[1] for row in conn.execute("PRAGMA table_info(case_workflow_jobs)").fetchall()}
            if "job_type" not in job_columns:
                conn.execute(
                    "ALTER TABLE case_workflow_jobs ADD COLUMN job_type TEXT NOT NULL DEFAULT 'init'"
                )
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS order_events_case_action "
                    "ON order_events(case_id, action_type)"
                )
            except sqlite3.IntegrityError:
                # Historical duplicate events are left untouched; V2 code still
                # checks receipts before claiming an execution.
                pass

    def _append_order_event(
        self,
        conn: sqlite3.Connection,
        *,
        action_type: str,
        order_id: str,
        case_id: str,
    ) -> dict[str, Any] | None:
        """Append an approved action and update its materialized sandbox view."""
        existing = conn.execute(
            "SELECT * FROM order_events WHERE case_id = ? AND action_type = ?",
            (case_id, action_type),
        ).fetchone()
        if existing is not None:
            return None
        event_id = f"evt-{uuid.uuid4().hex[:16]}"
        patch = _order_state_patch(action_type)
        row = conn.execute(
            "SELECT projection_json FROM order_projections WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        projection: dict[str, Any] = {"order_id": order_id}
        if row is not None:
            try:
                loaded = json.loads(str(row[0]))
                if isinstance(loaded, dict):
                    projection.update(loaded)
            except json.JSONDecodeError:
                pass
        projection.update(patch)
        conn.execute(
            """
            INSERT INTO order_events (
                event_id, case_id, order_id, action_type, event_type, state_patch_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                event_id,
                case_id,
                order_id,
                action_type,
                "after_sales_action_executed",
                json.dumps(patch, ensure_ascii=True, sort_keys=True),
            ),
        )
        conn.execute(
            """
            INSERT INTO order_projections (order_id, projection_json, updated_at)
            VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(order_id) DO UPDATE SET
                projection_json = excluded.projection_json,
                updated_at = excluded.updated_at
            """,
            (order_id, json.dumps(projection, ensure_ascii=True, sort_keys=True)),
        )
        return {
            "event_id": event_id,
            "case_id": case_id,
            "order_id": order_id,
            "action_type": action_type,
            "event_type": "after_sales_action_executed",
            "state_patch": patch,
        }


def _idempotency_key(
    action_type: str,
    order_id: str,
    message_text: str,
    *,
    scope: str | None = None,
) -> str:
    reason = _reason_code(action_type, message_text)
    prefix = f"{scope}:" if scope else "olist-demo:"
    return f"{prefix}{action_type}:{order_id}:{reason}"


def _as_optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _order_state_patch(action_type: str) -> dict[str, str]:
    """Map an approved sandbox action to the customer-visible order projection.

    Olist is historical data, so this intentionally models a request lifecycle
    rather than claiming to move real money or alter a real logistics provider.
    """
    patches: dict[str, dict[str, str]] = {
        "cancel_order": {"order_status": "canceled", "cancellation_status": "confirmed"},
        "refund_request": {"refund_status": "requested"},
        "change_address": {"address_change_status": "requested"},
        "invoice_request": {"invoice_status": "requested"},
        "complaint_escalation": {"complaint_status": "opened"},
        "open_support_case": {"support_status": "opened"},
    }
    return dict(patches.get(action_type, {"support_status": "opened"}))


def _reason_code(action_type: str, message_text: str) -> str:
    lowered = message_text.lower()
    reasons = []
    if "delayed" in lowered or "延迟" in lowered:
        reasons.append("delivery_delay")
    if "low review" in lowered or "低分" in lowered:
        reasons.append("low_review")
    if "canceled" in lowered or "取消" in lowered:
        reasons.append("canceled")
    if not reasons:
        reasons.append("generic")
    return "+".join(sorted(reasons))


def _case_prefix(action_type: str) -> str:
    prefixes = {
        "open_support_case": "CASE",
        "refund_request": "REFUND",
        "cancel_order": "CANCEL",
        "change_address": "ADDR",
        "invoice_request": "INV",
        "complaint_escalation": "COMP",
    }
    return prefixes.get(action_type, "CASE")


def format_order_status(status: OrderStatusView | None) -> str:
    if status is None:
        return "没有找到该订单。请确认 order_id 是否完整。"
    delay = "暂无延迟信息" if status.delay_days is None else f"延迟 {status.delay_days} 天"
    lifecycle = []
    if status.refund_status:
        lifecycle.append(f"退款申请：{status.refund_status}")
    if status.address_change_status:
        lifecycle.append(f"改址申请：{status.address_change_status}")
    if status.invoice_status:
        lifecycle.append(f"发票申请：{status.invoice_status}")
    lifecycle_text = f"\n售后处理状态：{'；'.join(lifecycle)}。" if lifecycle else ""
    return (
        f"订单 {status.order_id} 当前状态：{status.status}。\n"
        f"客户州：{status.customer_state}；类目：{status.category_summary}。\n"
        f"下单时间：{status.purchase_timestamp}；预计送达：{status.estimated_delivery_date}；"
        f"实际送达：{status.delivered_customer_date or '未送达'}；{delay}。\n"
        f"支付金额：{status.payment_value:.2f}；"
        f"评价分：{status.review_score if status.review_score is not None else '暂无'}。"
        f"{lifecycle_text}"
    )


def format_after_sales_report(report: dict[str, object]) -> str:
    categories = report.get("high_risk_categories", [])
    orders = report.get("priority_orders", [])
    lines = ["审核台优先处理建议：", str(report.get("summary", "")), "", "高风险类目 Top："]
    for item in categories[:5]:
        lines.append(
            "- {category}: risk_score={risk_score:.3f}, delay={delay_rate:.2%}, "
            "low_review={low_review_rate:.2%}, cancel={cancellation_rate:.2%}; {action}".format(
                category=item["category"],
                risk_score=float(item["risk_score"]),
                delay_rate=float(item["delay_rate"]),
                low_review_rate=float(item["low_review_rate"]),
                cancellation_rate=float(item["cancellation_rate"]),
                action=item["recommended_action"],
            )
        )
    lines.extend(["", "优先跟进订单 Top："])
    for item in orders[:8]:
        lines.append(
            "- {order_id}: score={priority_score:.2f}, status={status}, delay={delay}, "
            "review={review}; {action}".format(
                order_id=item["order_id"],
                priority_score=float(item["priority_score"]),
                status=item["status"],
                delay=item["delay_days"],
                review=item["review_score"],
                action=item["recommended_action"],
            )
        )
    lines.append("")
    lines.append("注意：以上是审核台只读排序建议；退款、取消、改地址、发票和工单创建仍需 HITL 确认。")
    return "\n".join(lines)


def _ops_risk_score(risk: dict) -> float:
    volume_factor = min(int(risk["order_count"]) / 10000, 1.0)
    return round(
        float(risk["delay_rate"]) * 0.35
        + float(risk["low_review_rate"]) * 0.40
        + float(risk["cancellation_rate"]) * 0.15
        + volume_factor * 0.10,
        4,
    )


def _category_action(risk: dict) -> str:
    actions = []
    if float(risk["delay_rate"]) >= 0.08:
        actions.append("check logistics SLA and delayed-order follow-up")
    if float(risk["low_review_rate"]) >= 0.12:
        actions.append("review customer feedback and improve support scripts")
    if float(risk["cancellation_rate"]) >= 0.02:
        actions.append("inspect cancellation reasons before promotion")
    return "; ".join(actions) if actions else "monitor weekly trend"


def _order_priority(order: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    delay_days = order.get("delay_days")
    review_score = order.get("review_score")
    if isinstance(delay_days, int) and delay_days > 0:
        score += min(delay_days, 30) * 0.5
        reasons.append(f"delayed {delay_days} day(s)")
    if isinstance(review_score, int) and review_score <= 2:
        score += (3 - review_score) * 4
        reasons.append(f"low review score {review_score}")
    if order.get("status") == "canceled":
        score += 5
        reasons.append("canceled order")
    payment_value = float(order.get("payment_value") or 0)
    if payment_value >= 300:
        score += 2
        reasons.append(f"high value {payment_value:.2f}")
    return round(score, 2), reasons


def _order_action(order: dict) -> str:
    if order.get("status") == "canceled":
        return "prepare cancellation explanation and retention check"
    review_score = order.get("review_score")
    delay_days = order.get("delay_days")
    if isinstance(delay_days, int) and delay_days > 0 and isinstance(review_score, int) and review_score <= 2:
        return "prepare escalation draft and proactive apology"
    if isinstance(delay_days, int) and delay_days > 0:
        return "send delayed-delivery follow-up"
    if isinstance(review_score, int) and review_score <= 2:
        return "inspect review issue and support recovery"
    return "monitor"
