#!/usr/bin/env python3
"""Multi-worker / SQLite concurrency drill for V2.

Starts a disposable two-worker uvicorn instance against the current checkout,
then exercises the write boundary:

* concurrent Case creation from different conversations;
* 30 concurrent approvals of the same Case (exactly one business write);
* mixed approve/reject races (one decision wins, no partial write);
* concurrent approvals of different Cases.

The report also scans the server log for ``database is locked``, tracebacks,
and 5xx responses.  It never touches V1 or the port-8000 service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "reports" / "multi-worker-runtime"
DEFAULT_OUTPUT = ROOT / "reports" / "multi_worker_report.json"
TOKEN = "local-review-demo"
CUSTOMER_HEADERS = {"X-Demo-Customer": "demo-customer"}
A = "203096f03d82e0dffbc41ebc2e2bcfb7"  # delivered, delayed -> review
B = "8aec3a066f732dd927ec8fef1752415b"  # invoiced, low value
C = "360787554f41824600bdcba9687cd4f0"  # invoiced, high value -> review
D = "1b9ecfe83cdc259250e1a8aca174f0ad"  # canceled
REVIEW_JSON = {
    "reviewer_id": "mw-reviewer",
    "role": "after_sales_operator",
    "auth_scopes": ["after_sales:write"],
}
REVIEW_SPECS = [
    ("complaint_escalation", A, "这笔订单遇到了问题，我想申请售后处理。"),
    ("refund_request", A, "我想为这笔订单申请退款。"),
    ("invoice_request", A, "我想为这笔订单申请发票。"),
    ("complaint_escalation", C, "这笔订单遇到了问题，我想申请售后处理。"),
    ("refund_request", C, "我想为这笔订单申请退款。"),
    ("invoice_request", C, "我想为这笔订单申请发票。"),
    ("invoice_request", B, "我想为这笔订单申请发票。"),
    ("complaint_escalation", B, "这笔订单遇到了问题，我想申请售后处理。"),
]



def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_runtime(destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for name in ("app", "frontend", "policies"):
        shutil.copytree(ROOT / name, destination / name, ignore=ignore)
    for name in ("olist_derived", "knowledge_base", "rescommons_derived"):
        shutil.copytree(ROOT / "data" / name, destination / "data" / name, ignore=ignore)
    (destination / "data").mkdir(parents=True, exist_ok=True)
    return destination


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _wait_ready(base: str, process: subprocess.Popen, log_path: Path, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early:\n{log_path.read_text(errors='replace')[-4000:]}")
        try:
            with _opener().open(f"{base}/runtime/status", timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception:  # noqa: BLE001 - readiness probe
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server not ready:\n{log_path.read_text(errors='replace')[-4000:]}")


def _start_server(runtime: Path, port: int, workers: int) -> tuple[subprocess.Popen, Path]:
    log_path = runtime / "server.log"
    env = {
        **os.environ,
        "PYTHONPATH": str(runtime),
        "OPENAI_API_KEY": "test-key",
        "DEMO_MODE": "1",
        "REVIEW_API_TOKEN": TOKEN,
        "RUNTIME_STORE_BACKEND": "memory",
        "PYTHONUNBUFFERED": "1",
    }
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                str(workers),
                "--log-level",
                "warning",
            ],
            cwd=runtime,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process, log_path


def _stop_server(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _read_case(db: Path, case_id: str) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            return {"status": None, "events": 0}
        events = conn.execute(
            "SELECT COUNT(*) FROM order_events WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
    return {"status": dict(row).get("status"), "events": int(events)}


def _read_notifications(support_db: Path, case_id: str) -> int:
    with sqlite3.connect(support_db) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE case_id = ? AND sender = 'staff'",
                (case_id,),
            ).fetchone()[0]
        )


async def _create_case(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    action_type: str,
    order_id: str,
    message: str,
) -> tuple[bool, str | None, int]:
    response = await client.post(
        "/customer/chat",
        headers=CUSTOMER_HEADERS,
        json={
            "message": message,
            "session_id": session_id,
            "requested_action": action_type,
            "page_context": {"surface": "order_detail", "selected_order_id": order_id},
        },
    )
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = {}
    return response.status_code == 200, body.get("case_id"), response.status_code


async def _decide(
    client: httpx.AsyncClient, case_id: str, decision: str, reviewer: str
) -> tuple[int, str]:
    response = await client.post(
        f"/review/cases/{case_id}/{decision}",
        headers={"X-Review-Token": TOKEN},
        json={**REVIEW_JSON, "reviewer_id": reviewer},
    )
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = {}
    return response.status_code, str(body.get("answer") or "")


def _codes(results: list[tuple[int, str]]) -> dict[str, int]:
    return {str(code): count for code, count in sorted(Counter(code for code, _ in results).items())}


async def _run_scenarios(base: str, db: Path, support_db: Path) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=base, timeout=90) as client:
        # 1. Concurrent creation of distinct Cases.  Case idempotency is keyed
        # by action+order+reason, so distinct action/order pairs are used.
        create_specs = REVIEW_SPECS
        create_results = await asyncio.gather(
            *[
                _create_case(
                    client,
                    session_id=f"mw-create-{index}",
                    action_type=action,
                    order_id=order,
                    message=message,
                )
                for index, (action, order, message) in enumerate(create_specs)
            ]
        )
        created_ids = [case_id for ok, case_id, _ in create_results if ok and case_id]
        creation = {
            "requests": len(create_results),
            "http_codes": dict(
                sorted(Counter(str(code) for _, _, code in create_results).items())
            ),
            "unique_cases": len(set(created_ids)),
            "requests_detail": [
                {
                    "action": action,
                    "order": order,
                    "case_id": case_id,
                    "http_code": code,
                }
                for (action, order, _message), (ok, case_id, code) in zip(create_specs, create_results)
            ],
            "case_states": [_read_case(db, case_id) for case_id in created_ids],
        }
        creation["ok"] = (
            len(created_ids) == len(create_results)
            and len(set(created_ids)) == len(created_ids)
            and all(
                state["status"]
                in {"pending_review", "appealed_pending_review", "executed", "execution_failed"}
                for state in creation["case_states"]
            )
        )

        # 1b. Idempotent creation: ten identical requests collapse to one Case.
        idempotent_results = await asyncio.gather(
            *[
                _create_case(
                    client,
                    session_id=f"mw-idempotent-{index}",
                    action_type="complaint_escalation",
                    order_id=A,
                    message="这笔订单遇到了问题，我想申请售后处理。",
                )
                for index in range(10)
            ]
        )
        idempotent_ids = [case_id for ok, case_id, _ in idempotent_results if ok and case_id]
        idempotent = {
            "requests": len(idempotent_results),
            "http_codes": dict(
                sorted(Counter(str(code) for _, _, code in idempotent_results).items())
            ),
            "unique_cases": len(set(idempotent_ids)),
        }
        idempotent["ok"] = (
            len(idempotent_ids) == 10
            and len(set(idempotent_ids)) == 1
            and not any(code >= 500 for _, _, code in idempotent_results)
        )

        # 2. Thirty concurrent approvals of one Case.
        ok, same_case, _ = await _create_case(
            client,
            session_id="mw-same-approve",
            action_type="complaint_escalation",
            order_id=A,
            message="这笔订单遇到了问题，我想申请售后处理。",
        )
        assert ok and same_case
        same_results = await asyncio.gather(
            *[_decide(client, same_case, "approve", f"reviewer-{i}") for i in range(30)]
        )
        same = {
            "http_codes": _codes(same_results),
            "case": _read_case(db, same_case),
            "notifications": _read_notifications(support_db, same_case),
            "answers": sorted({answer for _, answer in same_results if answer}),
        }
        same["ok"] = (
            same["case"]["status"] == "executed"
            and same["case"]["events"] == 1
            and same["notifications"] == 1
            and not any(code >= 500 for code, _ in same_results)
        )

        # 3. Mixed approve/reject race on one Case.
        ok, mixed_case, _ = await _create_case(
            client,
            session_id="mw-mixed",
            action_type="refund_request",
            order_id=C,
            message="我想为这笔订单申请退款。",
        )
        assert ok and mixed_case
        mixed_results = await asyncio.gather(
            *[
                _decide(
                    client,
                    mixed_case,
                    "approve" if index % 2 == 0 else "reject",
                    f"mixed-{index}",
                )
                for index in range(30)
            ]
        )
        mixed_state = _read_case(db, mixed_case)
        mixed_notifications = _read_notifications(support_db, mixed_case)
        mixed = {
            "http_codes": _codes(mixed_results),
            "case": mixed_state,
            "notifications": mixed_notifications,
        }
        if mixed_state["status"] == "executed":
            mixed["ok"] = mixed_state["events"] == 1 and mixed_notifications == 1
        elif mixed_state["status"] == "rejected":
            mixed["ok"] = mixed_state["events"] == 0 and mixed_notifications == 0
        else:
            mixed["ok"] = False
        mixed["ok"] = bool(mixed["ok"] and not any(code >= 500 for code, _ in mixed_results))

        # 4. Concurrent approvals of different Cases (all review paths).
        distinct_review = REVIEW_SPECS
        seeds = await asyncio.gather(
            *[
                _create_case(
                    client,
                    session_id=f"mw-different-{index}",
                    action_type=action,
                    order_id=order,
                    message=message,
                )
                for index, (action, order, message) in enumerate(distinct_review)
            ]
        )
        different_ids = [case_id for ok, case_id, _ in seeds if ok and case_id]
        approvals = await asyncio.gather(
            *[_decide(client, case_id, "approve", f"different-{index}") for index, case_id in enumerate(different_ids)]
        )
        different_states = [_read_case(db, case_id) for case_id in different_ids]
        different = {
            "cases": len(different_ids),
            "unique_cases": len(set(different_ids)),
            "approvals": _codes(approvals),
            "states": different_states,
            "notifications": [_read_notifications(support_db, case_id) for case_id in different_ids],
        }
        different["ok"] = (
            len(different_ids) == 8
            and len(set(different_ids)) == 8
            and all(state["status"] == "executed" and state["events"] == 1 for state in different_states)
            and all(count == 1 for count in different["notifications"])
            and not any(code >= 500 for code, _ in approvals)
        )

        return {
            "concurrent_creation": creation,
            "idempotent_creation": idempotent,
            "same_case_approve_30": same,
            "mixed_approve_reject_30": mixed,
            "different_cases_approve_8": different,
        }


def _log_scan(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    locked = [line for line in text.splitlines() if "database is locked" in line]
    tracebacks = [line for line in text.splitlines() if "Traceback (most recent call last)" in line]
    errors = [line for line in text.splitlines() if "ERROR" in line or " 500 " in line]
    return {
        "database_locked": len(locked),
        "tracebacks": len(tracebacks),
        "error_lines": len(errors),
        "samples": (locked + tracebacks + errors)[:10],
    }


def run_drill(
    *,
    runtime: Path = DEFAULT_RUNTIME,
    port: int | None = None,
    workers: int = 2,
) -> dict[str, Any]:
    runtime = _build_runtime(runtime)
    port = port or _free_port()
    base = f"http://127.0.0.1:{port}"
    process, log_path = _start_server(runtime, port, workers)
    try:
        _wait_ready(base, process, log_path)
        scenarios = asyncio.run(
            _run_scenarios(base, runtime / "data" / "cases.db", runtime / "data" / "support_conversations.db")
        )
        log = _log_scan(log_path)
    finally:
        _stop_server(process)
    ok = all(section.get("ok") for section in scenarios.values()) and not any(
        log[key] for key in ("database_locked", "tracebacks", "error_lines")
    )
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workers": workers,
        "base": base,
        "runtime": str(runtime),
        "scenarios": scenarios,
        "log": log,
        "ok": ok,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Multi-Worker / SQLite Drill",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- workers: {report['workers']}",
        f"- runtime: `{report['runtime']}`",
        f"- result: **{'PASS' if report['ok'] else 'FAIL'}**",
        "",
        "| scenario | result | details |",
        "|---|---|---|",
    ]
    details = {
        "concurrent_creation": lambda s: f"{s['requests']} requests, {s['unique_cases']} unique cases",
        "idempotent_creation": lambda s: f"{s['requests']} requests, {s['unique_cases']} unique case",
        "same_case_approve_30": lambda s: f"{s['case']['status']}, events={s['case']['events']}, notifications={s['notifications']}",
        "mixed_approve_reject_30": lambda s: f"{s['case']['status']}, events={s['case']['events']}, notifications={s['notifications']}",
        "different_cases_approve_8": lambda s: f"{len(s['states'])} cases executed",
    }
    for name, section in report["scenarios"].items():
        lines.append(f"| {name} | {'PASS' if section.get('ok') else 'FAIL'} | {details[name](section)} |")
    log = report["log"]
    lines.extend(
        [
            "",
            "## Server log scan",
            "",
            f"- database locked: {log['database_locked']}",
            f"- tracebacks: {log['tracebacks']}",
            f"- error lines: {log['error_lines']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_drill(runtime=args.runtime, port=args.port or None, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "workers": report["workers"],
                "scenarios": {name: section.get("ok") for name, section in report["scenarios"].items()},
                "log": report["log"],
            },
            ensure_ascii=False,
        )
    )
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
