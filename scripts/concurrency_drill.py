#!/usr/bin/env python3
"""Run system-level concurrent review drills against an isolated V2 instance."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path

import httpx

TOKEN = "local-review-demo"
ORDER = "203096f03d82e0dffbc41ebc2e2bcfb7"


def read_case(db: Path, case_id: str) -> dict:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        status = dict(row)["status"] if row else None
        events = conn.execute(
            "SELECT COUNT(*) FROM order_events WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
    return {"status": status, "events": events}


def read_notifications(db: Path, case_id: str) -> int:
    with sqlite3.connect(db) as conn:
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE case_id = ? AND sender = 'staff'",
                (case_id,),
            ).fetchone()[0]
        except sqlite3.OperationalError:
            return -1


async def seed_case(
    client: httpx.AsyncClient,
    nonce: str,
    *,
    order: str = ORDER,
    kind: str = "refund",
) -> str:
    messages = {
        "refund": f"订单 {order} 晚到了很多天，我要申请退款。并发测试 {nonce}",
        "invoice": f"我想为订单 {order} 申请发票。并发测试 {nonce}",
        "complaint": f"我要投诉订单 {order} 的配送服务。并发测试 {nonce}",
    }
    response = await client.post(
        "/customer/chat",
        json={"message": messages[kind], "session_id": f"concurrency-{nonce}"},
    )
    response.raise_for_status()
    case_id = response.json().get("case_id")
    if not case_id:
        raise RuntimeError(f"seed failed: {response.text[:500]}")
    return case_id


async def decide(client: httpx.AsyncClient, case_id: str, decision: str, reviewer: str) -> dict:
    response = await client.post(
        f"/review/cases/{case_id}/{decision}",
        headers={"X-Review-Token": TOKEN},
        json={
            "reviewer_id": reviewer,
            "role": "after_sales_operator",
            "auth_scopes": ["after_sales:write"],
        },
    )
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text[:200]}
    return {"status_code": response.status_code, "answer": body.get("answer"), "body": body}


def summarize(results: list[dict]) -> dict:
    codes: dict[str, int] = {}
    for result in results:
        key = str(result["status_code"])
        codes[key] = codes.get(key, 0) + 1
    return {
        "codes": codes,
        "answers": sorted({str(result.get("answer")) for result in results}),
        "server_errors": [r for r in results if r["status_code"] >= 500],
    }


async def main(base: str, db: Path, support_db: Path) -> dict:
    async with httpx.AsyncClient(base_url=base, timeout=60) as client:
        case_a = await seed_case(client, "same-approve")
        same_approve = await asyncio.gather(
            *[decide(client, case_a, "approve", f"reviewer-{i}") for i in range(20)]
        )
        same_state = read_case(db, case_a)

        case_b = await seed_case(client, "approve-reject", kind="complaint")
        mixed = await asyncio.gather(
            *[
                decide(client, case_b, "approve" if i % 2 == 0 else "reject", f"mixed-{i}")
                for i in range(20)
            ]
        )
        mixed_state = read_case(db, case_b)

        case_c = await seed_case(
            client,
            "different-a",
            order="8aec3a066f732dd927ec8fef1752415b",
            kind="invoice",
        )
        case_d = await seed_case(
            client,
            "different-b",
            order="360787554f41824600bdcba9687cd4f0",
            kind="invoice",
        )
        different = await asyncio.gather(
            decide(client, case_c, "approve", "different-a"),
            decide(client, case_d, "approve", "different-b"),
        )

        report = {
            "same_case_approve_20": {
                **summarize(same_approve),
                **same_state,
                "notifications": read_notifications(support_db, case_a),
            },
            "mixed_approve_reject_20": {
                **summarize(mixed),
                **mixed_state,
                "notifications": read_notifications(support_db, case_b),
            },
            "different_cases_approve": {
                "results": summarize(different),
                "case_c": {**read_case(db, case_c), "notifications": read_notifications(support_db, case_c)},
                "case_d": {**read_case(db, case_d), "notifications": read_notifications(support_db, case_d)},
            },
        }
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--support-db", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(main(args.base, args.db, args.support_db))
    print(json.dumps(report, ensure_ascii=False, indent=2))
