#!/usr/bin/env python3
"""Run the real LangGraph on 20 real OList order rows from a frozen fixture."""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import AgentGraph
from app.olist.service import OrderStatusView, SQLiteCaseService
from evaluation.policy_workflow_graph import build_actions, make_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "evaluation" / "policy_real_order_fixtures.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation" / "policy_real_order_report.json"


def load_fixtures(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_view(row: dict) -> OrderStatusView:
    return OrderStatusView(
        order_id=row["order_id"],
        status=row["status"],
        customer_state=row.get("customer_state") or "",
        purchase_timestamp=row.get("purchase_timestamp") or "",
        estimated_delivery_date=row.get("estimated_delivery_date") or "",
        delivered_customer_date=row.get("delivered_customer_date") or "",
        delay_days=row.get("delay_days"),
        payment_value=float(row.get("payment_value") or 0),
        review_score=row.get("review_score"),
        category_summary=row.get("category_summary") or "",
        refund_status=row.get("refund_status"),
        address_change_status=row.get("address_change_status"),
        invoice_status=row.get("invoice_status"),
    )


async def run_scenario(use_policy: bool, fixtures: list[dict]) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        case_service = SQLiteCaseService(Path(tmp) / "cases.db")
        views = {row["order_id"]: make_view(row) for row in fixtures}
        ids = list(views)
        actions = build_actions(views, case_service, use_policy)
        graph = AgentGraph(actions).build(InMemorySaver())
        for index, fixture in enumerate(fixtures):
            order_id = fixture["order_id"]
            state = make_state(
                f"real-policy-{use_policy}-{index}",
                order_id,
                ids,
                risk_level=str(fixture.get("user_risk_level") or ""),
                repeat_claim=bool(fixture.get("repeat_claim")),
                refund_failure=bool(fixture.get("refund_failure")),
            )
            result = await graph.ainvoke(
                state, {"configurable": {"thread_id": state["session_id"]}}
            )
            pending = result.get("pending_side_effect") or {}
            auto = not bool(pending.get("requires_confirmation"))
            rows.append(
                {
                    "order_id": order_id,
                    "group": fixture["fixture_group"],
                    "auto": auto,
                    "status": (result.get("policy_decision") or {}).get("decision")
                    if isinstance(result.get("policy_decision"), dict)
                    else None,
                    "case_id": result.get("case_id"),
                }
            )
    auto_count = sum(row["auto"] for row in rows)
    return {
        "use_policy": use_policy,
        "total": len(rows),
        "auto": auto_count,
        "review": len(rows) - auto_count,
        "auto_rate": round(auto_count / len(rows), 4) if rows else 0,
        "review_rate": round((len(rows) - auto_count) / len(rows), 4) if rows else 0,
        "rows": rows,
    }


async def main_async(fixtures_path: Path, output: Path) -> dict:
    fixtures = load_fixtures(fixtures_path)
    without = await run_scenario(False, fixtures)
    with_policy = await run_scenario(True, fixtures)
    policy_auto = sum(
        1
        for row in with_policy["rows"]
        if row["group"] in {"auto_low_delay", "control_auto"} and row["status"] == "auto"
    )
    existing_auto = with_policy["auto"] - policy_auto
    false_auto = sum(
        1
        for row in with_policy["rows"]
        if row["group"] not in {"auto_low_delay", "control_auto"} and row["status"] == "auto"
    )
    report = {
        "demo_only": True,
        "source": f"frozen real OList order rows ({len(fixtures)})",
        "fixtures": str(fixtures_path),
        "without_policy": without,
        "with_policy": with_policy,
        "review_reduction_pp": round((without["review_rate"] - with_policy["review_rate"]) * 100, 2),
        "policy_auto": policy_auto,
        "existing_auto": existing_auto,
        "false_auto": false_auto,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Real OList Order Policy Workflow Bench",
        "",
        f"> {len(fixtures)} frozen real OList order rows; real V2 LangGraph; not live conversations.",
        "- without policy: "
        f"auto {without['auto']} / review {without['review']} "
        f"({without['review_rate']:.2%} review)",
        "- with policy: "
        f"auto {with_policy['auto']} / review {with_policy['review']} "
        f"({with_policy['review_rate']:.2%} review)",
        f"- review reduction: {report['review_reduction_pp']:.2f} pp",
        f"- policy-driven auto: {policy_auto}",
        f"- pre-existing auto: {existing_auto}",
        f"- false auto: {false_auto}",
        "",
        "| order_id | group | auto | policy |",
        "|---|---|---|---|",
    ]
    for row in with_policy["rows"]:
        md.append(f"| {row['order_id']} | {row['group']} | {row['auto']} | {row['status']} |")
    output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = asyncio.run(main_async(args.fixtures, args.output))
    print(
        json.dumps(
            {
                "total": report["with_policy"]["total"],
                "without_review": report["without_policy"]["review_rate"],
                "with_review": report["with_policy"]["review_rate"],
                "reduction_pp": report["review_reduction_pp"],
                "policy_auto": report["policy_auto"],
                "existing_auto": report["existing_auto"],
                "false_auto": report["false_auto"],
            },
            ensure_ascii=False,
        )
    )
    if report["false_auto"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
