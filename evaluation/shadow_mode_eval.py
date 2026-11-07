#!/usr/bin/env python3
"""Replay shadow suggestions against a reviewer-label file.

The shipped labels are synthetic (``synthetic_rule_for_plumbing_only``): they
exercise the comparison pipeline, not a production accuracy claim.  Replace the
labels file with real reviewer decisions before drawing product conclusions.

This script also proves the shadow path is non-executing: it never invokes the
graph and asserts the business Case store and order-event store stay empty.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEMO_MODE", "1")

from app.after_sales.complexity import assess_request  # noqa: E402
from app.after_sales.shadow import ShadowService, SQLiteShadowStore  # noqa: E402
from app.olist.service import SQLiteCaseService  # noqa: E402
from app.policy.engine import PolicyEngine  # noqa: E402
from app.policy.guard import PolicyGuard  # noqa: E402
from evaluation.complex_partial_auto_eval import (  # noqa: E402
    SyntheticOlist,
    load_cases,
    make_view,
    order_id_for,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "complex_request_cases.jsonl"
DEFAULT_LABELS = ROOT / "evaluation" / "shadow_reviewer_labels.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "shadow_mode_report.json"


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(item["case_id"]): item
        for item in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }


def label_for(
    labels: dict[str, Any], action_type: str, order_id: str, aliases: dict[str, str]
) -> dict[str, Any] | None:
    for decision in labels.get("decisions", []):
        if decision.get("action_type") != action_type:
            continue
        alias = decision.get("alias")
        expected_order = aliases.get(alias, "") if alias else ""
        if expected_order and expected_order == order_id:
            return decision
    return None


def main_async(cases_path: Path, labels_path: Path, output: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    labels = load_labels(labels_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        case_service = SQLiteCaseService(tmp_path / "cases.db")
        shadow = ShadowService(SQLiteShadowStore(tmp_path / "shadow.db"), enabled=True, sample_rate=1.0)
        all_views = {}
        case_rows = []
        case_sessions: list[str] = []
        for case in cases:
            case_id = str(case["id"])
            aliases = {
                spec["alias"]: order_id_for(case_id, spec["alias"])
                for spec in case["orders"]
                if spec.get("alias")
            }
            views = {
                aliases[spec["alias"]]: make_view(aliases[spec["alias"]], spec)
                for spec in case["orders"]
                if spec.get("alias")
            }
            all_views.update(views)
            guard = PolicyGuard(PolicyEngine(), SyntheticOlist(views), case_service)
            message = str(case["message"])
            for alias, order_id in aliases.items():
                message = message.replace("{" + alias + "}", order_id)
            analysis = assess_request(message, order_lookup=views.get)
            request_id = f"shadow-replay-{case_id}"
            case_sessions.append(request_id)
            suggestions = shadow.build_suggestions(
                request_id=request_id,
                session_id=request_id,
                analysis=analysis,
                order_lookup=views.get,
                policy_guard=guard,
            )
            shadow.store.record_suggestions(suggestions)
            rows = []
            for suggestion in suggestions:
                dummy_case = f"{request_id}-{suggestion.sub_task_index}"
                shadow.link_case(
                    case_id=dummy_case,
                    request_id=request_id,
                    sub_task_index=suggestion.sub_task_index,
                    action_type=suggestion.action_type,
                )
                decision = label_for(
                    labels.get(case_id, {}),
                    suggestion.action_type,
                    suggestion.order_id,
                    aliases,
                )
                if decision:
                    shadow.record_human_decision(
                        case_id=dummy_case,
                        decision=str(decision["human_decision"]),
                        reviewer_id="synthetic-reviewer",
                        reason=str(decision.get("reason") or ""),
                    )
                rows.append(
                    {
                        "suggestion": suggestion.to_dict(),
                        "order_alias": next(
                            (alias for alias, order_id in aliases.items() if order_id == suggestion.order_id),
                            None,
                        ),
                        "label": decision,
                    }
                )
            case_rows.append({"id": case_id, "category": case["category"], "suggestions": rows})
        metrics = shadow.metrics()
        metrics["gates"]["labels_synthetic"] = True
        metrics["gates"]["recommendation"] = "keep_shadow"
        business_cases = [
            record for session_id in case_sessions for record in case_service.list_by_session(session_id)
        ]
        order_events = sum(case_service.count_order_events(order_id) for order_id in all_views)

    summary = metrics["summary"]
    report = {
        "demo_only": True,
        "label_source": "synthetic_rule_for_plumbing_only",
        "scope": (
            "24 controlled complex requests replayed through the shadow suggestion path; "
            "no graph execution and no business writes. Synthetic labels are for plumbing "
            "and dashboard verification, not production accuracy."
        ),
        "summary": summary,
        "gates": metrics["gates"],
        "by_risk": metrics["by_risk"],
        "by_action": metrics["by_action"],
        "by_complexity": metrics["by_complexity"],
        "execution_check": {
            "business_cases_created": len(business_cases),
            "order_events_created": order_events,
        },
        "cases": case_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Shadow Mode Replay",
        "",
        "> 24 controlled complex requests through the shadow suggestion path. No graph execution,",
        "> no order writes. **Labels are synthetic (`synthetic_rule_for_plumbing_only`) and are",
        "> not a production accuracy claim.**",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| shadow suggestions | {summary['suggestions']} |",
        f"| linked cases | {summary['linked_cases']} |",
        f"| decided by human labels | {summary['decided']} |",
        f"| auto suggestions | {summary['auto_suggestions']} |",
        f"| auto approved | {summary['auto_approved']} |",
        f"| **false auto** | **{summary['false_auto']}** |",
        f"| business approval rate (proxy) | {summary['business_approval_rate'] or 0:.2%} |",
        f"| automation eligibility labels | {summary['automation_labeled']} |",
        f"| review suggestions | {summary['review_suggestions']} |",
        f"| review suggestions approved by human (business decision) | {summary['review_approved']} |",
        f"| missed auto (requires eligibility labels) | {summary['missed_auto']} |",
        f"| **high-risk auto suggestions** | **{summary['high_risk_auto_suggestions']}** |",
        f"| routing agreement | {summary['routing_agreement']:.2%} |",
        f"| business cases created | {len(business_cases)} |",
        f"| order events created | {order_events} |",
        "",
        "## Gates",
        "",
        f"- high-risk auto zero: {metrics['gates']['high_risk_auto_zero']}",
        f"- false auto zero: {metrics['gates']['false_auto_zero']}",
        f"- minimum decided samples: {metrics['gates']['minimum_decided_samples']}",
        f"- recommendation: `{metrics['gates']['recommendation']}`",
        "",
        "## By risk",
        "",
        "| risk | suggestions | decided | auto | false auto | missed auto | high-risk auto |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["by_risk"]:
        md.append(
            f"| {row['key']} | {row['suggestions']} | {row['decided']} | "
            f"{row['auto_suggestions']} | {row['false_auto']} | {row['missed_auto']} | "
            f"{row['high_risk_auto']} |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This replay proves the shadow pipeline is non-executing and can compare suggestions",
            "  with human labels.",
            "- It does **not** measure live reviewer agreement because the labels are synthetic.",
            "- Replace `evaluation/shadow_reviewer_labels.jsonl` with real reviewer decisions before",
            "  using the gates to expand automation.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = main_async(args.cases, args.labels, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["gates"], ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if (
        report["summary"]["high_risk_auto_suggestions"]
        or report["summary"]["false_auto"]
        or report["execution_check"]["business_cases_created"]
        or report["execution_check"]["order_events_created"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
