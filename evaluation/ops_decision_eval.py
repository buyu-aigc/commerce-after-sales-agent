from __future__ import annotations

from app.olist.service import OlistService, format_after_sales_report


def main() -> None:
    service = OlistService()
    report = service.after_sales_priority_report("生成审核台优先处理队列，列出最需要人工跟进的类目和订单")
    categories = list(report["high_risk_categories"])
    orders = list(report["priority_orders"])

    checks = {
        "has_top_categories": len(categories) >= 5,
        "has_top_orders": len(orders) >= 8,
        "category_scores_sorted": _is_sorted_desc([float(item["risk_score"]) for item in categories]),
        "order_scores_sorted": _is_sorted_desc([float(item["priority_score"]) for item in orders]),
        "category_actions_present": all(item.get("recommended_action") for item in categories),
        "order_reasons_present": all(item.get("reasons") for item in orders),
        "read_only_hitl_boundary": any("HITL" in str(rule) for rule in report.get("decision_rules", [])),
    }
    passed = sum(int(ok) for ok in checks.values())
    total = len(checks)

    print("Ops decision eval")
    print(f"checks,{passed}/{total},{passed / total:.2%}")
    for name, ok in checks.items():
        print(f"{name},{'pass' if ok else 'fail'}")
    print("")
    print(format_after_sales_report(report))


def _is_sorted_desc(values: list[float]) -> bool:
    return all(left >= right for left, right in zip(values, values[1:]))


if __name__ == "__main__":
    main()
