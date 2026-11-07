"""Build a compact Agent dataset and eval cases from the public Olist CSV files.

Raw data source:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

This script expects CSV mirrors under data/olist_raw/ and writes:
- data/olist_derived/agent_records.json
- data/olist_derived/order_facts_index.json
- data/olist_derived/category_risk_index.json
- data/olist_derived/eval_cases.jsonl
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "olist_raw"
OUT = ROOT / "data" / "olist_derived"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RAW / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    customers = {r["customer_id"]: r for r in read_csv("olist_customers_dataset.csv")}
    orders = read_csv("olist_orders_dataset.csv")
    items = read_csv("olist_order_items_dataset.csv")
    payments = read_csv("olist_order_payments_dataset.csv")
    reviews = read_csv("olist_order_reviews_dataset.csv")
    products = {r["product_id"]: r for r in read_csv("olist_products_dataset.csv")}
    sellers = {r["seller_id"]: r for r in read_csv("olist_sellers_dataset.csv")}
    translations = {
        r["product_category_name"]: r["product_category_name_english"]
        for r in read_csv("product_category_name_translation.csv")
    }

    items_by_order: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        items_by_order[item["order_id"]].append(item)

    payments_by_order: dict[str, list[dict[str, str]]] = defaultdict(list)
    for payment in payments:
        payments_by_order[payment["order_id"]].append(payment)

    reviews_by_order: dict[str, list[dict[str, str]]] = defaultdict(list)
    for review in reviews:
        reviews_by_order[review["order_id"]].append(review)

    enriched_orders: list[dict[str, object]] = []
    for order in orders:
        order_items = items_by_order.get(order["order_id"], [])
        if not order_items:
            continue

        customer = customers.get(order["customer_id"], {})
        product_summaries = []
        seller_ids = set()
        total_item_value = 0.0
        total_freight_value = 0.0
        for item in order_items:
            product = products.get(item["product_id"], {})
            seller = sellers.get(item["seller_id"], {})
            seller_ids.add(item["seller_id"])
            price = _to_float(item.get("price"))
            freight = _to_float(item.get("freight_value"))
            total_item_value += price
            total_freight_value += freight
            category_pt = product.get("product_category_name", "")
            product_summaries.append(
                {
                    "product_id": item["product_id"],
                    "category": translations.get(category_pt, category_pt or "unknown"),
                    "seller_id": item["seller_id"],
                    "seller_city": seller.get("seller_city", ""),
                    "seller_state": seller.get("seller_state", ""),
                    "price": price,
                    "freight_value": freight,
                    "shipping_limit_date": item.get("shipping_limit_date", ""),
                }
            )

        payment_value = sum(
            _to_float(p.get("payment_value")) for p in payments_by_order.get(order["order_id"], [])
        )
        review_scores = [
            int(r["review_score"])
            for r in reviews_by_order.get(order["order_id"], [])
            if r.get("review_score", "").isdigit()
        ]
        review_score = review_scores[0] if review_scores else None
        status = order["order_status"]
        delivered = order.get("order_delivered_customer_date", "")
        estimated = order.get("order_estimated_delivery_date", "")

        enriched_orders.append(
            {
                "order_id": order["order_id"],
                "customer_id": order["customer_id"],
                "customer_city": customer.get("customer_city", ""),
                "customer_state": customer.get("customer_state", ""),
                "status": status,
                "purchase_timestamp": order.get("order_purchase_timestamp", ""),
                "approved_at": order.get("order_approved_at", ""),
                "delivered_carrier_date": order.get("order_delivered_carrier_date", ""),
                "delivered_customer_date": delivered,
                "estimated_delivery_date": estimated,
                "delay_days": _delay_days(delivered, estimated),
                "item_count": len(order_items),
                "total_item_value": round(total_item_value, 2),
                "total_freight_value": round(total_freight_value, 2),
                "payment_value": round(payment_value, 2),
                "review_score": review_score,
                "seller_ids": sorted(seller_ids),
                "products": product_summaries,
            }
        )

    delivered_late = [o for o in enriched_orders if (o["delay_days"] or 0) > 0]
    low_review = [o for o in enriched_orders if o["review_score"] is not None and o["review_score"] <= 2]
    canceled = [o for o in enriched_orders if o["status"] == "canceled"]
    shipped = [o for o in enriched_orders if o["status"] == "shipped"]
    delivered = [o for o in enriched_orders if o["status"] == "delivered" and o["review_score"] is not None]
    category_counts = Counter(p["category"] for o in enriched_orders for p in o["products"])

    selected_orders = (
        _take(delivered_late, 120)
        + _take(low_review, 120)
        + _take(canceled, 80)
        + _take(shipped, 80)
    )
    selected_orders += _take(delivered, 200)
    unique: dict[str, dict[str, object]] = {str(o["order_id"]): o for o in selected_orders}

    dataset = {
        "metadata": {
            "name": "Olist marketplace support agent dataset",
            "source": "Olist Brazilian E-Commerce Public Dataset",
            "source_url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
            "raw_record_counts": {
                "orders": len(orders),
                "order_items": len(items),
                "customers": len(customers),
                "products": len(products),
                "sellers": len(sellers),
                "payments": len(payments),
                "reviews": len(reviews),
            },
            "derived_order_count": len(unique),
            "top_categories": category_counts.most_common(20),
        },
        "orders": list(unique.values()),
    }
    (OUT / "agent_records.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "order_facts_index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "source": dataset["metadata"]["source"],
                    "source_url": dataset["metadata"]["source_url"],
                    "order_count": len(enriched_orders),
                },
                "orders": {str(order["order_id"]): order for order in enriched_orders},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    category_risk_index = build_category_risk_index(enriched_orders)
    (OUT / "category_risk_index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "source": dataset["metadata"]["source"],
                    "source_url": dataset["metadata"]["source_url"],
                    "category_count": len(category_risk_index),
                    "basis": (
                        "Aggregated from full Olist orders/items/products/payments/"
                        "reviews/sellers/customers tables"
                    ),
                },
                "categories": category_risk_index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    eval_cases = build_eval_cases(list(unique.values()))
    with (OUT / "eval_cases.jsonl").open("w", encoding="utf-8") as f:
        for case in eval_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT / 'agent_records.json'} ({len(unique)} orders)")
    print(f"Wrote {OUT / 'order_facts_index.json'} ({len(enriched_orders)} orders)")
    print(f"Wrote {OUT / 'category_risk_index.json'} ({len(category_risk_index)} categories)")
    print(f"Wrote {OUT / 'eval_cases.jsonl'} ({len(eval_cases)} cases)")


def build_eval_cases(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for order in orders[:80]:
        order_id = str(order["order_id"])
        status = str(order["status"])
        cases.append(
            {
                "id": f"status-{order_id[:8]}",
                "task": "route",
                "message": f"帮我查一下订单 {order_id} 现在是什么状态",
                "expected_intent": "order_status",
                "expected": {"order_id": order_id, "status": status},
            }
        )

    late_orders = [o for o in orders if (o["delay_days"] or 0) > 0]
    for order in late_orders[:60]:
        order_id = str(order["order_id"])
        cases.append(
            {
                "id": f"delay-{order_id[:8]}",
                "task": "route",
                "message": f"订单 {order_id} 延迟了吗？帮我判断是否需要安抚客户",
                "expected_intent": "escalation",
                "expected": {"order_id": order_id, "delay_days": order["delay_days"]},
            }
        )

    low_reviews = [o for o in orders if o["review_score"] is not None and o["review_score"] <= 2]
    for order in low_reviews[:60]:
        order_id = str(order["order_id"])
        cases.append(
            {
                "id": f"review-{order_id[:8]}",
                "task": "route",
                "message": f"客户给订单 {order_id} 打了低分，生成一段客服跟进话术",
                "expected_intent": "escalation",
                "expected": {"order_id": order_id, "review_score": order["review_score"]},
            }
        )

    categories = []
    seen = set()
    for order in orders:
        for product in order["products"]:
            category = product["category"]
            if category not in seen and category != "unknown":
                seen.add(category)
                categories.append(category)
    for category in categories[:50]:
        cases.append(
            {
                "id": f"qa-{category}",
                "task": "route",
                "message": f"{category} 类目的订单主要有哪些物流和评价风险？",
                "expected_intent": "qa",
                "expected": {"category": category},
            }
        )
    return cases


def build_category_risk_index(orders: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for order in orders:
        seen_categories = {str(product["category"]) for product in order["products"]}
        for category in seen_categories:
            if category != "unknown":
                buckets[category].append(order)

    index: dict[str, dict[str, object]] = {}
    for category, rows in buckets.items():
        delay_values = [int(o["delay_days"]) for o in rows if o["delay_days"] is not None]
        review_scores = [int(o["review_score"]) for o in rows if o["review_score"] is not None]
        delayed = [o for o in rows if (o["delay_days"] or 0) > 0]
        low_review = [o for o in rows if o["review_score"] is not None and o["review_score"] <= 2]
        canceled = [o for o in rows if o["status"] == "canceled"]
        index[category] = {
            "category": category,
            "order_count": len(rows),
            "delayed_order_count": len(delayed),
            "low_review_order_count": len(low_review),
            "canceled_order_count": len(canceled),
            "delay_rate": round(len(delayed) / len(rows), 4),
            "low_review_rate": round(len(low_review) / len(rows), 4),
            "cancellation_rate": round(len(canceled) / len(rows), 4),
            "avg_delay_days": round(sum(delay_values) / len(delay_values), 2) if delay_values else None,
            "avg_review_score": round(sum(review_scores) / len(review_scores), 2) if review_scores else None,
            "avg_payment_value": round(
                sum(float(o["payment_value"]) for o in rows) / max(len(rows), 1),
                2,
            ),
            "sample_order_ids": [str(o["order_id"]) for o in rows[:5]],
            "sample_delayed_order_ids": [str(o["order_id"]) for o in delayed[:5]],
            "sample_low_review_order_ids": [str(o["order_id"]) for o in low_review[:5]],
        }
    return dict(sorted(index.items()))


def _to_float(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def _delay_days(delivered: str, estimated: str) -> int | None:
    if not delivered or not estimated:
        return None
    from datetime import datetime

    delivered_dt = datetime.fromisoformat(delivered)
    estimated_dt = datetime.fromisoformat(estimated)
    return (delivered_dt.date() - estimated_dt.date()).days


def _take(rows: list[dict[str, object]], n: int) -> list[dict[str, object]]:
    return rows[:n]


if __name__ == "__main__":
    main()
