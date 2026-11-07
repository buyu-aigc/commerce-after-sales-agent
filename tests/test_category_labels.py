"""Every Olist category has a customer-facing Chinese name, and one table feeds every surface."""

import re

from app.agent.actions import _customer_category_label
from app.agent.customer_order_agent import CATEGORY_LABELS as PRESENTER_LABELS
from app.olist.catalog import load_category_risk_index
from app.olist.category_labels import CATEGORY_LABELS, category_label

CJK = re.compile(r"[一-鿿]")


def test_every_category_in_the_data_has_a_chinese_name():
    categories = set(load_category_risk_index()["categories"]) | {"unknown"}
    missing = sorted(c for c in categories if not CJK.search(CATEGORY_LABELS.get(c, "")))
    assert not missing
    assert len(categories) == 74


def test_demo_category_names_are_unchanged():
    assert {
        c: CATEGORY_LABELS[c] for c in ("health_beauty", "bed_bath_table", "computers_accessories", "baby")
    } == {
        "health_beauty": "健康与美容用品",
        "bed_bath_table": "家居用品",
        "computers_accessories": "电脑配件",
        "baby": "母婴用品",
    }


def test_one_table_feeds_cards_planner_and_replies():
    assert _customer_category_label("sports_leisure") == "运动休闲用品"
    assert PRESENTER_LABELS is CATEGORY_LABELS


def test_multi_category_orders_and_unknown_codes():
    assert category_label("toys、baby") == "玩具、母婴用品"
    assert category_label("brand_new_code") == "brand new code"
    assert category_label("") == "商品"


def test_customer_orders_carry_the_display_name():
    import app.main as main

    orders = main._customer_context_payload("demo-customer")["orders"]
    assert orders and all(CJK.search(o["category_label"]) for o in orders)
