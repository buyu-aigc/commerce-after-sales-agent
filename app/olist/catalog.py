"""Compact Olist dataset loader."""

import json
import os
from functools import lru_cache
from pathlib import Path

_DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "olist_derived" / "agent_records.json"
_DEFAULT_ORDER_FACTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "olist_derived" / "order_facts_index.json"
)
_DEFAULT_CATEGORY_RISK_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "olist_derived" / "category_risk_index.json"
)


def dataset_path() -> Path:
    override = os.environ.get("OLIST_AGENT_DATASET_PATH")
    return Path(override) if override else _DEFAULT_DATASET_PATH


def order_facts_path() -> Path:
    override = os.environ.get("OLIST_ORDER_FACTS_PATH")
    return Path(override) if override else _DEFAULT_ORDER_FACTS_PATH


def category_risk_path() -> Path:
    override = os.environ.get("OLIST_CATEGORY_RISK_PATH")
    return Path(override) if override else _DEFAULT_CATEGORY_RISK_PATH


@lru_cache(maxsize=1)
def load_dataset() -> dict:
    with dataset_path().open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_order_facts_index() -> dict:
    path = order_facts_path()
    if not path.exists():
        return {"metadata": {}, "orders": orders_by_id()}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_category_risk_index() -> dict:
    path = category_risk_path()
    if not path.exists():
        return {"metadata": {}, "categories": {}}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_orders() -> list[dict]:
    return list(load_dataset()["orders"])


def orders_by_id() -> dict[str, dict]:
    return {o["order_id"]: o for o in load_orders()}


def full_orders_by_id() -> dict[str, dict]:
    return dict(load_order_facts_index()["orders"])


def category_risks_by_name() -> dict[str, dict]:
    return dict(load_category_risk_index()["categories"])
