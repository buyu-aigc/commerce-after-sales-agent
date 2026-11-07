"""Resolve typed, conjunctive order constraints against identity-scoped facts.

The model extracts meaning. This module never reads free text to infer a
category, an action, or permission. Zero/multiple matches remain unresolved.
"""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrderConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal[
        "category", "status", "currency", "amount", "purchased_at", "delivered_at", "estimated_delivery_at"
    ]
    operator: Literal["eq", "gte", "lte"] = "eq"
    value: str = Field(min_length=1, max_length=120)
    evidence_quote: str = Field(min_length=1, max_length=1000)


class OrderReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraints: list[OrderConstraint] = Field(min_length=1, max_length=10)


def _value(field: str, value):
    if value is None or isinstance(value, bool):
        raise ValueError("missing_order_fact")
    if field == "amount":
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError("invalid_amount")
        return result
    if field.endswith("_at"):
        # Compare customer calendar dates; no fuzzy dates or timezone inference.
        raw = str(value)
        if len(raw) == 10:
            return date.fromisoformat(raw)
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    result = str(value).strip().casefold()
    if not result:
        raise ValueError("missing_order_fact")
    return result


def resolve_order_reference(
    reference: OrderReference, *, orders: list[dict], allowed: set[str], evidence: list[str]
) -> dict:
    """Return auditable candidates; never trust a model-selected index here."""
    normalized = []
    try:
        for condition in reference.constraints:
            if not any(condition.evidence_quote in span for span in evidence):
                return {"status": "unverifiable_reference_evidence", "candidate_ids": []}
            if condition.field in {"category", "status", "currency"} and condition.operator != "eq":
                return {"status": "invalid_reference_operator", "candidate_ids": []}
            normalized.append((condition.field, condition.operator, _value(condition.field, condition.value)))
    except (ValueError, InvalidOperation, TypeError):
        return {"status": "invalid_reference_value", "candidate_ids": []}
    candidates = set()
    for order in orders:
        if not order.get("order_id") or order["order_id"] not in allowed:
            continue
        matched = True
        for field, operator, expected in normalized:
            try:
                actual = _value(field, order.get(field))
                matched = {"eq": actual == expected, "gte": actual >= expected, "lte": actual <= expected}[
                    operator
                ]
            except (ValueError, InvalidOperation, TypeError):
                matched = False
            if not matched:
                break
        if matched:
            candidates.add(order["order_id"])
    ids = sorted(candidates)
    return {
        "status": "unique" if len(ids) == 1 else "ambiguous_order" if ids else "unknown_order",
        "candidate_ids": ids,
    }
