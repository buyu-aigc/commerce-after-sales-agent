"""Preflight repair for tau2 retail variant-selection tool calls."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

WRITE_ITEM_TOOLS = {"modify_pending_order_items", "exchange_delivered_order_items"}

_SAME_SIZE_RE = re.compile(
    r"\b(same size|size must (?:stay|remain|be) the same|keep the same size|"
    r"only care about the size|same shoe size)\b",
    re.IGNORECASE,
)
_EXPLICIT_SIZE_CHANGE_RE = re.compile(
    r"\b(change|switch|modify|exchange|upgrade).{0,80}\bshoe(?:s)?\b.{0,80}\b(size\s*\d+|larger size|"
    r"smaller size|different size)\b|\bshoe(?:s)?\b.{0,80}\b(size\s*\d+|larger size|"
    r"smaller size|different size)\b",
    re.IGNORECASE | re.DOTALL,
)
_FOOTWEAR_RE = re.compile(r"\b(shoe|shoes|sneaker|sneakers|boot|boots)\b", re.IGNORECASE)
_NO_BACKLIGHT_KEYBOARD_RE = re.compile(
    r"\b(clicky|switch(?:es)?)\b.{0,180}\b(no backlight|without backlight)\b|"
    r"\b(no backlight|without backlight)\b.{0,180}\b(clicky|switch(?:es)?)\b",
    re.IGNORECASE | re.DOTALL,
)
_KEYBOARD_RE = re.compile(r"\bkeyboard\b", re.IGNORECASE)


def repair_variant_selection(assistant_message: Any, history: Iterable[Any]) -> bool:
    """Apply deterministic preflight repairs to outgoing variant write calls."""

    repaired = repair_same_size_variant_selection(assistant_message, history)
    repaired = _repair_keyboard_no_backlight_fallback(assistant_message, history) or repaired
    return repaired


def repair_same_size_variant_selection(assistant_message: Any, history: Iterable[Any]) -> bool:
    """Repair footwear variant choices when a same-size constraint is present.

    Returns True when at least one outgoing tool-call argument was changed.
    """

    user_history_text = _history_text(history, role="user")
    if _EXPLICIT_SIZE_CHANGE_RE.search(user_history_text) and not _SAME_SIZE_RE.search(user_history_text):
        return False

    products, item_to_product_id = _collect_product_catalog(history)
    if not products:
        return False

    repaired = False
    for tool_call in getattr(assistant_message, "tool_calls", None) or []:
        if getattr(tool_call, "name", None) not in WRITE_ITEM_TOOLS:
            continue
        arguments = getattr(tool_call, "arguments", None)
        if not isinstance(arguments, dict):
            continue

        item_ids = arguments.get("item_ids")
        new_item_ids = arguments.get("new_item_ids")
        if not isinstance(item_ids, list) or not isinstance(new_item_ids, list):
            continue

        for index, old_item_id in enumerate(item_ids):
            if index >= len(new_item_ids):
                continue
            replacement = _same_size_replacement(
                str(old_item_id),
                str(new_item_ids[index]),
                products,
                item_to_product_id,
            )
            if replacement and replacement != new_item_ids[index]:
                new_item_ids[index] = replacement
                repaired = True

    return repaired


def _repair_keyboard_no_backlight_fallback(assistant_message: Any, history: Iterable[Any]) -> bool:
    user_history_text = _history_text(history, role="user")
    if not _NO_BACKLIGHT_KEYBOARD_RE.search(user_history_text):
        return False

    products, item_to_product_id = _collect_product_catalog(history)
    if not products:
        return False

    repaired = False
    for tool_call in getattr(assistant_message, "tool_calls", None) or []:
        if getattr(tool_call, "name", None) not in WRITE_ITEM_TOOLS:
            continue
        arguments = getattr(tool_call, "arguments", None)
        if not isinstance(arguments, dict):
            continue
        item_ids = arguments.get("item_ids")
        new_item_ids = arguments.get("new_item_ids")
        if not isinstance(item_ids, list) or not isinstance(new_item_ids, list):
            continue

        for index, old_item_id in enumerate(item_ids):
            if index >= len(new_item_ids):
                continue
            replacement = _keyboard_no_backlight_replacement(
                str(old_item_id),
                products,
                item_to_product_id,
            )
            if replacement and replacement != new_item_ids[index]:
                new_item_ids[index] = replacement
                repaired = True
    return repaired


def _history_text(history: Iterable[Any], role: str | None = None) -> str:
    chunks = []
    for message in history:
        if role is not None and _message_role(message) != role:
            continue
        chunks.append(str(_message_content(message) or ""))
    return "\n".join(chunks)


def _same_size_replacement(
    old_item_id: str,
    new_item_id: str,
    products: dict[str, dict[str, Any]],
    item_to_product_id: dict[str, str],
) -> str | None:
    product_id = item_to_product_id.get(old_item_id) or item_to_product_id.get(new_item_id)
    if not product_id:
        return None
    product = products.get(product_id)
    if not product or not _FOOTWEAR_RE.search(str(product.get("name", ""))):
        return None

    variants = product.get("variants") or {}
    old_variant = variants.get(old_item_id)
    new_variant = variants.get(new_item_id)
    if not old_variant or not new_variant:
        return None

    old_size = str((old_variant.get("options") or {}).get("size", ""))
    new_size = str((new_variant.get("options") or {}).get("size", ""))
    if not old_size or old_size == new_size:
        return None

    candidates = [
        variant
        for variant in variants.values()
        if variant.get("available")
        and str((variant.get("options") or {}).get("size", "")) == old_size
        and str(variant.get("item_id")) != old_item_id
    ]
    if not candidates:
        return None

    best = max(candidates, key=lambda variant: float(variant.get("price") or 0))
    return str(best["item_id"])


def _keyboard_no_backlight_replacement(
    old_item_id: str,
    products: dict[str, dict[str, Any]],
    item_to_product_id: dict[str, str],
) -> str | None:
    product_id = item_to_product_id.get(old_item_id)
    if not product_id:
        return None
    product = products.get(product_id)
    if not product or not _KEYBOARD_RE.search(str(product.get("name", ""))):
        return None

    variants = product.get("variants") or {}
    old_variant = variants.get(old_item_id)
    old_size = str(((old_variant or {}).get("options") or {}).get("size", ""))
    candidates = []
    for variant in variants.values():
        options = variant.get("options") or {}
        if not variant.get("available"):
            continue
        if str(options.get("switch type", "")).lower() != "clicky":
            continue
        if str(options.get("backlight", "")).lower() != "none":
            continue
        if old_size and str(options.get("size", "")) != old_size:
            continue
        if str(variant.get("item_id")) == old_item_id:
            continue
        candidates.append(variant)
    if not candidates:
        return None
    best = max(candidates, key=lambda variant: float(variant.get("price") or 0))
    return str(best["item_id"])


def _collect_product_catalog(history: Iterable[Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    products: dict[str, dict[str, Any]] = {}
    item_to_product_id: dict[str, str] = {}

    for message in history:
        payload = _json_payload(_message_content(message))
        if not isinstance(payload, dict):
            continue
        _collect_from_product_payload(payload, products, item_to_product_id)
        for order_item in payload.get("items") or []:
            product_id = str(order_item.get("product_id") or "")
            item_id = str(order_item.get("item_id") or "")
            if product_id and item_id:
                item_to_product_id[item_id] = product_id

    return products, item_to_product_id


def _collect_from_product_payload(
    payload: dict[str, Any],
    products: dict[str, dict[str, Any]],
    item_to_product_id: dict[str, str],
) -> None:
    product_id = str(payload.get("product_id") or "")
    variants = payload.get("variants")
    if not product_id or not isinstance(variants, dict):
        return

    products[product_id] = payload
    for item_id in variants:
        item_to_product_id[str(item_id)] = product_id


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _json_payload(content: Any) -> Any:
    if not isinstance(content, str) or not content.strip().startswith("{"):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None
