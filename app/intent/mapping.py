from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentMapping:
    intent: str
    route_intent: str
    needs_order_id: bool
    side_effect_risk: str


INTENT_TO_ROUTE: dict[str, IntentMapping] = {
    "track_order": IntentMapping("track_order", "order_status", True, "read_only"),
    "cancel_order": IntentMapping("cancel_order", "escalation", True, "side_effect"),
    "change_order": IntentMapping("change_order", "escalation", True, "side_effect"),
    "get_refund": IntentMapping("get_refund", "escalation", True, "side_effect"),
    "track_refund": IntentMapping("track_refund", "order_status", True, "read_only"),
    "complaint": IntentMapping("complaint", "escalation", False, "side_effect"),
    "contact_human_agent": IntentMapping("contact_human_agent", "escalation", False, "side_effect"),
    "check_refund_policy": IntentMapping("check_refund_policy", "policy", False, "read_only"),
    "check_cancellation_fee": IntentMapping("check_cancellation_fee", "policy", False, "read_only"),
    "delivery_period": IntentMapping("delivery_period", "policy", False, "read_only"),
    "delivery_options": IntentMapping("delivery_options", "policy", False, "read_only"),
    "check_payment_methods": IntentMapping("check_payment_methods", "policy", False, "read_only"),
    "payment_issue": IntentMapping("payment_issue", "escalation", False, "side_effect"),
    "get_invoice": IntentMapping("get_invoice", "policy", False, "read_only"),
    "check_invoice": IntentMapping("check_invoice", "policy", False, "read_only"),
    "place_order": IntentMapping("place_order", "policy", False, "read_only"),
    "review": IntentMapping("review", "escalation", False, "side_effect"),
    "change_shipping_address": IntentMapping("change_shipping_address", "escalation", True, "side_effect"),
    "set_up_shipping_address": IntentMapping("set_up_shipping_address", "policy", False, "read_only"),
    "create_account": IntentMapping("create_account", "policy", False, "read_only"),
    "delete_account": IntentMapping("delete_account", "policy", False, "side_effect"),
    "edit_account": IntentMapping("edit_account", "policy", False, "side_effect"),
    "switch_account": IntentMapping("switch_account", "policy", False, "read_only"),
    "recover_password": IntentMapping("recover_password", "policy", False, "read_only"),
    "registration_problems": IntentMapping("registration_problems", "policy", False, "read_only"),
    "newsletter_subscription": IntentMapping("newsletter_subscription", "policy", False, "side_effect"),
    "contact_customer_service": IntentMapping("contact_customer_service", "policy", False, "read_only"),
}


def map_intent(intent: str) -> IntentMapping:
    return INTENT_TO_ROUTE.get(intent, IntentMapping(intent, "policy", False, "unknown"))
