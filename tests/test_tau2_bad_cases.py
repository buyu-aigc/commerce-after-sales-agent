from benchmark_adapters.tau2_confirmation_guard import (
    CONFIRMATION_SCOPE_CLARIFICATION,
    is_broad_confirmation,
    needs_item_scope_clarification,
)
from benchmark_adapters.tau2_variant_guard import repair_same_size_variant_selection, repair_variant_selection


def test_tau2_task6_broad_multi_item_confirmation_needs_clarification() -> None:
    previous_assistant = (
        "I can exchange the water bottle for the larger size and the desk lamp "
        "for a less bright battery-powered version. Please confirm both items "
        "before I proceed."
    )

    assert needs_item_scope_clarification(
        "Yes, I confirm both exchanges.",
        previous_assistant,
    )
    assert "exact item names" in CONFIRMATION_SCOPE_CLARIFICATION


def test_scoped_confirmation_is_not_blocked() -> None:
    previous_assistant = (
        "Please list exactly which item names you want to proceed with for the exchange."
    )

    assert not needs_item_scope_clarification("yes, desk lamp only", previous_assistant)
    assert not is_broad_confirmation("yes, desk lamp only")


def test_plain_yes_to_single_item_context_is_not_blocked() -> None:
    previous_assistant = "Please confirm that you want to exchange the desk lamp."

    assert not needs_item_scope_clarification("yes", previous_assistant)


def test_tau2_task20_same_size_footwear_variant_is_repaired() -> None:
    class ToolCall:
        name = "modify_pending_order_items"
        arguments = {
            "order_id": "#W9911714",
            "item_ids": ["9791469541"],
            "new_item_ids": ["4153505238"],
            "payment_method_id": "gift_card_4332117",
        }

    class Assistant:
        tool_calls = [ToolCall()]

    history = [
        {
            "role": "user",
            "content": (
                "Upgrade everything to the most expensive variants, but make sure "
                "the new shoe is still the same size."
            ),
        },
        {
            "role": "tool",
            "content": """{
                "name": "Running Shoes",
                "product_id": "6938111410",
                "variants": {
                    "4153505238": {
                        "item_id": "4153505238",
                        "options": {"size": "8"},
                        "available": true,
                        "price": 158.67
                    },
                    "9791469541": {
                        "item_id": "9791469541",
                        "options": {"size": "9"},
                        "available": true,
                        "price": 147.05
                    },
                    "4107812777": {
                        "item_id": "4107812777",
                        "options": {"size": "9"},
                        "available": true,
                        "price": 155.33
                    }
                }
            }""",
        },
    ]

    assert repair_same_size_variant_selection(Assistant(), history)
    assert ToolCall.arguments["new_item_ids"] == ["4107812777"]


def test_explicit_footwear_size_change_is_not_repaired() -> None:
    class ToolCall:
        name = "modify_pending_order_items"
        arguments = {
            "item_ids": ["9791469541"],
            "new_item_ids": ["4153505238"],
        }

    class Assistant:
        tool_calls = [ToolCall()]

    history = [
        {"role": "user", "content": "Please change my running shoes to size 8."},
        {
            "role": "tool",
            "content": """{
                "name": "Running Shoes",
                "product_id": "6938111410",
                "variants": {
                    "4153505238": {
                        "item_id": "4153505238",
                        "options": {"size": "8"},
                        "available": true,
                        "price": 158.67
                    },
                    "9791469541": {
                        "item_id": "9791469541",
                        "options": {"size": "9"},
                        "available": true,
                        "price": 147.05
                    },
                    "4107812777": {
                        "item_id": "4107812777",
                        "options": {"size": "9"},
                        "available": true,
                        "price": 155.33
                    }
                }
            }""",
        },
    ]

    assert not repair_same_size_variant_selection(Assistant(), history)
    assert ToolCall.arguments["new_item_ids"] == ["4153505238"]


def test_assistant_variant_description_does_not_count_as_user_size_change() -> None:
    class ToolCall:
        name = "modify_pending_order_items"
        arguments = {
            "item_ids": ["9791469541"],
            "new_item_ids": ["4153505238"],
        }

    class Assistant:
        tool_calls = [ToolCall()]

    history = [
        {"role": "user", "content": "Please upgrade my items to the most expensive variants."},
        {
            "role": "assistant",
            "content": "The most expensive running shoes option is item 4153505238, size 8.",
        },
        {
            "role": "tool",
            "content": """{
                "name": "Running Shoes",
                "product_id": "6938111410",
                "variants": {
                    "4153505238": {
                        "item_id": "4153505238",
                        "options": {"size": "8"},
                        "available": true,
                        "price": 158.67
                    },
                    "9791469541": {
                        "item_id": "9791469541",
                        "options": {"size": "9"},
                        "available": true,
                        "price": 147.05
                    },
                    "4107812777": {
                        "item_id": "4107812777",
                        "options": {"size": "9"},
                        "available": true,
                        "price": 155.33
                    }
                }
            }""",
        },
    ]

    assert repair_same_size_variant_selection(Assistant(), history)
    assert ToolCall.arguments["new_item_ids"] == ["4107812777"]


def test_tau2_task0_keyboard_no_backlight_fallback_is_repaired() -> None:
    class ToolCall:
        name = "exchange_delivered_order_items"
        arguments = {
            "order_id": "#W2378156",
            "item_ids": ["1151293680", "4983901480"],
            "new_item_ids": ["6342039236", "7747408585"],
            "payment_method_id": "credit_card_9513926",
        }

    class Assistant:
        tool_calls = [ToolCall()]

    history = [
        {
            "role": "user",
            "content": (
                "I want the mechanical keyboard with clicky switches, RGB backlight, "
                "full size. If that is not available, I will go for no backlight."
            ),
        },
        {
            "role": "tool",
            "content": """{
                "order_id": "#W2378156",
                "items": [
                    {
                        "name": "Mechanical Keyboard",
                        "product_id": "1656367028",
                        "item_id": "1151293680",
                        "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"}
                    },
                    {
                        "name": "Smart Thermostat",
                        "product_id": "4896585277",
                        "item_id": "4983901480",
                        "options": {"compatibility": "Apple HomeKit", "color": "black"}
                    }
                ]
            }""",
        },
        {
            "role": "tool",
            "content": """{
                "name": "Mechanical Keyboard",
                "product_id": "1656367028",
                "variants": {
                    "7706410293": {
                        "item_id": "7706410293",
                        "options": {"switch type": "clicky", "backlight": "none", "size": "full size"},
                        "available": true,
                        "price": 269.16
                    },
                    "6342039236": {
                        "item_id": "6342039236",
                        "options": {"switch type": "clicky", "backlight": "white", "size": "full size"},
                        "available": true,
                        "price": 244.91
                    },
                    "2299424241": {
                        "item_id": "2299424241",
                        "options": {"switch type": "clicky", "backlight": "RGB", "size": "80%"},
                        "available": true,
                        "price": 237.48
                    }
                }
            }""",
        },
    ]

    assert repair_variant_selection(Assistant(), history)
    assert ToolCall.arguments["new_item_ids"] == ["7706410293", "7747408585"]
