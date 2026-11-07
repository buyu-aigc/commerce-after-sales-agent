"""Extract explicitly supplied address evidence, without inferring missing values."""

import re


def address_fields(text: str) -> dict[str, str]:
    fields = {}
    recipient = re.search(r"(?:收件人|联系人|recipient)\s*[:：]?\s*([^，,；;。\n]+)", text, re.I)
    phone = re.search(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", text)
    destination = re.search(
        r"(?:新地址|收货地址|改到|送到|寄到|(?:address|地址)\s*[:：])\s*[:：]?\s*([^，,；;。\n]{6,})",
        text,
        re.I,
    )
    if recipient:
        fields["recipient"] = recipient[1].strip()
    if phone:
        fields["phone"] = phone[0]
    if destination:
        fields["address"] = destination[1].strip()
    return fields


def missing_address_fields(text: str) -> list[str]:
    fields = address_fields(text)
    return [
        label
        for key, label in (("recipient", "收件人"), ("phone", "联系电话"), ("address", "详细新地址"))
        if not fields.get(key)
    ]
