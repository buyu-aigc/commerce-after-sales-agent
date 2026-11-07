from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RepairResult:
    ok: bool
    value: str = ""
    error_code: str = ""
    message: str = ""


def repair_order_id(raw: str) -> RepairResult:
    normalized = re.sub(r"[^0-9a-fA-F]", "", raw).lower()
    matches = re.findall(r"[0-9a-f]{32}", normalized)
    if len(matches) == 1:
        return RepairResult(ok=True, value=matches[0])
    if len(matches) > 1:
        return RepairResult(
            ok=False,
            error_code="ambiguous_order_id",
            message="检测到多个可能的 order_id，请明确要查询哪一个订单。",
        )
    if 0 < len(normalized) < 32:
        return RepairResult(
            ok=False,
            error_code="incomplete_order_id",
            message="order_id 不完整，请提供完整的 32 位订单号。",
        )
    return RepairResult(
        ok=False,
        error_code="missing_order_id",
        message="没有检测到有效 order_id，请提供完整的 32 位订单号。",
    )
