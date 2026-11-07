from __future__ import annotations

from app.tools.repair import repair_order_id

CASES = [
    {
        "id": "exact",
        "raw": "203096f03d82e0dffbc41ebc2e2bcfb7",
        "ok": True,
        "value": "203096f03d82e0dffbc41ebc2e2bcfb7",
    },
    {
        "id": "spaced",
        "raw": "203096f0 3d82 e0df fbc41ebc2e2bcfb7",
        "ok": True,
        "value": "203096f03d82e0dffbc41ebc2e2bcfb7",
    },
    {
        "id": "upper-with-prefix",
        "raw": "订单：203096F03D82E0DFFBC41EBC2E2BCFB7",
        "ok": True,
        "value": "203096f03d82e0dffbc41ebc2e2bcfb7",
    },
    {"id": "short", "raw": "203096f03d82", "ok": False, "error_code": "incomplete_order_id"},
    {"id": "missing", "raw": "帮我查一下订单状态", "ok": False, "error_code": "missing_order_id"},
    {
        "id": "ambiguous",
        "raw": "203096f03d82e0dffbc41ebc2e2bcfb7 53cdb2fc8bc7dce0b6741e2150273451",
        "ok": False,
        "error_code": "ambiguous_order_id",
    },
]


def main() -> None:
    passed = 0
    failures = []
    for case in CASES:
        result = repair_order_id(case["raw"])
        ok = result.ok == case["ok"]
        if result.ok:
            ok = ok and result.value == case["value"]
        else:
            ok = ok and result.error_code == case["error_code"]
        passed += int(ok)
        if not ok:
            failures.append(case["id"])
    print(f"Tool repair eval cases: {len(CASES)}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {passed / len(CASES):.2%}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure}")


if __name__ == "__main__":
    main()
