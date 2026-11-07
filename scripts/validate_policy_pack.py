#!/usr/bin/env python3
"""Validate a policy pack, its effective windows, and rule conflicts.

Exit code 1 means the pack must not be deployed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.policy.engine import DEFAULT_PACK
from app.policy.schema import load_pack, validate_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args()
    pack = load_pack(args.pack)
    report = validate_pack(pack)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"{status} {args.pack} version={report['version']} hash={report['pack_hash'][:16]}…")
        for error in report["errors"]:
            print(f"  ERROR   {error['code']}: {error['message']}")
        for warning in report["warnings"]:
            print(f"  WARNING {warning['code']}: {warning['message']}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
