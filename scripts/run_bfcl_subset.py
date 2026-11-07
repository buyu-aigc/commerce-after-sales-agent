"""Prepare BFCL subset generation/evaluation commands.

BFCL evaluates model/tool-calling capability, not the Olist business graph. This
launcher keeps the dependency outside the main service environment and writes a
reproducible manifest for benchmark runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "benchmark_runs" / "bfcl"
SUMMARY = ROOT / "scripts" / "summarize_bfcl_results.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or prepare a BFCL subset benchmark.")
    parser.add_argument("--bfcl-root", required=True, help="External BFCL project root.")
    parser.add_argument("--model", default="gpt-4.1-mini-FC", help="BFCL-supported model name.")
    parser.add_argument(
        "--test-category",
        action="append",
        default=[],
        help="BFCL category, e.g. simple_python, multiple_python, parallel_python, multi_turn_base.",
    )
    parser.add_argument("--generate", action="store_true", help="Run BFCL response generation.")
    parser.add_argument("--evaluate", action="store_true", help="Run BFCL evaluation.")
    parser.add_argument("--partial-eval", action="store_true", help="Evaluate only generated subset ids.")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bfcl_root = Path(args.bfcl_root).expanduser().resolve()
    categories = args.test_category or ["simple_python", "multiple_python", "parallel_python"]
    if not args.generate and not args.evaluate:
        args.generate = True
        args.evaluate = True

    commands: list[list[str]] = []
    if args.generate:
        commands.append(
            [
                "python",
                "-m",
                "bfcl_eval.openfunctions_evaluation",
                "--model",
                args.model,
                "--test-category",
                *categories,
            ]
        )
    if args.evaluate:
        command = ["bfcl", "evaluate", "--model", args.model, "--test-category", *categories]
        if args.partial_eval:
            command.append("--partial-eval")
        commands.append(command)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark": "Berkeley Function Calling Leaderboard",
        "purpose": "tool/function-calling reliability benchmark, separate from Olist business eval",
        "bfcl_root": str(bfcl_root),
        "model": args.model,
        "categories": categories,
        "commands": commands,
        "notes": [
            "Install BFCL externally, e.g. pip install bfcl-eval or editable upstream checkout.",
            "Use --partial-eval only for smoke/subset runs; do not compare partial scores to leaderboard.",
            "Run scripts/summarize_bfcl_results.py after evaluation to extract score CSV/JSON.",
            (
                "BFCL scores support the tool-governance claim, "
                "not the end-to-end e-commerce task-success claim."
            ),
        ],
    }
    (RUN_DIR / "last_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for command in commands:
        print("Command:")
        print(" ".join(command))
    print(f"Manifest: {RUN_DIR / 'last_manifest.json'}")
    if args.dry_run:
        return 0
    if not bfcl_root.exists():
        raise SystemExit(f"BFCL root does not exist: {bfcl_root}")

    env = os.environ.copy()
    env.setdefault("BFCL_PROJECT_ROOT", str(bfcl_root))
    for command in commands:
        completed = subprocess.run(command, cwd=bfcl_root, env=env, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    if not args.skip_summary and SUMMARY.exists():
        score_root = bfcl_root / "score"
        summary_command = [
            sys.executable,
            str(SUMMARY),
            "--score-root",
            str(score_root),
            "--model",
            args.model,
        ]
        return int(subprocess.run(summary_command, cwd=ROOT, check=False).returncode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
