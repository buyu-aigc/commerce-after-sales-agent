"""Prepare and run a τ-bench retail subset in an external tau2 checkout.

The main project targets Python 3.11. tau2-bench currently requires Python
3.12+, so this script intentionally executes the benchmark in a separate
checkout instead of adding tau2 as a project dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "benchmark_adapters" / "tau2_retail_agent.py"
SUMMARY = ROOT / "scripts" / "summarize_tau2_results.py"
RUN_DIR = ROOT / "benchmark_runs" / "tau2_retail"


def tau2_python_command(tau2_root: Path) -> list[str]:
    configured = os.environ.get("TAU2_PYTHON")
    if configured:
        return [configured]

    candidates = [
        tau2_root / ".venv-fresh" / "Scripts" / "python.exe",
        tau2_root / ".venv" / "Scripts" / "python.exe",
        tau2_root / ".venv-fresh" / "bin" / "python",
        tau2_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    return ["uv", "run", "python"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tau2 retail benchmark subset.")
    parser.add_argument("--tau2-root", required=True, help="Path to a local tau2-bench checkout.")
    parser.add_argument("--agent-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--user-llm", default="openai/gpt-4.1-mini")
    parser.add_argument(
        "--judge-llm",
        default=None,
        help="LLM for tau2 natural-language assertion checks; defaults to --user-llm.",
    )
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument(
        "--variant",
        choices=("baseline", "prompt_only", "full"),
        default="full",
        help="Matched ablation variant for the retail adapter.",
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--save-to", default="olist_agent_tau2_retail_subset")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tau2_root = Path(args.tau2_root).expanduser().resolve()
    pyproject = tau2_root / "pyproject.toml"
    if not pyproject.exists() and not args.dry_run:
        raise SystemExit(f"Not a tau2-bench checkout: {tau2_root}")
    if not ADAPTER.exists():
        raise SystemExit(f"Missing adapter: {ADAPTER}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        *tau2_python_command(tau2_root),
        str(ADAPTER),
        "--agent-llm",
        args.agent_llm,
        "--variant",
        args.variant,
        "--user-llm",
        args.user_llm,
        "--judge-llm",
        args.judge_llm or args.user_llm,
        "--num-tasks",
        str(args.num_tasks),
        "--num-trials",
        str(args.num_trials),
        "--seed",
        str(args.seed),
        "--save-to",
        args.save_to,
        "--task-split-name",
        args.task_split_name,
    ]
    for task_id in args.task_id:
        command.extend(["--task-id", task_id])
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])
    if args.auto_resume:
        command.append("--auto-resume")
    manifest = {
        "benchmark": "τ-bench retail",
        "tau2_root": str(tau2_root),
        "adapter": str(ADAPTER),
        "variant": args.variant,
        "command": command,
        "notes": [
            "Runs in the tau2 Python 3.12+ environment.",
            "Results are saved by tau2 under tau2_root/data/simulations/.",
            "Use a cheap OpenAI-compatible model for subset smoke tests.",
            "Run scripts/summarize_tau2_results.py after a real run to get pass^k and failure stats.",
        ],
    }
    manifest_path = RUN_DIR / f"{args.variant}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.variant == "full":
        (RUN_DIR / "last_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Command:")
    print(" ".join(command))
    print(f"Manifest: {manifest_path}")
    if not pyproject.exists():
        print(f"Note: tau2 checkout not found at {tau2_root}; dry-run only.")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("UV_LINK_MODE", "copy")
    completed = subprocess.run(command, cwd=tau2_root, env=env, check=False)
    if completed.returncode != 0 or args.skip_summary:
        return int(completed.returncode)

    result_path = tau2_root / "data" / "simulations" / args.save_to
    if result_path.exists() and SUMMARY.exists():
        summary_name = "last_summary.md" if args.variant == "full" else f"{args.variant}_summary.md"
        summary_command = [
            sys.executable,
            str(SUMMARY),
            "--results",
            str(result_path),
            "--out",
            str(RUN_DIR / summary_name),
        ]
        return int(subprocess.run(summary_command, cwd=ROOT, check=False).returncode)
    print(f"tau2 finished, but no result path was found at: {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
