#!/usr/bin/env python3
"""Dry-run/apply legacy session-thread pending Case binding for V2.

Default is read-only. Use --apply only after reviewing the JSON report and
backing up cases.db/checkpoints.db with the SQLite backup API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app import support_conversations
from app.case_workflow_service import CaseWorkflowService
from app.config.di import actions, agent_graph_builder
from app.olist.service import SQLiteCaseService


async def run(cases_db: Path, checkpoints_db: Path, apply: bool) -> dict:
    support_conversations.initialize()
    case_service = SQLiteCaseService(cases_db)
    workflow = CaseWorkflowService(case_service, actions)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoints_db)) as checkpointer:
        legacy_agent = agent_graph_builder.build(checkpointer=checkpointer)
        workflow.configure(checkpointer, legacy_agent=legacy_agent)
        return await workflow.migrate_legacy(apply=apply)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write legacy bindings; default dry-run")
    parser.add_argument("--cases-db", type=Path, default=Path("data/cases.db"))
    parser.add_argument("--checkpoints-db", type=Path, default=Path("data/checkpoints.db"))
    args = parser.parse_args()
    report = asyncio.run(run(args.cases_db, args.checkpoints_db, args.apply))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
