from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.intent.mapping import map_intent

ROOT = Path(__file__).resolve().parents[1]
INTENT_CASES = ROOT / "data" / "bitext_derived" / "intent_eval_cases.jsonl"
MULTI_CASES = ROOT / "data" / "bitext_derived" / "multi_intent_eval_cases.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def deterministic_eval() -> None:
    cases = load_jsonl(INTENT_CASES)
    correct = 0
    by_route_intent: dict[str, int] = {}
    for case in cases:
        mapped = map_intent(case["intent"])
        expected_intent = case.get("expected_intent", case.get("expected_skill"))
        if mapped.route_intent == expected_intent:
            correct += 1
        by_route_intent[mapped.route_intent] = by_route_intent.get(mapped.route_intent, 0) + 1

    multi_cases = load_jsonl(MULTI_CASES)
    print(f"Bitext intent eval cases: {len(cases)}")
    print(f"Mapping accuracy: {correct / len(cases):.2%}")
    print("Mapped route intents:")
    for route_intent, count in sorted(by_route_intent.items()):
        print(f"  {route_intent}: {count}")
    print(f"Multi-intent eval cases: {len(multi_cases)}")


async def llm_planner_eval(limit: int = 200) -> None:
    from app.config.llm_settings import resolve_llm_settings
    from app.llm.client import LlmClient
    from app.llm.intent_planner import IntentPlanner

    load_dotenv()
    settings = resolve_llm_settings(default_model="gpt-4o-mini")
    if settings.provider == "offline":
        raise SystemExit(
            "OPENAI_API_KEY or AIHUBMIX_API_KEY is not set. Set one before running "
            "RUN_LLM_ROUTER_EVAL=1."
        )
    client = LlmClient(
        api_key=settings.api_key,
        model=settings.model,
        base_url=settings.base_url,
    )
    planner = IntentPlanner(client.chat_openai)
    cases = load_jsonl(INTENT_CASES)[:limit]
    correct = 0
    wrong: dict[str, int] = {}
    for case in cases:
        result = await planner.classify(case["message"])
        expected_intent = case.get("expected_intent", case.get("expected_skill"))
        ok = result.intent == expected_intent
        correct += int(ok)
        if not ok:
            key = f"{expected_intent}->{result.intent}"
            wrong[key] = wrong.get(key, 0) + 1
    print(f"LLM planner route cases: {len(cases)}")
    print(f"LLM planner route accuracy: {correct / len(cases):.2%}")
    if wrong:
        print("Wrong predictions:")
        for key, count in sorted(wrong.items()):
            print(f"  {key}: {count}")


def main() -> None:
    deterministic_eval()
    if os.environ.get("RUN_LLM_ROUTER_EVAL") == "1":
        limit = int(os.environ.get("LLM_EVAL_LIMIT", "20"))
        asyncio.run(llm_planner_eval(limit=limit))


if __name__ == "__main__":
    main()
