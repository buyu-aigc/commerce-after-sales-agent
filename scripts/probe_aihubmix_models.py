from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from openai import OpenAI, RateLimitError

DEFAULT_MODELS = [
    "coding-glm-5-free",
    "coding-glm-5.3-free",
    "coding-glm-5.3-flash-free",
    "deepseek-v4-flash",
    "gpt-4.1-nano-free",
    "gemini-3.5-flash-lite-free",
    "qwen3.6-plus-preview-free",
    "hy3-free",
]

JSON_PROBE = (
    "Return ONLY valid compact JSON with exactly these keys: "
    '{"ok": true, "model_quality": "brief", "risk": "low"}.'
)


def main() -> None:
    args = _parse_args()
    api_key = os.environ.get("AIHUBMIX_API_KEY")
    if not api_key:
        raise SystemExit("AIHUBMIX_API_KEY is not set in the current shell.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    if args.list_free_models:
        model_ids = sorted(model.id for model in client.models.list().data)
        free_models = [model_id for model_id in model_ids if "free" in model_id.lower()]
        print(
            json.dumps(
                {
                    "total_models": len(model_ids),
                    "free_model_count": len(free_models),
                    "free_models": free_models,
                },
                ensure_ascii=False,
            )
        )
        return

    rows: list[dict[str, Any]] = []
    for model in args.models:
        row = probe_model(client, model, args.max_tokens)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
        if args.stop_on_quota and row.get("error_type") == "rate_limit":
            break

    ok_models = [row["model"] for row in rows if row.get("ok")]
    print(
        json.dumps(
            {
                "summary": {
                    "tested": len(rows),
                    "ok_models": ok_models,
                    "json_valid_models": [
                        row["model"] for row in rows if row.get("json_valid")
                    ],
                }
            },
            ensure_ascii=False,
        )
    )


def probe_model(client: OpenAI, model: str, max_tokens: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Do not use markdown.",
                },
                {"role": "user", "content": JSON_PROBE},
            ],
            max_tokens=max_tokens,
            temperature=0,
            stream=False,
        )
    except RateLimitError as exc:
        return _error_row(model, started, "rate_limit", exc)
    except Exception as exc:
        return _error_row(model, started, type(exc).__name__, exc)

    content = response.choices[0].message.content or ""
    json_valid = _is_json_object(content)
    return {
        "model": model,
        "ok": True,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "json_valid": json_valid,
        "content": content.strip()[:400],
    }


def _error_row(
    model: str,
    started: float,
    error_type: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "model": model,
        "ok": False,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "error_type": error_type,
        "error": str(exc)[:400],
    }


def _is_json_object(content: str) -> bool:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe low-cost AIHubMix chat models with a tiny JSON-only request."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model ids to test. Defaults to a small free/cheap candidate list.",
    )
    parser.add_argument("--base-url", default="https://aihubmix.com/v1")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--list-free-models",
        action="store_true",
        help="List account-visible model ids containing 'free' without sending chat requests.",
    )
    parser.add_argument(
        "--stop-on-quota",
        action="store_true",
        help="Stop after the first provider quota/rate-limit error.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
