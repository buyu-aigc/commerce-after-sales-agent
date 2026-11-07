from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def native_structured_output_enabled(llm: object) -> bool:
    override = os.environ.get("OPENAI_NATIVE_STRUCTURED_OUTPUT", "").lower().strip()
    if override in {"0", "false", "no"}:
        return False
    if override in {"1", "true", "yes"}:
        return True

    values = [repr(llm)]
    for attr in ("openai_api_base", "base_url", "model_name", "model"):
        value = getattr(llm, attr, None)
        if value is not None:
            values.append(str(value))
    fingerprint = " ".join(values).lower()
    # Several OpenAI-compatible gateways accept the request but do not honour
    # LangChain's provider-specific structured-output wrapper.  JSON text with
    # validation below is more portable for those gateways.
    incompatible_markers = ("deepseek", "aihubmix", "coding-glm", "glm-")
    return not any(marker in fingerprint for marker in incompatible_markers)


def json_instruction(schema: type[BaseModel]) -> str:
    fields = ", ".join(schema.model_fields)
    return (
        "\n\nReturn ONLY valid JSON. Do not wrap it in markdown. "
        f"The JSON object must match this schema name: {schema.__name__}. "
        f"Top-level fields: {fields}."
    )


def add_json_instruction(messages: list[Any], schema: type[BaseModel]) -> list[Any]:
    if not messages:
        return messages
    updated = list(messages)
    last = updated[-1]
    content = str(getattr(last, "content", ""))
    updated[-1] = HumanMessage(content=f"{content}{json_instruction(schema)}")
    return updated


def parse_json_model(content: object, schema: type[T]) -> T:
    text = str(getattr(content, "content", content)).strip()
    payload = _repair_payload(_extract_json(text), schema)
    return schema.model_validate(payload)


def _extract_json(text: str) -> Any:
    try:
        loaded = json.loads(text)
        if isinstance(loaded, (dict, list)):
            return loaded
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        loaded = json.loads(fenced.group(1))
        if isinstance(loaded, (dict, list)):
            return loaded
    for start_token, end_token in (("{", "}"), ("[", "]")):
        start = text.find(start_token)
        end = text.rfind(end_token)
        if start >= 0 and end > start:
            loaded = json.loads(text[start : end + 1])
            if isinstance(loaded, (dict, list)):
                return loaded
    raise ValueError("LLM response does not contain a JSON object")


def _task_plan_items(payload: Any) -> list[Any]:
    """Normalize common planner wrappers from OpenAI-compatible gateways.

    Some gateways return ``[{"tasks": [...]}]`` or ``{"task_plan": [...]}``
    even when the requested schema is the object ``TaskPlanResult``.  Treating
    the wrapper itself as a task silently drops every real task and forces the
    deterministic fallback.
    """
    if isinstance(payload, dict):
        for key in ("tasks", "task_plan", "plan"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return _task_plan_items(value)
        return []
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            for key in ("tasks", "task_plan", "plan"):
                value = payload[0].get(key)
                if isinstance(value, list):
                    return value
        return payload
    return []


def _repair_payload(payload: Any, schema: type[BaseModel]) -> dict[str, Any]:
    if schema.__name__ == "TaskPlanResult":
        tasks = _task_plan_items(payload)
        repaired_tasks = []
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                repaired = dict(task)
                if "intent" not in repaired and "type" in repaired:
                    repaired["intent"] = repaired["type"]
                if "intent" not in repaired and "task" in repaired:
                    repaired["intent"] = repaired["task"]
                if "index" not in repaired and isinstance(repaired.get("task"), int):
                    repaired["index"] = repaired["task"]
                repaired.setdefault("text", "")
                repaired.setdefault("side_effect", False)
                repaired.setdefault("action_type", "none")
                if repaired.get("action_type") is None:
                    repaired["action_type"] = "none"
                if repaired.get("depends_on") is None:
                    repaired["depends_on"] = []
                elif isinstance(repaired.get("depends_on"), int):
                    repaired["depends_on"] = [repaired["depends_on"]]
                repaired_tasks.append(repaired)
        return {"tasks": repaired_tasks}
    if schema.__name__ == "OlistTaskResult":
        repaired = dict(payload)
        if "order_id" not in repaired:
            repaired["order_id"] = repaired.get("orderId", repaired.get("order", ""))
        repaired.setdefault("category", "")
        repaired.setdefault("user_goal", repaired.get("goal", ""))
        return repaired
    if schema.__name__ == "InputGuardResult":
        repaired = dict(payload)
        if "reason" not in repaired:
            repaired["reason"] = (
                repaired.get("block_reason")
                or repaired.get("category")
                or repaired.get("explanation")
                or "guard did not provide reason"
            )
        return repaired
    if schema.__name__ == "OutputGuardResult":
        repaired = dict(payload)
        if "reason" not in repaired:
            repaired["reason"] = repaired.get("explanation") or repaired.get("comment") or ""
        return repaired
    if schema.__name__ == "JudgeResult":
        repaired = dict(payload)
        if "rationale" not in repaired:
            repaired["rationale"] = (
                repaired.get("explanation")
                or repaired.get("reason")
                or repaired.get("comment")
                or "judge did not provide rationale"
            )
        repaired.setdefault("pass_overall", False)
        return repaired
    return payload
