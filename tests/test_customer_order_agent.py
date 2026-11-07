from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.agent.customer_order_agent import CustomerOrderAgent
from app.config.di import olist_service


class _ToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_user_orders",
                        "args": {"category": "健康与美容用品", "limit": 5},
                        "id": "call-1",
                    }
                ],
            )
        return AIMessage(content="请回复序号或完整的订单号。")


@pytest.mark.anyio
async def test_order_agent_uses_scoped_search_tool():
    agent = CustomerOrderAgent(_ToolCallingModel(), olist_service)
    allowed = {
        "203096f03d82e0dffbc41ebc2e2bcfb7",
        "360787554f41824600bdcba9687cd4f0",
    }
    result = await agent.resolve("健康与美容用品不要了", allowed_order_ids=allowed)
    assert result is not None
    assert "请直接选择" in result["answer"]
    assert "完整订单号" not in result["answer"]
    assert {item["order_id"] for item in result["candidates"]} == allowed
    assert all(item["category"] == "健康与美容用品" for item in result["candidates"])


@pytest.mark.anyio
async def test_order_agent_cannot_search_unowned_orders():
    agent = CustomerOrderAgent(_ToolCallingModel(), olist_service)
    result = await agent.resolve(
        "健康与美容用品不要了",
        allowed_order_ids={"203096f03d82e0dffbc41ebc2e2bcfb7"},
    )
    assert result is not None
    assert [item["order_id"] for item in result["candidates"]] == [
        "203096f03d82e0dffbc41ebc2e2bcfb7"
    ]


class _PolicyToolModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "evaluate_after_sales_policy",
                        "args": {
                            "order_id": "203096f03d82e0dffbc41ebc2e2bcfb7",
                            "action_type": "refund_request",
                        },
                        "id": "call-policy",
                    }
                ],
            )
        return AIMessage(content="这笔退款需要人工审核。")


@pytest.mark.anyio
async def test_agent_can_call_policy_tool():
    from app.olist.service import InMemoryCaseService
    from app.policy.engine import PolicyEngine
    from app.policy.guard import PolicyGuard

    guard = PolicyGuard(PolicyEngine(), olist_service, InMemoryCaseService())
    agent = CustomerOrderAgent(_PolicyToolModel(), olist_service, guard)
    result = await agent.resolve(
        "这笔退款怎么处理？",
        allowed_order_ids={"203096f03d82e0dffbc41ebc2e2bcfb7"},
    )
    assert result is not None
    assert result["answer"] == "这笔退款需要人工审核。"
