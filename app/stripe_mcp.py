from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mcp_types as types

from app.mcp_client import RemoteMcpToolClient

DEFAULT_STRIPE_MCP_URL = "https://mcp.stripe.com"


@dataclass(frozen=True)
class StripeMcpConfig:
    url: str
    api_key: str

    @classmethod
    def from_env(cls) -> StripeMcpConfig | None:
        api_key = os.environ.get("STRIPE_API_KEY")
        if not api_key:
            return None
        return cls(url=os.environ.get("STRIPE_MCP_URL", DEFAULT_STRIPE_MCP_URL), api_key=api_key)


class StripeMcpAdapter:
    """Optional Stripe MCP adapter for payment/refund side-effect workflows."""

    def __init__(self, client: RemoteMcpToolClient) -> None:
        self._client = client

    @classmethod
    def from_config(cls, config: StripeMcpConfig) -> StripeMcpAdapter:
        return cls(
            RemoteMcpToolClient(
                url=config.url,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        )

    async def list_payment_tools(self) -> list[str]:
        tools = await self._client.list_tools()
        keywords = ("payment", "refund", "charge", "invoice", "customer")
        return [tool for tool in tools if any(keyword in tool.lower() for keyword in keywords)]

    async def create_refund(
        self,
        payment_intent: str,
        amount_cents: int | None = None,
    ) -> types.CallToolResult:
        arguments: dict[str, Any] = {"payment_intent": payment_intent}
        if amount_cents is not None:
            arguments["amount"] = amount_cents
        return await self._client.call_tool("create_refund", arguments)
