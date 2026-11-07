import mcp_types as types
import pytest

from app.stripe_mcp import StripeMcpAdapter, StripeMcpConfig


class _FakeRemoteMcpClient:
    def __init__(self) -> None:
        self.calls = []

    async def list_tools(self):
        return ["create_refund", "list_customers", "get_order_status"]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return types.CallToolResult(content=[], isError=False)


def test_stripe_config_from_env_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRIPE_API_KEY", raising=False)
    assert StripeMcpConfig.from_env() is None
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_x")
    config = StripeMcpConfig.from_env()
    assert config is not None
    assert config.api_key == "sk_test_x"


@pytest.mark.anyio
async def test_stripe_adapter_filters_payment_tools_and_calls_refund() -> None:
    client = _FakeRemoteMcpClient()
    adapter = StripeMcpAdapter(client)
    assert await adapter.list_payment_tools() == ["create_refund", "list_customers"]
    await adapter.create_refund("pi_test_123", amount_cents=500)
    assert client.calls == [("create_refund", {"payment_intent": "pi_test_123", "amount": 500})]
