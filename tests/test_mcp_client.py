import sys
from pathlib import Path

import pytest

from app.mcp_client import ExternalMcpToolClient


@pytest.mark.anyio
async def test_external_mcp_client_lists_and_calls_local_server() -> None:
    client = ExternalMcpToolClient(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=Path(__file__).resolve().parents[1],
    )

    tools = await client.list_tools()
    assert "get_order_status" in tools

    result = await client.call_tool(
        "get_order_status",
        {"order_id": "203096f03d82e0dffbc41ebc2e2bcfb7"},
    )
    assert result.structured_content is not None
    assert "delivered" in result.content[0].text
