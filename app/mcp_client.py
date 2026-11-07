from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client


class ExternalMcpToolClient:
    """Thin MCP client adapter for calling external enterprise tools.

    In production the command would point to a separately owned MCP server,
    such as an OMS, CRM, ticketing, coupon, or knowledge-base connector.
    """

    def __init__(self, command: str, args: list[str] | None = None, cwd: str | Path | None = None) -> None:
        self._server = StdioServerParameters(command=command, args=args or [], cwd=cwd)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        async with stdio_client(self._server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> list[str]:
        async with self.session() as session:
            result = await session.list_tools()
            return [tool.name for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        async with self.session() as session:
            result = await session.call_tool(name, arguments)
            if not isinstance(result, types.CallToolResult):
                raise RuntimeError(f"MCP tool {name} returned input-required result")
            return result


class RemoteMcpToolClient:
    """Remote Streamable HTTP MCP client.

    This is the adapter shape used for external SaaS or enterprise MCP servers.
    Stripe can be plugged in with its remote MCP URL and sandbox API key; an
    internal OMS/CRM MCP server would use the same class with enterprise auth.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout_seconds) as http_client:
            async with streamable_http_client(self._url, http_client=http_client) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

    async def list_tools(self) -> list[str]:
        async with self.session() as session:
            result = await session.list_tools()
            return [tool.name for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        async with self.session() as session:
            result = await session.call_tool(name, arguments)
            if not isinstance(result, types.CallToolResult):
                raise RuntimeError(f"MCP tool {name} returned input-required result")
            return result
