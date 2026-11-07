from __future__ import annotations

import json
from typing import Any

import anyio
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

from app.after_sales import AfterSalesDecisionEngine
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import OlistService, format_after_sales_report, format_order_status

olist_service = OlistService()
knowledge_base = MarkdownKnowledgeBase()
after_sales_engine = AfterSalesDecisionEngine()


def get_order_status(order_id: str) -> str:
    """Return deterministic Olist order status, delivery, payment, and review facts."""
    return format_order_status(olist_service.get_order_status(order_id))


def search_category_risk(query: str) -> list[dict[str, object]]:
    """Retrieve category-level logistics and review risk insights from Olist records."""
    return olist_service.category_insights(query)


def draft_escalation(order_id: str) -> dict[str, object]:
    """Draft a support escalation for delayed, canceled, or low-review orders."""
    draft = olist_service.escalation_draft(order_id)
    if draft is None:
        return {"found": False, "reason": "order_not_found"}
    return {"found": True, **draft}


def generate_after_sales_priority_report(query: str = "") -> dict[str, object]:
    """Generate a read-only after-sales operations decision report."""
    return olist_service.after_sales_priority_report(query)


def assess_after_sales_case(action_type: str, order_id: str, user_request: str) -> dict[str, object]:
    """Assess refund/cancel/address/invoice requests before any write action is executed."""
    order = olist_service.get_order_status(order_id)
    if order is None:
        return {"found": False, "reason": "order_not_found"}
    policy_hits = knowledge_base.search(user_request, k=3)
    case = after_sales_engine.assess(
        action_type=action_type,
        order=order,
        user_request=user_request,
        policy_sections=[
            {
                "section_title": hit.section_title,
                "source": hit.source,
                "source_type": hit.source_type,
                "text": hit.text,
                "score": hit.score,
            }
            for hit in policy_hits
        ],
    )
    return {"found": True, **case.model_dump()}


def list_enterprise_tool_boundaries() -> list[dict[str, object]]:
    """Return MCP-facing tool governance metadata for enterprise integration reviews."""
    return [
        {
            "tool": "get_order_status",
            "input_schema": {"order_id": "string[32], required"},
            "risk_level": "read",
            "side_effect": False,
            "auth_scope": "orders:read",
            "idempotency_required": False,
            "audit": "redacted order id, latency, role, tenant, result summary",
        },
        {
            "tool": "search_category_risk",
            "input_schema": {"query": "string, required"},
            "risk_level": "read",
            "side_effect": False,
            "auth_scope": "analytics:read",
            "idempotency_required": False,
            "audit": "query hash, tenant, latency, hit count",
        },
        {
            "tool": "generate_after_sales_priority_report",
            "input_schema": {"query": "string, optional"},
            "risk_level": "read",
            "side_effect": False,
            "auth_scope": "after_sales:ops_report",
            "idempotency_required": False,
            "audit": "manager role, tenant, generated report summary",
        },
        {
            "tool": "assess_after_sales_case",
            "input_schema": {
                "action_type": (
                    "enum[refund_request,cancel_order,change_address,"
                    "invoice_request,complaint_escalation,open_support_case]"
                ),
                "order_id": "string[32], required",
                "user_request": "string, required",
            },
            "risk_level": "medium",
            "side_effect": False,
            "auth_scope": "after_sales:assess",
            "idempotency_required": False,
            "audit": "decision outcome, risk level, policy refs, verifier flags",
        },
        {
            "tool": "draft_escalation",
            "input_schema": {"order_id": "string[32], required"},
            "risk_level": "medium",
            "side_effect": False,
            "auth_scope": "after_sales:draft",
            "idempotency_required": False,
            "audit": "draft reason and redacted message hash",
        },
        {
            "tool": "execute_side_effect",
            "input_schema": {
                "action_type": (
                    "enum[refund_request,cancel_order,change_address,"
                    "invoice_request,complaint_escalation,open_support_case]"
                ),
                "order_id": "string[32], required",
                "message_text": "string, required",
            },
            "risk_level": "critical",
            "side_effect": True,
            "auth_scope": "after_sales:write",
            "idempotency_required": True,
            "audit": "idempotency key, result id, duplicate flag, operator/session",
        },
    ]


async def list_tools(
    ctx: ServerRequestContext[Any],
    params: types.PaginatedRequestParams | None,
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name="get_order_status",
                description="Return Olist order status, delivery, payment, and review facts.",
                input_schema=_object_schema({"order_id": "32-character Olist order id"}, ["order_id"]),
            ),
            types.Tool(
                name="search_category_risk",
                description="Retrieve category-level logistics and review risk insights.",
                input_schema=_object_schema({"query": "Category or natural-language risk query"}, ["query"]),
            ),
            types.Tool(
                name="draft_escalation",
                description="Draft a support escalation for delayed, canceled, or low-review orders.",
                input_schema=_object_schema({"order_id": "32-character Olist order id"}, ["order_id"]),
            ),
            types.Tool(
                name="generate_after_sales_priority_report",
                description="Generate high-risk category and priority after-sales order recommendations.",
                input_schema=_object_schema({"query": "Natural-language operations decision request"}, []),
            ),
            types.Tool(
                name="assess_after_sales_case",
                description="Assess an after-sales case into approve/reject/review before write actions.",
                input_schema=_object_schema(
                    {
                        "action_type": (
                            "refund_request, cancel_order, change_address, "
                            "invoice_request, complaint_escalation, or open_support_case"
                        ),
                        "order_id": "32-character Olist order id",
                        "user_request": "Original customer or support-agent request",
                    },
                    ["action_type", "order_id", "user_request"],
                ),
            ),
            types.Tool(
                name="list_enterprise_tool_boundaries",
                description="List MCP tool risk levels, auth scopes, idempotency, and audit semantics.",
                input_schema=_object_schema({}, []),
            ),
        ]
    )


async def call_tool(
    ctx: ServerRequestContext[Any],
    params: types.CallToolRequestParams,
) -> types.CallToolResult:
    args = params.arguments or {}
    if params.name == "get_order_status":
        structured = {"answer": get_order_status(str(args.get("order_id", "")))}
    elif params.name == "search_category_risk":
        structured = {"insights": search_category_risk(str(args.get("query", "")))}
    elif params.name == "draft_escalation":
        structured = draft_escalation(str(args.get("order_id", "")))
    elif params.name == "generate_after_sales_priority_report":
        report = generate_after_sales_priority_report(str(args.get("query", "")))
        structured = {"report": report, "answer": format_after_sales_report(report)}
    elif params.name == "assess_after_sales_case":
        structured = assess_after_sales_case(
            str(args.get("action_type", "")),
            str(args.get("order_id", "")),
            str(args.get("user_request", "")),
        )
    elif params.name == "list_enterprise_tool_boundaries":
        structured = {"tools": list_enterprise_tool_boundaries()}
    else:
        return types.CallToolResult(
            content=[types.TextContent(text=f"Unknown tool: {params.name}")],
            is_error=True,
        )

    return types.CallToolResult(
        content=[types.TextContent(text=json.dumps(structured, ensure_ascii=False))],
        structured_content=structured,
    )


def create_server() -> Server:
    return Server(
        "olist-marketplace-support",
        version="0.2.0",
        description="MCP tools for the Olist marketplace support Agent.",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def _object_schema(properties: dict[str, str], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": description}
            for name, description in properties.items()
        },
        "required": required,
    }


async def run_server() -> None:
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(run_server)
