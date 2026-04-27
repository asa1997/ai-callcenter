import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client


# ── Server registry ───────────────────────────────────────────────────────────
# Maps each tool to the HTTP server that handles it
# Uncomment as we build each server

from config import POLICY_SERVER_URL, DOCUMENT_SERVER_URL, RISK_SERVER_URL, CUSTOMER_SERVER_URL

TOOL_TO_SERVER = {
    "query_policies":        POLICY_SERVER_URL,
    "get_customer_profile":  CUSTOMER_SERVER_URL,
    "get_risk_profile":      RISK_SERVER_URL,
    "extract_document_info": DOCUMENT_SERVER_URL,
}


class MCPClient:
    """
    Single MCP client connecting to all MCP servers over HTTP/SSE.
    Servers must be running before this client is used.

    AIRS intercepts at two points:
    - Before tool call  (inspect tool name + arguments)
    - After tool call   (inspect response before agent sees it)
    """

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the appropriate MCP server over HTTP."""

        if tool_name not in TOOL_TO_SERVER:
            raise ValueError(
                f"Unknown tool: {tool_name}. "
                f"Available: {list(TOOL_TO_SERVER.keys())}"
            )

        server_url = TOOL_TO_SERVER[tool_name]

        # ── AIRS intercepts BEFORE tool call ─────────────────────────────────
        print(f"[MCP CLIENT] Tool:      {tool_name}")
        print(f"[MCP CLIENT] Arguments: {arguments}")
        print(f"[MCP CLIENT] Server:    {server_url}")

        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                response = result.content[0].text

                # ── AIRS intercepts AFTER tool call ───────────────────────────
                print(f"[MCP CLIENT] Response: {len(response)} chars received")
                print()

                return response

    def call_tool_sync(self, tool_name: str, arguments: dict) -> str:
        """Synchronous wrapper — agent uses this."""
        return asyncio.run(self.call_tool(tool_name, arguments))


# ── Convenience functions for agent.py ───────────────────────────────────────

def query_policies(query: str) -> str:
    """
    Query the Policy RAG MCP server.
    Requires policy_server.py running on port 8001.
    """
    client = MCPClient()
    return client.call_tool_sync("query_policies", {"query": query})


def extract_document_info(document_text: str, customer_id: str) -> str:
    """
    Send document to Document Processing MCP server.
    Requires document_server.py running on port 8004.
    """
    client = MCPClient()
    return client.call_tool_sync(
        "extract_document_info",
        {
            "document_text": document_text,
            "customer_id":   customer_id
        }
    )


async def extract_document_info_async(document_text: str, customer_id: str) -> str:
    """
    Async version — use from async endpoints (e.g. /upload).
    Requires document_server.py running on port 8004.
    """
    client = MCPClient()
    return await client.call_tool(
        "extract_document_info",
        {
            "document_text": document_text,
            "customer_id":   customer_id
        }
    )

def get_customer_profile(customer_id: str) -> str:
    """Get customer profile from Customer Profile MCP server."""
    client = MCPClient()
    return client.call_tool_sync(
        "get_customer_profile",
        {"customer_id": customer_id}
    )


def get_risk_profile(customer_id: str) -> str:
    """Get risk profile from Credit Risk MCP server."""
    client = MCPClient()
    return client.call_tool_sync(
        "get_risk_profile",
        {"customer_id": customer_id}
    )

# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MCP Client over HTTP...")
    print("Requires policy_server.py running on port 8001\n")

    result = query_policies(
        "what credit score do I need for a personal loan?"
    )
    print("RESULT:")
    print(result)
