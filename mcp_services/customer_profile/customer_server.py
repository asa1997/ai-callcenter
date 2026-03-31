import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
from starlette.responses import Response
import uvicorn
from config import CUSTOMER_SERVER_PORT

# ── Load customer data ────────────────────────────────────────────────────────
DATA_PATH = "mcp_services/customer_profile/data/customers.json"

print("Customer Profile Server starting (HTTP/SSE mode)...")
print(f"  Loading customer data from: {DATA_PATH}")

with open(DATA_PATH, "r") as f:
    customers = json.load(f)

print(f"  Loaded {len(customers)} customer profiles")
print("Customer Profile Server ready.\n")

# ── MCP Server ────────────────────────────────────────────────────────────────
server = Server("Customer Profile Service")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_customer_profile",
            description=(
                "Retrieve a customer profile by customer ID. "
                "Returns name, segment, account status, relationship "
                "value and tenure. Use this to personalise responses."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID to look up"
                    }
                },
                "required": ["customer_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict | None
) -> list[types.TextContent]:
    if arguments is None:
        arguments = {}
    if name == "get_customer_profile":
        return await handle_get_customer_profile(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def handle_get_customer_profile(
    arguments: dict
) -> list[types.TextContent]:

    customer_id = arguments.get("customer_id", "").strip()

    # ── AIRS intercepts here ──────────────────────────────────────────────────
    # In secured system AIRS checks:
    # Does customer_id match the authenticated session?
    # If not — cross customer access attempt — BLOCK
    print(f"[TOOL CALLED] get_customer_profile")
    print(f"[CUSTOMER ID] {customer_id}")

    if not customer_id:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "customer_id is required"})
        )]

    # Look up customer
    profile = customers.get(customer_id)

    if not profile:
        print(f"[NOT FOUND]   customer_id={customer_id}")
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "error":       f"Customer {customer_id} not found",
                "customer_id": customer_id
            })
        )]

    print(f"[FOUND]       {profile['name']} — {profile['segment']} segment")
    print()

    return [types.TextContent(
        type="text",
        text=json.dumps(profile, indent=2)
    )]


# ── Raw ASGI app ──────────────────────────────────────────────────────────────
sse_transport = SseServerTransport("/messages")


async def asgi_app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    if scope["type"] != "http":
        return

    path = scope.get("path", "")

    if path == "/sse":
        async with sse_transport.connect_sse(scope, receive, send) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options()
            )
    elif path.startswith("/messages"):
        await sse_transport.handle_post_message(scope, receive, send)
    else:
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"Not found"})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting Customer Profile MCP server on http://localhost:{CUSTOMER_SERVER_PORT}")
    print("  GET  /sse      ← MCP client connects here")
    print("  POST /messages ← MCP tool calls arrive here")
    print()
    uvicorn.run(asgi_app, host="0.0.0.0", port=CUSTOMER_SERVER_PORT)
