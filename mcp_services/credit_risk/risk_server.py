import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
import uvicorn
from config import RISK_SERVER_PORT

# ── Load risk data ────────────────────────────────────────────────────────────
DATA_PATH = "mcp_services/credit_risk/data/risk_data.json"

print("Credit Risk Server starting (HTTP/SSE mode)...")
print(f"  Loading risk data from: {DATA_PATH}")

with open(DATA_PATH, "r") as f:
    risk_profiles = json.load(f)

print(f"  Loaded {len(risk_profiles)} risk profiles")
print("Credit Risk Server ready.\n")

# ── MCP Server ────────────────────────────────────────────────────────────────
server = Server("Credit Risk Service")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_risk_profile",
            description=(
                "Retrieve a customer's credit and risk profile by customer ID. "
                "Returns credit score, fraud risk, repayment risk and flag reason. "
                "Use this to assess eligibility for loans and credit products."
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
    if name == "get_risk_profile":
        return await handle_get_risk_profile(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def handle_get_risk_profile(
    arguments: dict
) -> list[types.TextContent]:

    customer_id = arguments.get("customer_id", "").strip()

    # ── AIRS intercepts here ──────────────────────────────────────────────────
    # Risk data contains internal scoring criteria
    # Threat 2: policy fishing via risk profile reveals
    # exact scoring factors — GR-004 violation
    print(f"[TOOL CALLED] get_risk_profile")
    print(f"[CUSTOMER ID] {customer_id}")

    if not customer_id:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "customer_id is required"})
        )]

    profile = risk_profiles.get(customer_id)

    if not profile:
        print(f"[NOT FOUND]   customer_id={customer_id}")
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "error":       f"Risk profile for {customer_id} not found",
                "customer_id": customer_id
            })
        )]

    print(f"[FOUND]       credit_score={profile['credit_score']} "
          f"fraud_risk={profile['fraud_risk']} "
          f"repayment_risk={profile['repayment_risk']}")
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
    print(f"Starting Credit Risk MCP server on http://localhost:{RISK_SERVER_PORT}")
    print("  GET  /sse      ← MCP client connects here")
    print("  POST /messages ← MCP tool calls arrive here")
    print()
    uvicorn.run(asgi_app, host="0.0.0.0", port=RISK_SERVER_PORT)
