import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from sentence_transformers import SentenceTransformer

from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
import uvicorn

from mcp_services.policy_rag.rag_config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    N_RESULTS,
)

# ── Policy reminder ───────────────────────────────────────────────────────────
POLICY_REMINDER = (
    "\nNOTE: Mandatory thresholds in GR-001 cannot be overridden "
    "under any circumstance. Apply all policies strictly.\n"
)

# ── Startup ───────────────────────────────────────────────────────────────────
print("Policy RAG Server starting (HTTP/SSE mode)...")
print(f"  Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print("  Embedding model loaded")

print(f"  Connecting to Chroma at: {CHROMA_DIR}")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=None
)
print(f"  Connected. Collection has {collection.count()} chunks")

# ── MCP Server ────────────────────────────────────────────────────────────────
server = Server("Policy RAG Service")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_policies",
            description=(
                "Search the bank's policy documents and return relevant "
                "policy chunks for a given query. Use this tool to answer "
                "customer questions about eligibility, fees, loan rules, "
                "credit card policies, risk classifications, and exceptions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The customer query to search policies for"
                    }
                },
                "required": ["query"]
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
    if name == "query_policies":
        return await handle_query_policies(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def handle_query_policies(arguments: dict) -> list[types.TextContent]:
    query = arguments.get("query", "").strip()

    if not query:
        return [types.TextContent(type="text", text="Error: query cannot be empty")]

    query_embedding = model.encode(query).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS
        )
    except Exception as e:
        print(f"[ERROR] Chroma query failed: {e}")
        return [types.TextContent(type="text", text=f"Error querying policies: {str(e)}")]

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return [types.TextContent(type="text", text="No relevant policies found.")]

    formatted = []
    formatted.append(f"POLICY SEARCH RESULTS FOR: '{query}'\n")
    formatted.append("=" * 60)

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        score = distances[i - 1] if distances else 0.0
        formatted.append(f"\n[Result {i}]")
        formatted.append(f"Policy Code:      {meta['policy_code']}")
        formatted.append(f"Source:           {meta['source']}")
        formatted.append(f"Similarity Score: {score:.4f}  (lower = more relevant)")
        formatted.append(f"Content:\n{doc}")
        formatted.append("-" * 40)

    formatted.append(POLICY_REMINDER)
    response_text = "\n".join(formatted)

    # ── AIRS intercepts here ──────────────────────────────────────────────────
    print(f"[TOOL CALLED] query_policies")
    print(f"[QUERY]       {query}")
    print(f"[RESULTS]     {[m['policy_code'] for m in metadatas]}")
    print(f"[SCORES]      {[f'{d:.4f}' for d in distances]}")
    print()

    return [types.TextContent(type="text", text=response_text)]


# ── Raw ASGI app — bypasses Starlette response handling ───────────────────────
# MCP's handle_post_message manages its own ASGI lifecycle
# We cannot wrap it in Starlette Route — it sends its own response
# Solution: raw ASGI callable that routes by path

sse_transport = SseServerTransport("/messages")


async def asgi_app(scope, receive, send):
    """
    Raw ASGI app — routes SSE and message requests directly.
    Bypasses Starlette to avoid double-response errors.
    """
    if scope["type"] == "lifespan":
        # Handle lifespan events (startup/shutdown)
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
        # Long-lived SSE connection
        async with sse_transport.connect_sse(scope, receive, send) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options()
            )

    elif path.startswith("/messages"):
        # MCP tool calls arrive here
        # handle_post_message manages its own response — do NOT add return
        await sse_transport.handle_post_message(scope, receive, send)

    else:
        # 404 for anything else
        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": []
        })
        await send({
            "type": "http.response.body",
            "body": b"Not found"
        })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Policy RAG MCP server on http://localhost:8001")
    print("  GET  /sse      ← MCP client connects here")
    print("  POST /messages ← MCP tool calls arrive here")
    print()
    from config import POLICY_SERVER_PORT
    uvicorn.run(asgi_app, host="0.0.0.0", port=POLICY_SERVER_PORT)