import asyncio

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

# Create the MCP server and give it a name
server = Server("Example Server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="say_hello",
            description="Say hello",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        types.Tool(
            name="add_numbers",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict | None):
    if arguments is None:
        arguments = {}

    if name == "say_hello":
        name_arg = arguments.get("name", "world")
        return [types.TextContent(type="text", text=f"Hello {name_arg}! I am a real MCP server.")] 

    if name == "add_numbers":
        a = int(arguments.get("a", 0))
        b = int(arguments.get("b", 0))
        return [types.TextContent(type="text", text=str(a + b))]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
       

if __name__ == "__main__":
    asyncio.run(main())
