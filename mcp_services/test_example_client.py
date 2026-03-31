import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Connect to our example server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_services/example_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("\n=== TOOLS AVAILABLE ON THIS MCP SERVER ===")
            for tool in tools.tools:
                print(f"  • {tool.name}: {tool.description}")

            # Call say_hello
            print("\n=== CALLING say_hello ===")
            result = await session.call_tool("say_hello", {"name": "Security Architect"})
            print(f"  Result: {result.content[0].text}")

            # Call add_numbers
            print("\n=== CALLING add_numbers ===")
            result = await session.call_tool("add_numbers", {"a": 42, "b": 58})
            print(f"  Result: {result.content[0].text}")

            # INTERACTIVE MODE (moved inside main)
            print("\n=== INTERACTIVE MODE ===")
            print("Type 'say_hello <name>' or 'add_numbers <a> <b>' or 'quit'")
            while True:
                cmd = input("> ").strip()
                if cmd == "quit":
                    break
                elif cmd.startswith("say_hello "):
                    name = cmd.split(" ", 1)[1]
                    result = await session.call_tool("say_hello", {"name": name})
                    print(f"Result: {result.content[0].text}")
                elif cmd.startswith("add_numbers "):
                    parts = cmd.split()
                    if len(parts) == 3:
                        a, b = int(parts[1]), int(parts[2])
                        result = await session.call_tool("add_numbers", {"a": a, "b": b})
                        print(f"Result: {result.content[0].text}")
                    else:
                        print("Usage: add_numbers <a> <b>")
                else:
                    print("Unknown command")

asyncio.run(main())