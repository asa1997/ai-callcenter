import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import json
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizer

from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response
import uvicorn

# ── Configuration ─────────────────────────────────────────────────────────────
from config import (
    DOCUMENT_MODEL_PATH as MODEL_PATH,
    DOCUMENT_MODEL_NAME as MODEL_NAME,
    DOCUMENT_MAX_LENGTH as MAX_LENGTH,
    DOCUMENT_SERVER_PORT,
)

# ── Startup ───────────────────────────────────────────────────────────────────
print("Document Processing Server starting (HTTP/SSE mode)...")

print(f"  Loading tokenizer: {MODEL_NAME}")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
print("  Tokenizer loaded")

print(f"  Loading ONNX model: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH)
print("  ONNX model loaded")
print("Document Processing Server ready.\n")

# ── Field extraction helpers ──────────────────────────────────────────────────

def extract_income(text: str) -> int | None:
    """
    Extract income from document text.
    Handles Indian number format: Rs. 12,00,000
    """
    patterns = [
        # Matches: income/salary Rs. 12,00,000
        r'(?:income|salary|earnings|ctc|annual)[^\d]*(?:rs\.?|inr|₹)?\s*([\d,]+)',
        # Matches: Rs. 12,00,000 per annum
        r'(?:rs\.?|inr|₹)\s*([\d,]+)(?:\s*(?:per annum|pa|p\.a\.|annually|per year))',
        # Matches: Gross Total Income: Rs. 12,00,000
        r'gross\s+total\s+income[^\d]*(?:rs\.?|inr|₹)?\s*([\d,]+)',
        # Matches: Net Income: Rs. 9,50,000
        r'net\s+income[^\d]*(?:rs\.?|inr|₹)?\s*([\d,]+)',
        # Matches any Rs. amount with per annum
        r'([\d,]+)(?:\s*(?:per annum|pa|p\.a\.|annually|per year))',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            # Remove commas — handles both 8,00,000 and 800,000
            raw = match.group(1).replace(",", "")
            try:
                value = int(raw)
                # Sanity check — income should be reasonable
                if 10000 <= value <= 100000000:
                    return value
            except ValueError:
                continue
    return None


def extract_employment_type(text: str) -> str:
    """Extract employment type from document text."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["salaried", "employed by", "employee", "payslip", "salary slip"]):
        return "salaried"
    if any(w in text_lower for w in ["self-employed", "self employed", "business", "proprietor", "itr", "profit"]):
        return "self_employed"
    if any(w in text_lower for w in ["freelance", "consultant", "contract"]):
        return "freelancer"
    return "unknown"


def extract_document_type(text: str) -> str:
    """Classify what type of document this is."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["salary slip", "payslip", "pay stub"]):
        return "salary_slip"
    if any(w in text_lower for w in ["itr", "income tax return", "form 16"]):
        return "itr"
    if any(w in text_lower for w in ["bank statement", "account statement"]):
        return "bank_statement"
    if any(w in text_lower for w in ["income", "salary", "employment"]):
        return "income_proof"
    return "unknown"


def run_model(text: str) -> np.ndarray:
    """
    Run DistilBERT ONNX model on input text.
    Returns the CLS token embedding — represents the whole document.
    This is the AIRS model scanning target.
    """
    inputs = tokenizer(
        text,
        return_tensors="np",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True
    )
    outputs = session.run(
        None,
        {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
    )
    # Return CLS token embedding (first token)
    return outputs[0][0][0]


# ── MCP Server ────────────────────────────────────────────────────────────────
server = Server("Document Processing Service")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="extract_document_info",
            description=(
                "Extract structured information from a customer uploaded document. "
                "Returns income, employment type, and document type. "
                "Use this tool when a customer uploads a financial document "
                "such as a salary slip, ITR, or income proof."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "document_text": {
                        "type": "string",
                        "description": "The full text content of the uploaded document"
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID who uploaded the document"
                    }
                },
                "required": ["document_text", "customer_id"]
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
    if name == "extract_document_info":
        return await handle_extract_document_info(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def handle_extract_document_info(
    arguments: dict
) -> list[types.TextContent]:

    document_text = arguments.get("document_text", "").strip()
    customer_id   = arguments.get("customer_id", "unknown")

    if not document_text:
        return [types.TextContent(
            type="text",
            text="Error: document_text cannot be empty"
        )]

    # ── AIRS intercept point 1 — inspect INPUT before model runs ─────────────
    print(f"[TOOL CALLED] extract_document_info")
    print(f"[CUSTOMER]    {customer_id}")
    print(f"[DOC LENGTH]  {len(document_text)} chars")

    

    # ── Run ONNX model ────────────────────────────────────────────────────────
    try:
        embedding = run_model(document_text)
        print(f"[MODEL]       ONNX inference complete — embedding shape: {embedding.shape}")
    except Exception as e:
        print(f"[ERROR]       Model inference failed: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error processing document: {str(e)}"
        )]

    # ── Extract fields ────────────────────────────────────────────────────────
    income          = extract_income(document_text)
    employment_type = extract_employment_type(document_text)
    document_type   = extract_document_type(document_text)

    # ── Build result ──────────────────────────────────────────────────────────
    result = {
        "customer_id":      customer_id,
        "document_type":    document_type,
        "employment_type":  employment_type,
        "income":           income,
        "income_formatted": f"Rs. {income:,}" if income else "Not found",
      
        "raw_text_preview": document_text[:200],
        "full_text": document_text,
    }

    # ── AIRS intercept point 2 — inspect OUTPUT before agent sees it ──────────
    print(f"[EXTRACTED]   income={result['income_formatted']}")
    print(f"[EXTRACTED]   employment_type={employment_type}")
    print(f"[EXTRACTED]   document_type={document_type}")
   
    print()

    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
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
    print("Starting Document Processing MCP server on http://localhost:8004")
    print("  GET  /sse      ← MCP client connects here")
    print("  POST /messages ← MCP tool calls arrive here")
    print()
    uvicorn.run(asgi_app, host="0.0.0.0", port=DOCUMENT_SERVER_PORT)
