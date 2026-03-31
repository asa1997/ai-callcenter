import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from agent.agent import handle_message
from agent.mcp_client import extract_document_info

app = FastAPI(
    title="AI Call Center Agent",
    description="Palo Alto AIRS COE Lab — Vulnerable Baseline System",
    version="1.0.0"
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    customer_id:      str
    message:          str
    document_context: str = ""   # optional — JSON string from /upload


class ChatResponse(BaseModel):
    customer_id:  str
    response:     str
    intent:       str
    tools_called: list[str]
    escalated:    bool


class UploadRequest(BaseModel):
    customer_id:   str
    document_text: str


class UploadResponse(BaseModel):
    customer_id:      str
    document_type:    str
    employment_type:  str
    income:           int | None
    income_formatted: str
    document_context: str    # pass this to /chat as document_context
    message:          str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "system":  "AI Call Center Agent",
        "status":  "running",
        "warning": "Vulnerable baseline — no guardrails active",
        "airs":    "Not yet integrated — Palo Alto COE workshop",
        "endpoints": {
            "POST /chat":   "Send a customer query or complaint",
            "POST /upload": "Upload a document for processing"
        }
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Receives customer message and optional document_context.
    Routes to agent — classifies intent, calls MCP tools, generates response.

    Intentionally vulnerable:
    - No input validation
    - document_context passed directly to agent
    - Threat 3: poisoned document_context executes in agent
    """
    if not request.customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")
    if not request.message:
        raise HTTPException(status_code=400, detail="message is required")

    result = handle_message(
        request.customer_id,
        request.message,
        request.document_context
    )

    return ChatResponse(
        customer_id=  request.customer_id,
        response=     result["response"],
        intent=       result["intent"],
        tools_called= result["tools_called"],
        escalated=    result["escalated"]
    )


@app.post("/upload", response_model=UploadResponse)
def upload(request: UploadRequest):
    """
    Document upload endpoint.

    Customer submits document text.
    Calls Document Processing MCP server (port 8004).
    DistilBERT ONNX model extracts structured fields.
    Returns document_context — pass this to /chat.

    Intentionally vulnerable:
    - No input sanitisation
    - full_text returned unfiltered
    - Threat 3: injection in document flows to agent
    - AIRS guardrail intercepts at MCP output in secured system
    """
    if not request.customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")
    if not request.document_text:
        raise HTTPException(status_code=400, detail="document_text is required")

    # Call document MCP server via mcp_client
    raw_result = extract_document_info(
        request.document_text,
        request.customer_id
    )

    # Parse JSON result from document server
    result = json.loads(raw_result)

    # document_context is what the customer passes back to /chat
    # It contains full_text — injection flows through here unblocked
    document_context = json.dumps({
        "income":           result.get("income"),
        "income_formatted": result.get("income_formatted", "Not found"),
        "employment_type":  result.get("employment_type", "unknown"),
        "document_type":    result.get("document_type", "unknown"),
        "full_text":        result.get("full_text", ""),
    })

    return UploadResponse(
        customer_id=     request.customer_id,
        document_type=   result.get("document_type", "unknown"),
        employment_type= result.get("employment_type", "unknown"),
        income=          result.get("income"),
        income_formatted=result.get("income_formatted", "Not found"),
        document_context=document_context,
        message=         "Document processed. Pass document_context to /chat."
    )

# Add at bottom:
from config import API_HOST, API_PORT

if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.app:app",
        "--host", API_HOST,
        "--port", str(API_PORT)
    ])