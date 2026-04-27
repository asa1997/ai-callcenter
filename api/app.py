import os
import io
import json
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from agent.agent import handle_message
from agent.mcp_client import extract_document_info

app = FastAPI(
    title="AI Call Center Agent",
    description="Palo Alto AIRS COE Lab — Vulnerable Baseline System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class UploadResponse(BaseModel):
    customer_id:      str
    document_type:    str
    employment_type:  str
    income:           int | None
    income_formatted: str
    document_context: str    # pass this to /chat as document_context
    filename:         str
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


ALLOWED_EXTENSIONS = {".txt", ".pdf", ".csv", ".json"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text content from an uploaded file."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages).strip()
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse PDF: {e}"
            )

    # text-based files
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


@app.post("/upload", response_model=UploadResponse)
async def upload(
    customer_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Document upload endpoint.

    Customer uploads a file (PDF, TXT, CSV, JSON).
    Calls Document Processing MCP server (port 8004).
    DistilBERT ONNX model extracts structured fields.
    Returns document_context — pass this to /chat.

    Intentionally vulnerable:
    - No input sanitisation
    - full_text returned unfiltered
    - Threat 3: injection in document flows to agent
    - AIRS guardrail intercepts at MCP output in secured system
    """
    if not customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10 MB.")
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    document_text = _extract_text_from_file(file_bytes, file.filename)
    if not document_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    # Call document MCP server via mcp_client
    raw_result = extract_document_info(document_text, customer_id)

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
        customer_id=     customer_id,
        document_type=   result.get("document_type", "unknown"),
        employment_type= result.get("employment_type", "unknown"),
        income=          result.get("income"),
        income_formatted=result.get("income_formatted", "Not found"),
        document_context=document_context,
        filename=        file.filename,
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