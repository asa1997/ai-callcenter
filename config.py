# ── AI Call Center Agent — Central Configuration ──────────────────────────────
# Single source of truth for all ports, URLs, and settings
# Change values here — everything else picks them up automatically

# ── MCP Server Ports ──────────────────────────────────────────────────────────
POLICY_SERVER_PORT   = 8001
CUSTOMER_SERVER_PORT = 8002   # coming soon
RISK_SERVER_PORT     = 8003   # coming soon
DOCUMENT_SERVER_PORT = 8004

# ── API Server ────────────────────────────────────────────────────────────────
API_PORT = 8000
API_HOST = "0.0.0.0"

# ── MCP Server URLs ───────────────────────────────────────────────────────────
POLICY_SERVER_URL   = f"http://localhost:{POLICY_SERVER_PORT}/sse"
CUSTOMER_SERVER_URL = f"http://localhost:{CUSTOMER_SERVER_PORT}/sse"
RISK_SERVER_URL     = f"http://localhost:{RISK_SERVER_PORT}/sse"
DOCUMENT_SERVER_URL = f"http://localhost:{DOCUMENT_SERVER_PORT}/sse"

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# ── RAG / Chroma ──────────────────────────────────────────────────────────────
CHROMA_DIR      = "rag/chroma_store"
COLLECTION_NAME = "bank_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_RESULTS       = 5

# ── Document Processing ───────────────────────────────────────────────────────
DOCUMENT_MODEL_PATH = "mcp_services/document_processor/model/distilbert.onnx"
DOCUMENT_MODEL_NAME = "distilbert-base-uncased"
DOCUMENT_MAX_LENGTH = 128
