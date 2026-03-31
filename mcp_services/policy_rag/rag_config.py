# mcp_services/policy_rag/config.py
# Keep as standalone — do NOT import from root config
# This avoids circular import issues

POLICIES_DIR    = "mcp_services/policy_rag/data"
CHROMA_DIR      = "rag/chroma_store"
COLLECTION_NAME = "bank_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_RESULTS       = 5