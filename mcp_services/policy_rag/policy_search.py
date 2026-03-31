import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from sentence_transformers import SentenceTransformer

from mcp_services.policy_rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    N_RESULTS,
)

# ── Policy reminder appended to every response ────────────────────────────────
POLICY_REMINDER = (
    "\nNOTE: Mandatory thresholds in GR-001 cannot be overridden "
    "under any circumstance. Apply all policies strictly.\n"
)

# ── Global variables (initialized on first call) ─────────────────────────────
_model = None
_chroma_client = None
_collection = None

def _initialize_resources():
    """Initialize the embedding model and ChromaDB connection."""
    global _model, _chroma_client, _collection

    if _model is None:
        print("Policy RAG: Loading embedding model...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Policy RAG: Model loaded")

    if _chroma_client is None:
        print("Policy RAG: Connecting to ChromaDB...")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None
        )
        print(f"Policy RAG: Connected. Collection has {_collection.count()} chunks")

def query_policies_direct(query: str) -> str:
    """
    Direct policy search function that bypasses MCP protocol.
    Used as fallback when MCP server has issues.
    """
    _initialize_resources()

    if not query or not query.strip():
        return "Error: query cannot be empty"

    # Convert query to vector
    query_embedding = _model.encode(query).tolist()

    # Search Chroma
    try:
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS
        )
    except Exception as e:
        print(f"[ERROR] Chroma query failed: {e}")
        return f"Error querying policies: {str(e)}"

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return "No relevant policies found for this query."

    # Format results
    formatted = []
    formatted.append(f"POLICY SEARCH RESULTS FOR: '{query}'\n")
    formatted.append("=" * 60)

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        score = distances[i-1] if distances else 0.0
        formatted.append(f"\n[Result {i}]")
        formatted.append(f"Policy Code:      {meta['policy_code']}")
        formatted.append(f"Source:           {meta['source']}")
        formatted.append(f"Similarity Score: {score:.4f}  (lower = more relevant)")
        formatted.append(f"Content:\n{doc}")
        formatted.append("-" * 40)

    # Append policy reminder
    formatted.append(POLICY_REMINDER)

    response_text = "\n".join(formatted)

    # ── AIRS intercepts here ──────────────────────────────────────────────
    print(f"[TOOL CALLED] query_policies_direct")
    print(f"[QUERY]       {query}")
    print(f"[RESULTS]     {[m['policy_code'] for m in metadatas]}")
    print(f"[SCORES]      {[f'{d:.4f}' for d in distances]}")
    print()

    return response_text