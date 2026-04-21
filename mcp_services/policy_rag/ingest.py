import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import re
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
from mcp_services.policy_rag.rag_config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    N_RESULTS,
    POLICIES_DIR,
)


def chunk_policy_file(filepath: str) -> list[dict]:
    """Split one policy file into chunks by policy code (SA-001 etc.)."""
    with open(filepath, "r") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    chunks = []
    pattern = r'(?=^[A-Z]{2}-\d{3}:)'
    sections = re.split(pattern, content, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section or section.startswith("==="):
            continue
        code_match = re.match(r'^([A-Z]{2}-\d{3}):', section)
        policy_code = code_match.group(1) if code_match else "UNKNOWN"
        chunks.append({
            "text": section,
            "source": filename,
            "policy_code": policy_code,
        })
    return chunks


def create_chroma_collection(chroma_dir: str, collection_name: str):
    """Create a fresh Chroma collection, deleting any existing one."""
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_policies(
    policies_dir: str = POLICIES_DIR,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
):
    """Ingest all policy files into Chroma. Returns model, collection, chunks."""
    model = SentenceTransformer(embedding_model)
    collection = create_chroma_collection(chroma_dir, collection_name)

    policy_files = sorted(
        f for f in os.listdir(policies_dir) if f.endswith(".txt")
    )

    all_chunks = []
    for filename in policy_files:
        filepath = os.path.join(policies_dir, filename)
        chunks = chunk_policy_file(filepath)
        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    texts    = [c["text"]         for c in all_chunks]
    sources  = [c["source"]       for c in all_chunks]
    codes    = [c["policy_code"]  for c in all_chunks]
    ids      = [f"{codes[i]}_{i}" for i in range(len(all_chunks))]

    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": sources[i], "policy_code": codes[i]}
            for i in range(len(all_chunks))
        ],
    )
    return model, collection, all_chunks


def query_collection(
    collection,
    model,
    query_text: str,
    n_results: int = 5
):
    """
    Query Chroma for relevant policy chunks.
    Used by policy_server.py on every MCP tool call.
    """
    embedding = model.encode(query_text).tolist()
    return collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )


def verify_collection(collection, model, query_text: str, n_results: int = 5):
    """Wrapper for testing — calls query_collection."""
    return query_collection(collection, model, query_text, n_results)


if __name__ == "__main__":
    print("Loading embedding model...")
    model, collection, all_chunks = ingest_policies()
    print(f"\nTotal chunks stored: {collection.count()}")

    print("\nRunning verification query...")
    results = verify_collection(
        collection, model,
        query_text="credit score for personal loan"
    )

    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        print(f"\nResult {i+1}: {meta['policy_code']} ({meta['source']})")
        print(f"  {doc[:150]}...")

    print("\n✅ Ingestion complete. Chroma is ready.")