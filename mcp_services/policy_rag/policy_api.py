from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp_services.policy_rag.policy_search import query_policies_direct

app = FastAPI(
    title="Policy RAG API",
    description="Direct API for policy document search",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    results: str

@app.post("/query", response_model=QueryResponse)
def query_policies(request: QueryRequest):
    """
    Query the policy documents using semantic search.
    Returns relevant policy chunks for the given query.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = query_policies_direct(request.query)
        return QueryResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/")
def root():
    return {
        "service": "Policy RAG API",
        "status": "running",
        "endpoints": {
            "POST /query": "Query policy documents"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)