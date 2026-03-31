# AI Call Center Agent 

Vulnerable baseline AI call center system demonstrating
real-world AI security threats 

## Architecture
- AI Agent (Ollama llama3.1)
- 4 MCP Services (Policy RAG, Customer Profile, Credit Risk, Document Processing)
- FastAPI REST API
- DistilBERT ONNX model for document processing

## Security Threats Demonstrated
- OWASP LLM02 — Sensitive Information Disclosure
- OWASP LLM06 — Excessive Agency
- OWASP LLM08 — Vector and Embedding Weaknesses
- OWASP LLM03 — Supply Chain Vulnerabilities (model scanning)

## Quick Start

### Prerequisites
- Python 3.11
- Ollama with llama3.1 model
- 8GB RAM minimum

### Setup
```bash
# Clone the repo
git clone <your-repo-url>
cd ai-callcenter

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and ingest policies into Chroma
python mcp_services/policy_rag/ingest.py

# Export DistilBERT model to ONNX
python mcp_services/document_processor/export_model.py

# Start all services
bash scripts/start_all.sh
```

### Test
```bash
# Normal query
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"12345","message":"Am I eligible for a personal loan?"}' \
  | python -m json.tool

# See DEMO_SCENARIOS.txt for full attack scenarios
```

## AIRS Integration Points
- Model Scanning: mcp_services/document_processor/model/distilbert.onnx
- Red Teaming: POST /chat and POST /upload
- API Docs: http://localhost:8000/docs
