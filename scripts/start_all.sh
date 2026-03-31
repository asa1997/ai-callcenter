#!/bin/bash
# ── AI Call Center Agent — Start All Services ─────────────────────────────────
# Run from project root: bash scripts/start_all.sh

# Always run from project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT

echo "=================================================="
echo " AI Call Center Agent — Palo Alto AIRS COE Lab"
echo "=================================================="
echo ""

# Check venv is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "ERROR: venv not active. Run: source venv/bin/activate"
    exit 1
fi

# Check Ollama is running
if ! curl -s http://localhost:11434 > /dev/null; then
    echo "ERROR: Ollama not running. Run: ollama serve"
    exit 1
fi

# ── Kill any existing processes on our ports ──────────────────────────────────
# ── Kill existing processes ───────────────────────────────────────────────────
echo "Cleaning up existing processes..."
for port in 8000 8001 8002 8003 8004; do
    pid=$(lsof -t -i :$port)
    if [ ! -z "$pid" ]; then
        kill -9 $pid 2>/dev/null
        echo "  Killed process $pid on port $port"
        # Wait until port is actually free
        while lsof -t -i :$port > /dev/null 2>&1; do
            sleep 0.5
        done
        echo "  Port $port confirmed free"
    else
        echo "  Port $port already free"
    fi
done
sleep 1
echo ""


# ── Clear Python cache ────────────────────────────────────────────────────────
echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "  ✅ Cache cleared"
echo ""

# ── Start services ────────────────────────────────────────────────────────────

echo "Starting Document Processing Server (port 8004)..."
python mcp_services/document_processor/document_server.py &
sleep 3

echo "Starting Policy RAG MCP Server     (port 8001)..."
python mcp_services/policy_rag/policy_server.py &
sleep 6

echo "Starting Customer Profile Server   (port 8002)..."
python mcp_services/customer_profile/customer_server.py &
sleep 3

echo "Starting Credit Risk Server        (port 8003)..."
python mcp_services/credit_risk/risk_server.py &
sleep 3

echo "Starting API Server                 (port 8000)..."
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 &
sleep 2

# ── Verify all services are up ────────────────────────────────────────────────
# ── Verify all services are up ────────────────────────────────────────────────
echo ""
echo "Verifying services..."

# Policy server needs extra time — retry up to 5 times
for i in {1..5}; do
    curl -s http://localhost:8001/sse > /dev/null && break
    echo "  Waiting for Policy Server... ($i/5)"
    sleep 2
done
curl -s http://localhost:8001/sse > /dev/null && echo "  ✅ Policy Server    (8001) — UP" || echo "  ❌ Policy Server    (8001) — FAILED"
curl -s http://localhost:8004/sse > /dev/null && echo "  ✅ Document Server  (8004) — UP" || echo "  ❌ Document Server  (8004) — FAILED"
curl -s http://localhost:8000/    > /dev/null && echo "  ✅ API Server       (8000) — UP" || echo "  ❌ API Server       (8000) — FAILED"
echo ""
echo "=================================================="
echo " System ready. Press Ctrl+C to stop all services."
echo "=================================================="
echo ""

wait
