#!/usr/bin/env bash
# Start the StackGP GUI (backend + frontend dev server)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== StackGP Data Modeling Studio ==="
echo ""

# Backend
echo "Starting backend (FastAPI)..."
cd "$ROOT_DIR"
python -m uvicorn gui.backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID (http://localhost:8000)"
echo ""

# Frontend dev server
echo "Starting frontend (Vite)..."
cd "$SCRIPT_DIR/../frontend"
npm run dev -- --port 5173 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Open: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all servers"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Servers stopped.'" EXIT
wait
