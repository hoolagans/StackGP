#!/usr/bin/env bash
# Build frontend and start production backend
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== StackGP GUI — Production Build ==="

# Build frontend
echo "Building frontend..."
cd "$SCRIPT_DIR/../frontend"
npm run build

echo ""
echo "Starting production server (FastAPI serves built assets)..."
cd "$ROOT_DIR"
python -m uvicorn gui.backend.main:app --host 0.0.0.0 --port 8000
