#!/bin/bash
# =============================================================================
# dLLM MCP Server — RunPod Setup Script
# =============================================================================
# Run this ONCE after spinning up your RunPod RTX 4090 pod.
# Everything from repo clone → dependencies → server start.
#
# Usage (on RunPod terminal):
#   bash setup_runpod.sh
#
# Time to complete: ~5-8 minutes (mainly downloading PyTorch + dLLM)
# Cost at $0.39/hr: ~$0.05
# =============================================================================

set -e  # Exit on any error

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         dLLM MCP Server — RunPod Setup               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check GPU ─────────────────────────────────────────────────────────
echo "🔍 Step 1/6: Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Step 2: Install conda env ─────────────────────────────────────────────────
echo "🐍 Step 2/6: Setting up Python environment..."
conda create -n dllm python=3.10 -y 2>/dev/null || echo "Env already exists, skipping"
source activate dllm || conda activate dllm

# ── Step 3: Install PyTorch with CUDA ─────────────────────────────────────────
echo "🔥 Step 3/6: Installing PyTorch with CUDA 12.4..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 -q
echo "✅ PyTorch installed"

# ── Step 4: Clone dLLM repo and install ───────────────────────────────────────
echo "📦 Step 4/6: Installing dLLM framework..."
if [ ! -d "dllm" ]; then
    git clone https://github.com/ZHZisZZ/dllm.git
fi
cd dllm
pip install -e . -q
cd ..
echo "✅ dLLM installed"

# ── Step 5: Install MCP server dependencies ───────────────────────────────────
echo "🔌 Step 5/6: Installing MCP server dependencies..."
pip install "mcp[cli]>=1.0.0" pydantic>=2.0.0 -q
echo "✅ MCP dependencies installed"

# ── Step 6: Start the server ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ Setup complete! Starting dLLM MCP Server...      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Server starting on HTTP transport at port 8000"
echo "Connect your MCP client to: http://<YOUR_RUNPOD_IP>:8000/mcp"
echo ""
echo "To test locally, in another terminal run:"
echo "  npx @modelcontextprotocol/inspector http://localhost:8000/mcp"
echo ""

python src/server.py http
