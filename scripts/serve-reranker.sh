#!/usr/bin/env bash
# Serve Jina Reranker v3 locally via llama.cpp (Apple Silicon / Metal).
#
# Prereq:
#   brew install llama.cpp
#   # Download a Jina Reranker v3 GGUF into models/.
#
# Listens on $RERANKER_PORT (default 8081). Endpoint:
#   POST /reranking   { "query": "...", "documents": [...] }

set -euo pipefail

MODEL_PATH="${RERANKER_MODEL:-models/jina-reranker-v3.Q4_K_M.gguf}"
PORT="${RERANKER_PORT:-8081}"
CTX_SIZE="${RERANKER_CTX:-8192}"
DEVICE="${RERANKER_DEVICE:-auto}" # auto | metal | gpu | cpu

if ! command -v llama-server >/dev/null 2>&1; then
  echo "ERROR: llama-server not found. Install via: brew install llama.cpp" >&2
  exit 127
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: model not found at $MODEL_PATH" >&2
  echo "Download a Jina Reranker v3 GGUF into models/ first." >&2
  exit 1
fi

gpu_layers="${RERANKER_GPU_LAYERS:-}"
if [ -z "$gpu_layers" ]; then
  case "$DEVICE" in
    auto)
      if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        gpu_layers=999
        DEVICE="metal"
      elif command -v nvidia-smi >/dev/null 2>&1; then
        gpu_layers=999
        DEVICE="gpu"
      else
        gpu_layers=0
        DEVICE="cpu"
      fi
      ;;
    metal|gpu)
      gpu_layers=999
      ;;
    cpu)
      gpu_layers=0
      ;;
    *)
      echo "ERROR: RERANKER_DEVICE must be one of: auto, metal, gpu, cpu" >&2
      exit 2
      ;;
  esac
fi

echo "Starting reranker on :$PORT (model=$MODEL_PATH, device=$DEVICE, gpu_layers=$gpu_layers)"
exec llama-server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --reranking \
  --n-gpu-layers "$gpu_layers"
