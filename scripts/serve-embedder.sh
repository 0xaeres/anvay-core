#!/usr/bin/env bash
# Serve Jina Embeddings v4 locally via llama.cpp (Apple Silicon / Metal).
#
# Prereq:
#   brew install llama.cpp
#   mkdir -p models
#   # Download a Jina v4 GGUF into models/, e.g.:
#   #   huggingface-cli download <jina-v4-gguf-repo> jina-embeddings-v4.Q4_K_M.gguf --local-dir models/
#
# Listens on $EMBEDDER_PORT (default 8080). Endpoint:
#   POST /v1/embeddings   { "input": [...] }  -> OpenAI-compatible embeddings
#
# Task-LoRA dual mode is handled at the client layer (anvay/ingest/embedder.py)
# by prepending the appropriate instruction prefix per chunk type.

set -euo pipefail

MODEL_PATH="${EMBEDDER_MODEL:-models/jina-embeddings-v4.Q4_K_M.gguf}"
PORT="${EMBEDDER_PORT:-8080}"
CTX_SIZE="${EMBEDDER_CTX:-8192}"
BATCH_SIZE="${EMBEDDER_BATCH:-1024}"
UBATCH_SIZE="${EMBEDDER_UBATCH:-1024}"
DEVICE="${EMBEDDER_DEVICE:-auto}" # auto | metal | gpu | cpu

if ! command -v llama-server >/dev/null 2>&1; then
  echo "ERROR: llama-server not found. Install via: brew install llama.cpp" >&2
  exit 127
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: model not found at $MODEL_PATH" >&2
  echo "Download a Jina v4 GGUF into models/ first." >&2
  exit 1
fi

gpu_layers="${EMBEDDER_GPU_LAYERS:-}"
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
      echo "ERROR: EMBEDDER_DEVICE must be one of: auto, metal, gpu, cpu" >&2
      exit 2
      ;;
  esac
fi

echo "Starting embedder on :$PORT (model=$MODEL_PATH, device=$DEVICE, batch=$BATCH_SIZE, ubatch=$UBATCH_SIZE, gpu_layers=$gpu_layers)"
exec llama-server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --embedding \
  --pooling mean \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH_SIZE" \
  --n-gpu-layers "$gpu_layers"
