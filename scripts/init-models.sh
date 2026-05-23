#!/usr/bin/env bash
# init-models.sh — run once as an init container under --profile full.
# Downloads Jina GGUF models into /models and warms the Ollama model cache.
# Safe to re-run: skips files that already exist.
#
# Volumes expected:
#   /models      — nexus_models (shared with embedder + reranker)
#   (Ollama cache managed separately via OLLAMA_BASE_URL env var)

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
mkdir -p "$MODELS_DIR"

# ── Jina Embedder ────────────────────────────────────────────────────────────
EMBEDDER_FILE="$MODELS_DIR/jina-embeddings-v4.Q4_K_M.gguf"
EMBEDDER_URL="https://huggingface.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF/resolve/main/jina-embeddings-v4-text-retrieval-Q4_K_M.gguf"

if [ ! -f "$EMBEDDER_FILE" ]; then
  echo "[init-models] Downloading Jina Embeddings v4 Q4_K_M (~2.0 GB)…"
  curl -L --progress-bar -o "$EMBEDDER_FILE" "$EMBEDDER_URL"
  echo "[init-models] Embedder downloaded."
else
  echo "[init-models] Embedder already present — skipping."
fi

# ── Jina Reranker ─────────────────────────────────────────────────────────────
RERANKER_FILE="$MODELS_DIR/jina-reranker-v3.Q4_K_M.gguf"
RERANKER_URL="https://huggingface.co/jinaai/jina-reranker-v3-GGUF/resolve/main/jina-reranker-v3-Q4_K_M.gguf"

if [ ! -f "$RERANKER_FILE" ]; then
  echo "[init-models] Downloading Jina Reranker v3 Q4_K_M (~0.4 GB)…"
  curl -L --progress-bar -o "$RERANKER_FILE" "$RERANKER_URL"
  echo "[init-models] Reranker downloaded."
else
  echo "[init-models] Reranker already present — skipping."
fi

# ── Ollama warm-up ────────────────────────────────────────────────────────────
# Pull the light LLM into the Ollama container's cache so first inference
# isn't blocked by a download. OLLAMA_BASE_URL is set by docker-compose.
LIGHT_LLM_MODEL="${LIGHT_LLM_MODEL:-qwen2.5:3b}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"

echo "[init-models] Waiting for Ollama to be ready…"
for i in $(seq 1 60); do
  if curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

if curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
  if curl -sf "$OLLAMA_BASE_URL/api/tags" | grep -q "$LIGHT_LLM_MODEL"; then
    echo "[init-models] $LIGHT_LLM_MODEL already pulled — skipping."
  else
    echo "[init-models] Pulling $LIGHT_LLM_MODEL into Ollama (~2.2 GB)…"
    curl -sf -X POST "$OLLAMA_BASE_URL/api/pull" \
      -H 'Content-Type: application/json' \
      -d "{\"name\": \"$LIGHT_LLM_MODEL\", \"stream\": false}"
    echo "[init-models] $LIGHT_LLM_MODEL ready."
  fi
else
  echo "[init-models] WARNING: Ollama not reachable at $OLLAMA_BASE_URL — skipping model pull."
fi

echo "[init-models] All models ready."
