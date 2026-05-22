#!/usr/bin/env bash
set -e

# Define directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

echo "Downloading Jina Embeddings v4 (Text Retrieval) Q4_K_M..."
EMBEDDER_URL="https://huggingface.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF/resolve/main/jina-embeddings-v4-text-retrieval-Q4_K_M.gguf"
EMBEDDER_DEST="$MODELS_DIR/jina-embeddings-v4.Q4_K_M.gguf"

if [ ! -f "$EMBEDDER_DEST" ]; then
    curl -L --progress-bar -o "$EMBEDDER_DEST" "$EMBEDDER_URL"
    echo "Embedder downloaded successfully."
else
    echo "Embedder already exists at $EMBEDDER_DEST"
fi

echo "----------------------------------------"

echo "Downloading Jina Reranker v3 Q4_K_M..."
RERANKER_URL="https://huggingface.co/jinaai/jina-reranker-v3-GGUF/resolve/main/jina-reranker-v3-Q4_K_M.gguf"
RERANKER_DEST="$MODELS_DIR/jina-reranker-v3.Q4_K_M.gguf"

if [ ! -f "$RERANKER_DEST" ]; then
    curl -L --progress-bar -o "$RERANKER_DEST" "$RERANKER_URL"
    echo "Reranker downloaded successfully."
else
    echo "Reranker already exists at $RERANKER_DEST"
fi

echo "All models downloaded to $MODELS_DIR"
