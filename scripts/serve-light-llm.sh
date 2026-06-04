#!/usr/bin/env bash
# Ensure Ollama is running and the light LLM model is pulled.
# Used by: contextual chunk enrichment.
#
# Prereq: install Ollama (brew install ollama, or https://ollama.com).

set -euo pipefail

MODEL="${LIGHT_LLM_MODEL:-qwen2.5:3b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama not found. Install via: brew install ollama" >&2
  exit 127
fi

# Start ollama in the background if it isn't already
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting ollama server in background..."
  nohup ollama serve >/tmp/nexus-ollama.log 2>&1 &
  for i in $(seq 1 30); do
    sleep 0.5
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
      break
    fi
  done
fi

# NOTE: avoid `! cmd | grep` under set -euo pipefail — grep exit 1 (no match)
# propagates through the pipe and aborts the script before we can act on it.
# Use a subshell + explicit exit-code capture instead.
if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qFx "$MODEL" 2>/dev/null; then
  echo "Pulling $MODEL..."
  ollama pull "$MODEL"
fi

echo "Ollama ready at http://localhost:11434 (model: $MODEL)"
