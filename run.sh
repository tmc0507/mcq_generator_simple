#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: ./run.sh <document> <model> [questions] [llama_cpp|transformers]"
  exit 1
fi

DOC="$1"
MODEL="$2"
QUESTIONS="${3:-20}"
BACKEND="${4:-llama_cpp}"

if [[ "$BACKEND" != "llama_cpp" && "$BACKEND" != "transformers" ]]; then
  echo "Unknown backend: $BACKEND" >&2
  exit 1
fi

mkdir -p build work output
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j >/dev/null

./build/doc_processor "$DOC" \
  --output work/chunks.json \
  --blocks-per-chunk 3 \
  --chunk-chars 12000

ARGS=(--backend "$BACKEND" --chunks work/chunks.json --model "$MODEL"
      --questions "$QUESTIONS" --output output/questions.json)
if [[ "$BACKEND" == "llama_cpp" ]]; then
  ARGS+=(--n-ctx 8192 --n-gpu-layers 20)
fi
python3 scripts/generate_mcq.py "${ARGS[@]}"
