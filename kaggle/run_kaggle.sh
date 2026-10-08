#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
DOCUMENT="${1:-input/Slides CHAPTER - SVM.pdf}"
MODEL="${2:-Qwen/Qwen3-8B}"
QUESTIONS="${3:-10}"

mkdir -p build work output
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/doc_processor "$DOCUMENT" \
  --output work/chunks.json \
  --blocks-per-chunk 3 \
  --chunk-chars 12000

python3 scripts/generate_mcq.py \
  --backend transformers \
  --chunks work/chunks.json \
  --model "$MODEL" \
  --questions "$QUESTIONS" \
  --output output/questions.json
