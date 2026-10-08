#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null || \
   ! command -v pdftotext >/dev/null || ! command -v unzip >/dev/null; then
  apt-get update
  apt-get install -y build-essential cmake poppler-utils unzip
fi

python3 -m pip install -r requirements-kaggle.txt
