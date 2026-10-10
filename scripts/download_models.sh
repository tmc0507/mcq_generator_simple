#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
BASE="https://github.com/tmc0507/mcq_generator_simple/releases/download/models-v1"
mkdir -p "$MODELS"

case "${1:-all}" in
  all) names=(modelv1 modelv2 modelv3) ;;
  modelv1|modelv2|modelv3) names=("$1") ;;
  *) echo "Usage: $0 [all|modelv1|modelv2|modelv3]" >&2; exit 2 ;;
esac

for name in "${names[@]}"; do
  case "$name" in
    modelv1)
      hash=d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5
      parts=("$name.gguf") ;;
    modelv2)
      hash=d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
      parts=("$name.gguf.part00" "$name.gguf.part01" "$name.gguf.part02") ;;
    modelv3)
      hash=7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5
      parts=("$name.gguf.part00" "$name.gguf.part01") ;;
  esac
  target="$MODELS/$name.gguf"
  if [[ -f "$target" ]] && [[ "$(sha256sum "$target" | cut -d' ' -f1)" == "$hash" ]]; then
    echo "$name already verified"
    continue
  fi
  temp="$(mktemp -d "$MODELS/.download.XXXXXX")"
  assembled="$temp/assembled.gguf"
  : > "$assembled"
  for part in "${parts[@]}"; do
    echo "Downloading $part"
    curl -fL --retry 8 --retry-all-errors --retry-delay 2 --continue-at - \
      --silent --show-error "$BASE/$part" -o "$temp/$part"
    cat "$temp/$part" >> "$assembled"
    rm "$temp/$part"
  done
  echo "$hash  $assembled" | sha256sum --check --status || {
    echo "Checksum mismatch: $name" >&2
    rm -rf "$temp"
    exit 1
  }
  mv "$assembled" "$target"
  rmdir "$temp"
  echo "Saved $target"
done
