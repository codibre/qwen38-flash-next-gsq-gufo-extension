#!/usr/bin/env bash
# Build Gufo with the qwen38_flash_next_gsq extension.
#
# The pinned GUFO_REF carries the core GGUF loader alignment hunk from
# gufo-org/gufo#939 that extensions cannot replace. Once that hunk lands on
# gufo main, set:  GUFO_REPO=https://github.com/gufo-org/gufo.git GUFO_REF=main
set -euo pipefail

GUFO_REPO=${GUFO_REPO:-https://github.com/Farenheith/gufo.git}
GUFO_REF=${GUFO_REF:-59f94a4a}
GUFO_DIR=${GUFO_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/build-gufo"}
EXT_DIR=$(cd "$(dirname "$0")/.." && pwd)
GUFO_EXTENSIONS_LIST=${GUFO_EXTENSIONS_LIST:-qwen38_flash_next_gsq}
PRESET=${PRESET:-release}

if [[ -d "$GUFO_DIR/.git" ]]; then
  git -C "$GUFO_DIR" fetch --quiet "$GUFO_REPO" "$GUFO_REF"
  git -C "$GUFO_DIR" checkout --quiet FETCH_HEAD
else
  git clone --quiet "$GUFO_REPO" "$GUFO_DIR"
  git -C "$GUFO_DIR" checkout --quiet "$GUFO_REF"
fi

# Explicit mode (-DGUFO_EXTENSIONS) alone excludes the in-tree
# qwen38_flash_next from the shipped set; never edit extensions.lock —
# the Nix release package derives mkServe options from it.
cd "$GUFO_DIR"

cmake --preset "$PRESET" \
  -DGUFO_EXTENSION_DIRS="$EXT_DIR" \
  -DGUFO_EXTENSIONS="$GUFO_EXTENSIONS_LIST"
cmake --build --preset "$PRESET" --target gufo --parallel "${JOBS:-4}"

echo
"$GUFO_DIR/build/$PRESET/gufo" diagnose --section extensions || true
echo
echo "Try: $GUFO_DIR/build/$PRESET/gufo chat --model <gsq-coder shard1>.gguf"
