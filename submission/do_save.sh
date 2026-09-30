#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-afa-lora}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../artifacts}"
STAMP="$(date +%Y%m%d%H%M%S)"
mkdir -p "$OUT_DIR"

"$SCRIPT_DIR/do_build.sh"
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$OUT_DIR/${DOCKER_IMAGE_TAG}_${STAMP}.tar.gz"
tar -czf "$OUT_DIR/model_${STAMP}.tar.gz" -C "$MODEL_DIR" .
ls -lh "$OUT_DIR"
