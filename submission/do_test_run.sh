#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-afa-lora}"
INPUT_DIR="${INPUT_DIR:-$SCRIPT_DIR/test/input}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
OUTPUT_DIR="$SCRIPT_DIR/test/output"

"$SCRIPT_DIR/do_build.sh"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
chmod o+rwx "$OUTPUT_DIR"
docker run --rm --platform=linux/amd64 --network none --gpus all \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$MODEL_DIR":/opt/ml/model:ro \
    "$DOCKER_IMAGE_TAG"
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(len(s), 'scores, range', min(s), max(s))" \
    "$OUTPUT_DIR/stacked-neoplastic-lesion-likelihoods.json"
