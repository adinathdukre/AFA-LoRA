#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-afa-lora}"
docker build "$SCRIPT_DIR" --platform=linux/amd64 --tag "$DOCKER_IMAGE_TAG" "$@"
