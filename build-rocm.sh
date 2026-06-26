#!/bin/bash
# Build (and optionally push) the Scriberr ROCm image for AMD gfx1151 GPUs.
set -e

IMAGE=${1:-ghcr.io/lucasmcoleman/scriberr:rocm}

echo "🏗️  Building $IMAGE from Dockerfile.rocm ..."
docker build -f Dockerfile.rocm -t "$IMAGE" .

echo "✅ Built $IMAGE"
echo "▶️  Run:  docker compose -f docker-compose.rocm.yml up -d"
echo "⬆️  Push: docker push $IMAGE"
