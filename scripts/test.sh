#!/usr/bin/env bash
# Default: 50k images at eight Euler steps; pass --num-samples 64 --skip-metrics for a preview.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python test.py --output-dir "${OUTPUT_DIR:-samples/test}" "$@"
