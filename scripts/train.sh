#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec torchrun --standalone --nnodes=1 --nproc_per_node="${NGPUS:-8}" train.py "$@"
