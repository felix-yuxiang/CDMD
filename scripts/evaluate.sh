#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output_root="${OUTPUT_DIR:-samples/euler-best}"
for steps in 1 2 4 8 16 32 64; do
    if [[ -e "$output_root/steps-$steps" ]]; then
        printf 'Output already exists: %s; choose a fresh OUTPUT_DIR.\n' "$output_root/steps-$steps" >&2
        exit 2
    fi
done
for steps in 1 2 4 8 16 32 64; do
    python -m torch.distributed.run --standalone --nproc_per_node "${NGPUS:-2}" \
        test.py --sampling-steps "$steps" --num-samples 50000 \
        --batch-size 32 --decode-batch-size 32 --output-dir "$output_root/steps-$steps" "$@"
done
