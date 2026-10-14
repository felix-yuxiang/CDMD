"""Generate images and compute Euler FID/IS metrics for checkpoint-best."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from sampler import SAMPLING_STEPS
from utils.evaluation import evaluate, load_vae_and_stats, validate_metric_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", default="cache/consistentdmd-ldit-best.pt")
    parser.add_argument("--sampling-steps", type=int, choices=SAMPLING_STEPS, default=8)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--skip-metrics", action="store_true", help="Keep generated PNGs without FID/IS")
    parser.add_argument("--keep-samples", action="store_true", help="Keep scored PNGs after successful FID/IS")
    parser.add_argument("--vae-checkpoint", default="cache/vavae-imagenet256-f16d32-dinov2.pt")
    parser.add_argument("--latent-stats-path", default="cache/latents_stats.pt")
    parser.add_argument("--ref-stats-path", default="cache/fid-50k-256.npz")
    args = parser.parse_args()
    if min(args.num_samples, args.batch_size, args.decode_batch_size) <= 0:
        parser.error("Sample and batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The released checkpoint requires a BF16-capable CUDA device")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The recorded evaluation requires BF16 support")
    device = torch.device("cuda", local_rank)
    if not args.skip_metrics:
        validate_metric_backend(args.ref_stats_path)
    for asset in (args.ckpt_path, args.vae_checkpoint, args.latent_stats_path):
        if not Path(asset).is_file():
            raise FileNotFoundError(f"Missing asset: {asset}; see README setup")
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
    try:
        from utils.checkpoint import load_student, public_metadata

        student, metadata = load_student(args.ckpt_path, device=device)
        vae, mean, std = load_vae_and_stats(args.vae_checkpoint, args.latent_stats_path, device)
        result = evaluate(
            student, vae, mean, std, output_dir=args.output_dir,
            num_samples=args.num_samples, batch_size=args.batch_size,
            decode_batch_size=args.decode_batch_size, steps=args.sampling_steps,
            skip_metrics=args.skip_metrics, keep_samples=args.keep_samples,
            ref_stats_path=args.ref_stats_path,
            ema_metadata=metadata.get("evaluation_ema", metadata.get("ema")),
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            result["checkpoint"] = public_metadata(metadata)
            result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(device)}
            (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
