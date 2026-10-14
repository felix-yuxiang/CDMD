"""Data-free ImageNet-256 training for the fixed checkpoint-best objective."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
import yaml

from losses import ConsistentDMDLoss, validate_m_ratio
from utils.ema import FACMEvaluationEMA, TrainingEMA


GLOBAL_BATCH_SIZE = 1024


def accumulation_steps(batch_size, world_size):
    if (type(batch_size) is not int or type(world_size) is not int
            or not 0 < batch_size <= GLOBAL_BATCH_SIZE
            or not 0 < world_size <= GLOBAL_BATCH_SIZE):
        raise ValueError("batch_size and world_size must be integers between 1 and 1024")
    steps = math.ceil(GLOBAL_BATCH_SIZE / (batch_size * world_size))
    if GLOBAL_BATCH_SIZE // world_size < steps:
        raise ValueError("Increase batch_size so every rank has a nonempty batch at each microstep")
    return steps


def rank_batch_sizes(batch_size, world_size, rank):
    """Split exactly 1024 examples, with the same number of forwards on each rank."""
    steps = accumulation_steps(batch_size, world_size)
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be an integer in [0, world_size)")
    remaining = GLOBAL_BATCH_SIZE // world_size + int(rank < GLOBAL_BATCH_SIZE % world_size)
    sizes = []
    for index in range(steps):
        # Reserve one example for every remaining forward, avoiding empty DDP batches.
        size = min(batch_size, remaining - (steps - index - 1))
        sizes.append(size)
        remaining -= size
    return tuple(sizes)


def load_config(path, overrides=None):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    config.setdefault("m_ratio", 0.5)
    expected = {
        "teacher_checkpoint", "vae_checkpoint", "latent_stats_path", "ref_stats_path",
        "output_dir", "train_steps", "batch_size", "global_batch_size", "seed",
        "attn_func", "bf16", "learning_rate", "eval_every", "save_every",
        "eval_samples", "eval_batch_size", "decode_batch_size", "m_ratio",
    }
    if set(config) != expected:
        raise ValueError(f"Config keys differ: missing={sorted(expected-set(config))}, unknown={sorted(set(config)-expected)}")
    config.update({key: value for key, value in (overrides or {}).items() if value is not None})
    config["m_ratio"] = validate_m_ratio(config["m_ratio"])
    if config["global_batch_size"] != GLOBAL_BATCH_SIZE:
        raise ValueError("The public training recipe fixes global_batch_size to 1024")
    for key in ("train_steps", "batch_size", "save_every", "eval_samples", "eval_batch_size", "decode_batch_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(config["eval_every"]) is not int or config["eval_every"] < 0:
        raise ValueError("eval_every must be a nonnegative integer (0 disables metrics)")
    if config["attn_func"] not in {"base", "fa2"}:
        raise ValueError("attn_func must be base or fa2")
    if type(config["bf16"]) is not bool or (config["attn_func"] != "base" and not config["bf16"]):
        raise ValueError("bf16 must be boolean; FlashAttention JVP requires bf16: true")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return config


def tracking_config(config, provenance):
    """Log the training recipe without local asset or checkpoint locations."""
    fields = ("train_steps", "batch_size", "global_batch_size", "seed", "attn_func", "bf16",
              "learning_rate", "eval_every", "save_every", "eval_samples",
              "eval_batch_size", "decode_batch_size", "m_ratio")
    result = {key: config[key] for key in fields}
    if provenance:
        result["continuation"] = {
            key: provenance[key] for key in
            ("source_step", "historical_global_batch_size", "global_batch_size")
            if type(provenance.get(key)) is int
        }
        # This link is explicitly supplied with --wandb-parent-run.
        parent = provenance.get("parent_wandb_run")
        if isinstance(parent, str) and len(parent.split("/")) == 3 and all(parent.split("/")):
            result["continuation"]["parent_wandb_run"] = parent
    return result


def start_tracker(args, output_dir, config, provenance):
    """Enable only the explicitly requested recipe, scalar and preview logging."""
    import os

    os.environ["WANDB_ERROR_REPORTING"] = "false"
    import wandb

    tracker_path = output_dir / "wandb_run.json"
    existing = json.loads(tracker_path.read_text()) if tracker_path.is_file() else None
    options = {"id": existing["id"], "resume": "must"} if existing else {}
    tracker = wandb.init(
        project=args.wandb_project, name=args.wandb_name, dir=str(output_dir),
        config=tracking_config(config, provenance),
        settings=wandb.Settings(console="off", host="", disable_code=True, disable_git=True,
                                x_disable_meta=True, x_disable_machine_info=True,
                                x_disable_stats=True),
        **options,
    )
    tracker_path.write_text(json.dumps({"id": tracker.id, "url": tracker.url,
                                       "parent_run": args.wandb_parent_run}, indent=2) + "\n")
    return tracker


def train_update(accelerator, student, objective, optimizer, scheduler,
                 training_ema, reporting_ema, *, batch_size, accumulation, step,
                 latent_shape=(32, 16, 16), num_classes=1000, local_batch_sizes=None):
    """One globally averaged update, retaining manual accumulation and EMA order."""
    sizes = (batch_size,) * accumulation if local_batch_sizes is None else tuple(local_batch_sizes)
    if len(sizes) != accumulation or not sizes or any(type(n) is not int or not 0 < n <= batch_size for n in sizes):
        raise ValueError("local_batch_sizes must contain one positive size per microstep, at most batch_size")
    world_size = accelerator.num_processes
    total_samples = batch_size * accumulation * world_size if local_batch_sizes is None else GLOBAL_BATCH_SIZE
    student.train()
    optimizer.zero_grad()
    finite_loss = torch.ones((), dtype=torch.bool, device=accelerator.device)
    scalar_sums = {}
    for micro_step, size in enumerate(sizes):
        noise = torch.randn(size, *latent_shape, device=accelerator.device)
        labels = torch.randint(0, num_classes, (size,), device=accelerator.device)
        context = accelerator.no_sync(student) if micro_step < accumulation - 1 else nullcontext()
        with context:
            with accelerator.autocast():
                loss, stats = objective(student, noise, labels)
            finite_loss.logical_and_(torch.isfinite(loss.detach()).all())
            # DDP averages rank gradients. This factor gives every example weight 1/1024.
            scaled_loss = (loss.mean() / accumulation if local_batch_sizes is None
                           else loss.mean() * (size * world_size / GLOBAL_BATCH_SIZE))
            accelerator.backward(scaled_loss)
        for key, value in stats.items():
            if torch.is_tensor(value) and value.ndim == 0:
                value = value.detach().to(torch.float64)
                if key.endswith("_rms"):
                    value = value.square()
                scalar_sums[key] = scalar_sums.get(key, 0) + value * size
    grad_norm = accelerator.clip_grad_norm_(student.parameters(), 1.0)
    grad_norm = torch.as_tensor(grad_norm, device=accelerator.device).detach()
    keys = sorted(scalar_sums)
    # One collective per update covers global diagnostics and failure agreement.
    packed = torch.stack([scalar_sums[key] for key in keys] + [
        finite_loss.to(torch.float64), torch.isfinite(grad_norm).to(torch.float64),
    ])
    if world_size > 1:
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    if not (packed[-2:] == world_size).all().item():
        optimizer.zero_grad()
        raise FloatingPointError("Nonfinite loss or gradients on at least one rank; optimizer and EMAs were not updated")
    for key, value in zip(keys, packed[:-2] / total_samples):
        stats[key] = value.sqrt() if key.endswith("_rms") else value
    optimizer.step()
    scheduler.step()
    training_ema.update(student)
    reporting_ema.update(training_step=step + 1)
    optimizer.zero_grad()
    return {**stats, "grad_norm": grad_norm}


def _root_action(accelerator, action):
    error = None
    if accelerator.is_main_process:
        try:
            action()
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
    if accelerator.num_processes > 1:
        payload = [error]
        torch.distributed.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error:
        raise RuntimeError(error)


def prepare_evaluation_directory(accelerator, path):
    """Preserve an earlier attempt when replaying evaluation after a checkpoint."""
    path = Path(path)

    def preserve_previous():
        if path.exists() and any(path.iterdir()):
            attempt = 1
            previous = path.with_name(f"{path.name}.previous_{attempt:03d}")
            while previous.exists() or previous.is_symlink():
                attempt += 1
                previous = path.with_name(f"{path.name}.previous_{attempt:03d}")
            path.rename(previous)

    _root_action(accelerator, preserve_previous)
    return path


def save_checkpoint(accelerator, training_ema, reporting_ema, output_dir,
                    *, step, config, best_fid, best_fid_step, provenance=None):
    """All ranks save RNG; a completed checkpoint becomes visible atomically."""
    root = Path(output_dir) / "checkpoints"
    final = root / f"step_{step:06d}"
    pending = root / f".step_{step:06d}.tmp"

    def prepare():
        root.mkdir(parents=True, exist_ok=True)
        if final.exists() or pending.exists():
            raise FileExistsError(f"Checkpoint destination already exists: {final} or {pending}")
        pending.mkdir()

    _root_action(accelerator, prepare)
    accelerator.wait_for_everyone()
    local_error = None
    try:
        # Calling this on every rank is essential: Accelerate writes one RNG file per rank.
        accelerator.save_state(str(pending))
    except Exception as error:
        local_error = f"rank {accelerator.process_index}: {type(error).__name__}: {error}"
    errors = [local_error]
    if accelerator.num_processes > 1:
        errors = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(errors, local_error)
    if any(errors):
        raise RuntimeError("Checkpoint save failed: " + " | ".join(error for error in errors if error))
    accelerator.wait_for_everyone()

    def finish():
        for rank in range(accelerator.num_processes):
            if not (pending / f"random_states_{rank}.pkl").is_file():
                raise RuntimeError(f"Checkpoint is missing rank {rank} RNG")
        torch.save(training_ema.ema_model.state_dict(), pending / "training_ema.pt")
        reporting_ema.save(pending, training_step=step)
        metadata = {
            "format": "consistentdmd_public_training_v1", "step": step,
            "world_size": accelerator.num_processes, "config": config,
            "ldit_model": "LightningDiT-XL/1", "dmf_depth": 10,
            "best_fid": best_fid, "best_fid_step": best_fid_step,
            "evaluation_protocol": {
                "backend": "facm_torch_fidelity_v1", "weights": "posthoc",
                "sigma_rel": 0.2, "seed": 0, "samples": config["eval_samples"],
                "guidance": 2.0, "bf16": True,
            },
            "evaluation_ema": reporting_ema.metadata(),
            "continuation": provenance,
        }
        (pending / "trainer_state.json").write_text(json.dumps(metadata, indent=2) + "\n")
        pending.rename(final)
        pointer = root / ".latest.json.tmp"
        pointer.write_text(json.dumps({"checkpoint": final.name, "step": step}) + "\n")
        pointer.replace(root / "latest.json")
        if best_fid_step == step:
            best_pointer = root / ".best.json.tmp"
            best_pointer.write_text(json.dumps({"checkpoint": final.name, "step": step, "fid": best_fid}) + "\n")
            best_pointer.replace(root / "best.json")
        # Full training states are large. Keep two latest states plus the best;
        # inspect the format marker before deleting any completed directory.
        candidates = []
        for path in root.glob("step_*"):
            state_file = path / "trainer_state.json"
            if path.is_symlink() or not path.is_dir() or not state_file.is_file():
                continue
            try:
                state = json.loads(state_file.read_text())
            except (OSError, ValueError):
                continue
            state_step = state.get("step")
            if (state.get("format") == "consistentdmd_public_training_v1"
                    and type(state_step) is int and path.name == f"step_{state_step:06d}"):
                candidates.append((state_step, path))
        retain = {path for _, path in sorted(candidates)[-2:]}
        if (root / "best.json").is_file():
            retain.add(root / json.loads((root / "best.json").read_text())["checkpoint"])
        import shutil
        for _, path in candidates:
            if path not in retain:
                shutil.rmtree(path)

    _root_action(accelerator, finish)
    accelerator.wait_for_everyone()
    return final


def resume_metadata(path, config, world_size):
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        path = path.parent / json.loads(path.read_text())["checkpoint"]
    metadata = json.loads((path / "trainer_state.json").read_text())
    if metadata.get("format") != "consistentdmd_public_training_v1":
        raise ValueError("--resume requires a full checkpoint saved by this public trainer")
    if metadata["world_size"] != world_size:
        raise ValueError("Exact native resume requires the saved world size")
    if validate_m_ratio(metadata["config"].get("m_ratio", 0.5)) != validate_m_ratio(config.get("m_ratio", 0.5)):
        raise ValueError("Resume config differs for m_ratio")
    # Output paths and total duration can change; the training stream cannot.
    for key in ("teacher_checkpoint", "batch_size", "global_batch_size", "seed", "attn_func", "bf16", "learning_rate",
                "vae_checkpoint", "latent_stats_path", "ref_stats_path", "eval_samples", "decode_batch_size"):
        if metadata["config"][key] != config[key]:
            raise ValueError(f"Resume config differs for {key}")
    required = ["training_ema.pt", "evaluation_ema.pt"]
    required.extend(f"random_states_{rank}.pkl" for rank in range(world_size))
    for filename in required:
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {path / filename}")
    return path, metadata


@torch.no_grad()
def write_preview(student, vae, mean, std, noise, labels, path, *, bf16, decode_batch_size):
    from torchvision.utils import save_image

    was_training = student.training
    student.eval()
    try:
        with torch.amp.autocast(noise.device.type, dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            t = torch.zeros(len(noise), device=noise.device)
            latents = noise + student(noise, t, r=torch.ones_like(t), y=labels)
        with torch.autocast(noise.device.type, enabled=False):
            images = torch.cat([
                vae.decode(chunk.float() * std + mean).float().cpu()
                for chunk in latents.split(decode_batch_size)
            ])
        if not torch.isfinite(images).all():
            raise FloatingPointError("Nonfinite training preview")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(((images + 1) * 0.5).clamp(0, 1), path, nrow=4)
    finally:
        student.train(was_training)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/train.yaml")
    for name in ("teacher_checkpoint", "vae_checkpoint", "latent_stats_path", "ref_stats_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--attn-func", choices=("base", "fa2"))
    parser.add_argument("--m-ratio", type=float,
                        help="Probability of using independently noised training states (default: 0.5)")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path)
    continuation.add_argument("--import-legacy-checkpoint", type=Path,
                              help="Import a full historical schema-2 training checkpoint")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Check one imported training update and decoding, without saving or W&B")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--wandb", action="store_true", help="Optionally send scalar/preview logs to W&B")
    parser.add_argument("--wandb-project", default="consistentDMD")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-parent-run", help="Parent entity/project/run-id for a new continuation")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test and not args.import_legacy_checkpoint:
        raise ValueError("--smoke-test requires --import-legacy-checkpoint")
    override_keys = ("teacher_checkpoint", "vae_checkpoint", "latent_stats_path", "ref_stats_path",
                     "output_dir", "train_steps", "batch_size", "attn_func", "m_ratio")
    config = load_config(args.config, {key: getattr(args, key) for key in override_keys})
    from accelerate import Accelerator
    from accelerate.utils import AutocastKwargs, DistributedDataParallelKwargs, set_seed
    from ldit.model import create_student, validate_ldit_attention_backend
    from utils.evaluation import evaluate, load_vae_and_stats

    for key in ("teacher_checkpoint", "vae_checkpoint", "latent_stats_path"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {config[key]}; run scripts/download_assets.py first")
    if config["eval_every"]:
        from utils.evaluation import validate_metric_backend
        validate_metric_backend(config["ref_stats_path"])

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by default; --device cpu --attn-func base is available for debugging")
    accelerator = Accelerator(
        cpu=args.device == "cpu", gradient_accumulation_steps=1,
        mixed_precision="bf16" if config["bf16"] else "no",
        kwargs_handlers=[AutocastKwargs(cache_enabled=False),
                        DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)],
    )
    if (accelerator.device.type == "cuda" and (config["bf16"] or config["eval_every"])
            and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("The selected training/evaluation recipe requires a BF16-capable GPU")
    accumulation = accumulation_steps(config["batch_size"], accelerator.num_processes)
    local_batch_sizes = rank_batch_sizes(config["batch_size"], accelerator.num_processes,
                                        accelerator.process_index)
    resume_path, saved = resume_metadata(args.resume, config, accelerator.num_processes) if args.resume else (None, None)
    if args.import_legacy_checkpoint:
        from utils.legacy import metadata, load_legacy_training, restore_legacy_rng
        legacy_metadata = [None]
        def inspect_legacy():
            legacy_metadata[0] = metadata(args.import_legacy_checkpoint, config)
        _root_action(accelerator, inspect_legacy)
        if accelerator.num_processes > 1:
            torch.distributed.broadcast_object_list(legacy_metadata, src=0)
        saved = legacy_metadata[0]
    step = saved["step"] if saved else 0
    best_fid = saved["best_fid"] if saved else None
    best_fid_step = saved["best_fid_step"] if saved else None
    provenance = saved.get("continuation") if saved else None
    if args.import_legacy_checkpoint:
        provenance = {
            "source_checkpoint": str(args.import_legacy_checkpoint.resolve()),
            "source_step": step,
            "parent_wandb_run": args.wandb_parent_run,
            "historical_global_batch_size": 1536,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "rng_policy": "restore archived rank 0; seed other ranks with seed + rank",
        }
    if step >= config["train_steps"]:
        raise ValueError(f"train_steps={config['train_steps']} must exceed checkpoint step {step}")
    output_dir = Path(config["output_dir"])

    def prepare_output():
        if not resume_path and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Use an empty output directory for a fresh run: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    _root_action(accelerator, prepare_output)
    accelerator.wait_for_everyone()
    set_seed(config["seed"] + accelerator.process_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    validate_ldit_attention_backend(config["attn_func"], accelerator.device)
    student = create_student(attn_func=config["attn_func"])
    student.load_teacher_core_state_dict_(torch.load(config["teacher_checkpoint"], map_location="cpu", weights_only=True, mmap=True))
    teacher = deepcopy(student.dit).to(accelerator.device).eval().requires_grad_(False)
    objective = ConsistentDMDLoss(teacher, bf16=config["bf16"], m_ratio=config["m_ratio"])
    vae, mean, std = load_vae_and_stats(config["vae_checkpoint"], config["latent_stats_path"], accelerator.device)
    training_ema = TrainingEMA(student, device=accelerator.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), weight_decay=0.0)
    # Identical to constant_with_warmup with zero warmup; avoids a diffusers dependency.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    student, optimizer, scheduler = accelerator.prepare(student, optimizer, scheduler)
    if resume_path:
        accelerator.load_state(str(resume_path))
        training_ema.ema_model.load_state_dict(torch.load(resume_path / "training_ema.pt", map_location=accelerator.device, weights_only=True), strict=True)
    raw_student = accelerator.unwrap_model(student)
    reporting_ema = FACMEvaluationEMA(raw_student, start_step=step, checkpoint_folder=output_dir / ".posthoc_ema")
    if resume_path:
        reporting_ema.load(resume_path, training_step=step)
    if args.import_legacy_checkpoint:
        import_report = load_legacy_training(
            args.import_legacy_checkpoint, raw_student, optimizer, scheduler,
            training_ema, reporting_ema, saved,
        )
        provenance["import_report"] = import_report
    tracker = None
    if args.wandb and not args.smoke_test:
        def initialize_tracker():
            nonlocal tracker
            import random
            import numpy as np
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            devices = list(range(torch.cuda.device_count())) if accelerator.device.type == "cuda" else []
            try:
                # Tracker retries must not advance the restored training RNG.
                with torch.random.fork_rng(devices=devices):
                    tracker = start_tracker(args, output_dir, config, provenance)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        _root_action(accelerator, initialize_tracker)

    def preview(noise, labels, name):
        # Calling the unwrapped model avoids a rank-zero-only DDP collective.
        write_preview(raw_student, vae, mean, std, noise, labels,
                      output_dir / "previews" / name, bf16=config["bf16"],
                      decode_batch_size=config["decode_batch_size"])

    if args.import_legacy_checkpoint:
        restore_legacy_rng(args.import_legacy_checkpoint, accelerator.process_index, config["seed"])
    if args.smoke_test:
        stats = train_update(accelerator, student, objective, optimizer, scheduler,
                             training_ema, reporting_ema, batch_size=config["batch_size"],
                             accumulation=accumulation, local_batch_sizes=local_batch_sizes, step=step)
        def finish_smoke():
            noise = torch.randn(4, 32, 16, 16, device=accelerator.device)
            preview(noise, torch.tensor([207, 360, 387, 974], device=accelerator.device), "smoke.png")
            result = {"passed": True, "optimizer_updates": 1,
                      "step": step + 1, "global_batch_size": GLOBAL_BATCH_SIZE,
                      "world_size": accelerator.num_processes, "rank_zero_batch_sizes": local_batch_sizes,
                      "reporting_ema": reporting_ema.metadata(),
                      "scalars": {key: value.item() for key, value in stats.items() if value.numel() == 1}}
            if not all(math.isfinite(value) for value in result["scalars"].values()):
                raise FloatingPointError("Smoke test produced nonfinite scalar diagnostics")
            (output_dir / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
            print("GPU smoke test passed: " + json.dumps(result), flush=True)
        _root_action(accelerator, finish_smoke)
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return
    if args.import_legacy_checkpoint:
        save_checkpoint(accelerator, training_ema, reporting_ema, output_dir,
                        step=step, config=config, best_fid=best_fid, best_fid_step=best_fid_step,
                        provenance=provenance)
        accelerator.print(f"Imported full training state at step {step}; native checkpoint saved")
    if not resume_path and not args.import_legacy_checkpoint:
        # Preserve the two startup sanity-check noise draws on every rank.
        torch.randn(4, 32, 16, 16, device=accelerator.device)
        initial_noise = torch.randn(4, 32, 16, 16, device=accelerator.device)
        _root_action(accelerator, lambda: preview(initial_noise, torch.tensor([207, 360, 387, 974], device=accelerator.device), "initial.png"))
    accelerator.print(f"Global batch=1024: {accelerator.num_processes} ranks, "
                      f"microbatch cap={config['batch_size']}, {accumulation} rounds, "
                      f"rank-zero sizes={local_batch_sizes}, m_ratio={config['m_ratio']}")
    try:
        while step < config["train_steps"]:
            stats = train_update(accelerator, student, objective, optimizer, scheduler,
                                 training_ema, reporting_ema, batch_size=config["batch_size"],
                                 accumulation=accumulation, local_batch_sizes=local_batch_sizes, step=step)
            if step % 500 == 0:
                def log_preview():
                    noise = torch.randn(min(16, config["batch_size"]), 32, 16, 16, device=accelerator.device)
                    labels = torch.randint(0, 1000, (len(noise),), device=accelerator.device)
                    name = f"step_{step + 1:06d}.png"
                    preview(noise, labels, name)
                    scalars = {key: value.item() for key, value in stats.items() if value.numel() == 1}
                    print(f"step={step + 1} " + json.dumps(scalars), flush=True)
                    if tracker:
                        import wandb
                        tracker.log({**scalars, "preview": wandb.Image(str(output_dir / "previews" / name))}, step=step + 1, commit=True)
                _root_action(accelerator, log_preview)
            step += 1
            improved = False
            if config["eval_every"] and step % config["eval_every"] == 0:
                evaluation_dir = prepare_evaluation_directory(
                    accelerator, output_dir / "evaluations" / f"step_{step:06d}",
                )
                with reporting_ema.on_device(accelerator.device) as reporting_student:
                    result = evaluate(
                        reporting_student, vae, mean, std, output_dir=evaluation_dir,
                        num_samples=config["eval_samples"], batch_size=config["eval_batch_size"],
                        decode_batch_size=config["decode_batch_size"], steps=1, seed=0,
                        ref_stats_path=config["ref_stats_path"], ema_metadata=reporting_ema.metadata(),
                    )
                fid = float(result["fid"])
                if not math.isfinite(fid):
                    raise FloatingPointError("Evaluation returned a nonfinite FID")
                improved = best_fid is None or fid < best_fid
                if improved:
                    best_fid, best_fid_step = fid, step
                if tracker:
                    tracker.log({"fid": fid}, step=step, commit=True)
            if improved or step % config["save_every"] == 0 or step == config["train_steps"]:
                save_checkpoint(accelerator, training_ema, reporting_ema, output_dir,
                                step=step, config=config, best_fid=best_fid, best_fid_step=best_fid_step,
                                provenance=provenance)
            accelerator.wait_for_everyone()
    finally:
        if tracker:
            tracker.finish()
        accelerator.end_training()


if __name__ == "__main__":
    main()
