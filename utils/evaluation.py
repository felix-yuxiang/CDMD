"""Portable FID/IS evaluation using the frozen checkpoint's FACM protocol."""

import importlib
import importlib.metadata
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from sampler import SAMPLING_STEPS, euler_sample, indexed_noise
from utils.checkpoint import public_metadata


FIDELITY_COMMIT = "bfe72b995473e4fee9351705d7b005f17442a4ca"
DEFAULT_REFERENCE = "cache/fid-50k-256.npz"


def reference_statistics(path):
    with np.load(Path(path).expanduser(), allow_pickle=False) as stats:
        if "mu" not in stats or "sigma" not in stats:
            raise ValueError("FID reference must contain mu and sigma")
        mean, covariance = stats["mu"], stats["sigma"]
        if mean.shape != (2048,) or covariance.shape != (2048, 2048):
            raise ValueError("FID reference must have mu=(2048,), sigma=(2048,2048)")
        mean, covariance = mean.astype(np.float32), covariance.astype(np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("FID reference must be finite")
        return mean, covariance


def validate_metric_backend(ref_stats_path=DEFAULT_REFERENCE):
    """Reject the incompatible PyPI backend before allocating models."""
    try:
        importlib.import_module("torch_fidelity")
        package = importlib.metadata.distribution("torch-fidelity")
        source = json.loads(package.read_text("direct_url.json") or "{}")
        defaults = importlib.import_module("torch_fidelity.defaults").DEFAULTS
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError("Install the pinned LTH14/torch-fidelity requirement") from error
    pinned = source.get("vcs_info", {}).get("commit_id") == FIDELITY_COMMIT
    pinned |= source.get("url") == f"https://github.com/LTH14/torch-fidelity/archive/{FIDELITY_COMMIT}.tar.gz"
    if not pinned or "fid_statistics_file" not in defaults:
        raise RuntimeError(f"Evaluation requires LTH14/torch-fidelity@{FIDELITY_COMMIT}")
    reference_statistics(ref_stats_path)


def _latent_statistics(mean, std, device):
    mean, std = (torch.as_tensor(value, device=device, dtype=torch.float32) for value in (mean, std))
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)
    shape = (1, 32, 16, 16)
    if torch.broadcast_shapes(mean.shape, shape) != shape or torch.broadcast_shapes(std.shape, shape) != shape:
        raise ValueError("Latent statistics must broadcast to (1,32,16,16)")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Latent statistics must be finite with positive standard deviations")
    return mean, std


def load_vae_and_stats(vae_checkpoint, latent_stats_path, device):
    from ldit.autoencoder import VAEDecoder

    vae = VAEDecoder()
    payload = torch.load(vae_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("VAE checkpoint must contain a tensor state dictionary")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("VAE checkpoint must contain a tensor state dictionary")
    required = ("decoder.", "post_quant_conv.")
    unused = ("encoder.", "quant_conv.", "loss.", "foundation_model.", "linear_proj.")
    unexpected = [key for key in state if not key.startswith(required + unused)]
    if unexpected:
        raise ValueError(f"Unexpected VAE checkpoint tensors: {unexpected[:16]}")
    vae.load_state_dict({key: value for key, value in state.items() if key.startswith(required)}, strict=True)
    vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    stats = torch.load(latent_stats_path, map_location="cpu", weights_only=True)
    mean, std = _latent_statistics(stats["mean"], stats["std"], device)
    return vae, mean, std


def _broadcast(value):
    if dist.is_initialized():
        payload = [value]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]
    return value


def _root_action(rank, description, action):
    error, result = None, None
    if rank == 0:
        try:
            result = action()
        except Exception as caught:
            error = f"{description}: {type(caught).__name__}: {caught}"
    error = _broadcast(error)
    if error is not None:
        raise RuntimeError(error)
    return _broadcast(result)


def _generation_errors(error, world_size):
    errors = [error]
    if dist.is_initialized():
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
    return [item for item in errors if item is not None]


def images_to_uint8(images):
    return images.mul(255.0).round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()


def decode_latents(vae, latents, mean, std, batch_size):
    images = []
    for chunk in latents.split(batch_size):
        decoded = vae.decode(chunk.float() * std + mean).float()
        if not torch.isfinite(decoded).all():
            raise FloatingPointError("VAE produced non-finite pixels")
        images.append(((decoded + 1.0) * 0.5).clamp_(0.0, 1.0).cpu())
    return torch.cat(images)


def _preview(samples, path, count):
    columns = min(8, count)
    grid = Image.new("RGB", (256 * columns, 256 * math.ceil(count / columns)))
    for index in range(count):
        with Image.open(samples / f"{index:08d}.png") as source:
            grid.paste(source.convert("RGB"), ((index % columns) * 256, (index // columns) * 256))
    grid.save(path)
    grid.close()


def _metrics(samples, reference, device):
    import torch_fidelity

    raw = torch_fidelity.calculate_metrics(
        input1=str(samples), input2=None, fid_statistics_file=str(reference),
        cuda=device.type == "cuda", isc=True, fid=True, kid=False, prc=False, verbose=True,
    )
    metrics = {"fid": float(raw["frechet_inception_distance"]),
               "inception_score_mean": float(raw["inception_score_mean"]),
               "inception_score_std": float(raw["inception_score_std"])}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"Non-finite metrics: {metrics}")
    return metrics


@torch.inference_mode()
def evaluate(student, vae, mean, std, *, output_dir, num_samples=50000, batch_size=32,
             decode_batch_size=32, steps=1, seed=0, skip_metrics=False, keep_samples=False,
             ema_metadata=None, ref_stats_path=DEFAULT_REFERENCE):
    """Call on every rank; rank zero scores indexed PNGs on shared storage.

    Failed or unscored samples remain available for inspection. Successful
    metrics remove samples unless keep_samples is set. RNG, modes and precision
    flags are restored even when preparation, generation or scoring fails.
    """
    if steps not in SAMPLING_STEPS or min(num_samples, batch_size, decode_batch_size) <= 0 or seed < 0:
        raise ValueError("Invalid sampling steps, sample count, batch size or seed")
    student = getattr(student, "module", student)
    vae = getattr(vae, "module", vae)
    device = next(student.parameters()).device
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    root = Path(output_dir).expanduser().resolve()
    student_training, vae_training = student.training, vae.training
    vae_parameter = next(vae.parameters())
    vae_device, vae_dtype = vae_parameter.device, vae_parameter.dtype
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)

            def prepare():
                if not skip_metrics:
                    validate_metric_backend(ref_stats_path)
                if root.exists() and any(root.iterdir()):
                    raise FileExistsError(f"Output directory must be new or empty: {root}")
                root.mkdir(parents=True, exist_ok=True)
                (root / "samples").mkdir()
                if not skip_metrics:
                    mu, sigma = reference_statistics(ref_stats_path)
                    np.savez(root / "reference_stats.npz", mu=mu, sigma=sigma)

            _root_action(rank, "Preparing evaluation failed", prepare)
            started, local_error = time.monotonic(), None
            try:
                student.eval()
                vae.to(device=device, dtype=torch.float32).eval()
                mean, std = _latent_statistics(mean, std, device)
                indices = list(range(rank, num_samples, world))
                for offset in range(0, len(indices), batch_size):
                    batch_indices = indices[offset:offset + batch_size]
                    labels = torch.tensor([index % 1000 for index in batch_indices], device=device, dtype=torch.long)
                    noise = indexed_noise(batch_indices, seed=seed, device=device)
                    latents = euler_sample(student, noise, labels, steps=steps)
                    with torch.autocast(device.type, enabled=False):
                        images = decode_latents(vae, latents, mean, std, decode_batch_size)
                    if images.shape != (len(batch_indices), 3, 256, 256):
                        raise ValueError(f"Expected decoded RGB images at 256px, got {tuple(images.shape)}")
                    for array, index in zip(images_to_uint8(images), batch_indices):
                        Image.fromarray(array).save(root / "samples" / f"{index:08d}.png")
                    if rank == 0 and (offset == 0 or (offset // batch_size + 1) % 10 == 0):
                        print(f"Generated {offset + len(batch_indices)}/{len(indices)} rank-zero images", flush=True)
            except Exception as error:
                local_error = f"rank {rank}: {type(error).__name__}: {error}"
            errors = _generation_errors(local_error, world)
            if errors:
                raise RuntimeError("Generation failed: " + " | ".join(errors))
            generation_seconds = time.monotonic() - started

            def score():
                metric_started = time.monotonic()
                samples = root / "samples"
                files = {path.name for path in samples.glob("*.png")}
                if files != {f"{index:08d}.png" for index in range(num_samples)}:
                    raise RuntimeError(f"Expected exactly {num_samples} indexed PNGs; found {len(files)}")
                _preview(samples, root / "preview.png", min(64, num_samples))
                metrics = {}
                if not skip_metrics:
                    with torch.autocast(device.type, enabled=False):
                        metrics = _metrics(samples, root / "reference_stats.npz", device)
                protocol = {
                    "name": "datafree_facm", "metric_backend": None if skip_metrics else "torch_fidelity",
                    "weights": "posthoc", "ema": (None if ema_metadata is None else
                                                    public_metadata({"ema": ema_metadata})["ema"]),
                    "num_samples": num_samples,
                    "steps": steps, "nfe": steps, "mode": "euler", "seed": seed,
                    "guidance": 2.0,
                    "metric_commit": None if skip_metrics else FIDELITY_COMMIT,
                    "noise_policy": "per_sample_index_and_stream",
                    "class_policy": "global_sample_index_mod_num_classes", "num_classes": 1000,
                    "model_dtype": "bfloat16", "vae_dtype": "float32",
                    "pixel_conversion": "round_clamp_uint8_png",
                    "matmul_allow_tf32": False, "cudnn_allow_tf32": True,
                    "fid_stats_sha256": (None if skip_metrics else
                                         hashlib.sha256(Path(ref_stats_path).expanduser().read_bytes()).hexdigest()),
                    "fid_mean_dtype": "float32", "fid_covariance_dtype": "float64",
                    "resolution": 256, "latent_shape": [32, 16, 16], "world_size": world,
                    "batch_size_per_rank": batch_size, "decode_batch_size": decode_batch_size,
                    "preview_sample_indices": list(range(min(64, num_samples))),
                }
                result = {**metrics, "preview_path": "preview.png", "protocol": protocol,
                          "timing": {"generation_seconds": generation_seconds,
                                     "metric_seconds": time.monotonic() - metric_started,
                                     "generation_includes": "noise, sampling, VAE decoding, PNG writes and rank synchronization"}}
                for filename, value in (("protocol.json", protocol), ("results.json", result)):
                    (root / filename).write_text(json.dumps(value, indent=2) + "\n")
                if not skip_metrics:
                    if not keep_samples:
                        shutil.rmtree(samples)
                    (root / "reference_stats.npz").unlink()
                return result

            return _root_action(rank, "Scoring evaluation failed", score)
    finally:
        student.train(student_training)
        vae.to(device=vae_device, dtype=vae_dtype).train(vae_training)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
