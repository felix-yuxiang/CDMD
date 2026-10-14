"""Deterministic Euler sampler for checkpoint-best (noise time 0 → data time 1)."""

import torch


SAMPLING_STEPS = (1, 2, 4, 8, 16, 32, 64)


def indexed_noise(indices, shape=(32, 16, 16), seed=0, device="cpu"):
    """CPU FP32 noise, independent of batch size and distributed partitioning."""
    device = torch.device(device)
    samples = []
    for index in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed((seed + int(index) * 1_000_003) % (2**63 - 1))
        samples.append(torch.randn(shape, generator=generator, dtype=torch.float32))
    return torch.stack(samples).to(device, non_blocking=device.type == "cuda")


@torch.inference_mode()
def euler_sample(model, noise, labels, steps=8, model_dtype=torch.bfloat16):
    """FP32 weights, autocast model calls, FP64 integration, FP32 output.

    The one-step endpoint arithmetic is retained from the archived evaluator.
    The model owns the checkpoint's fixed conditioning; there is no CFG pass.
    """
    if steps not in SAMPLING_STEPS:
        raise ValueError(f"steps must be one of {SAMPLING_STEPS}")

    def velocity(x, source, target):
        t = torch.full((len(noise),), source, device=noise.device, dtype=model_dtype)
        r = torch.full_like(t, target)
        forward = getattr(model, "_original_forward", model.forward)
        with torch.autocast(noise.device.type, dtype=model_dtype, enabled=model_dtype != torch.float32):
            output = forward(x.to(model_dtype), t=t, r=r, y=labels)
        if not torch.is_tensor(output) or output.shape != noise.shape:
            raise ValueError("Student must return a velocity with the input latent shape")
        return output.to(torch.float64)

    x = noise.to(torch.float64)
    if steps == 1:
        x = x + (1.0 - 0.0) * velocity(x, 0.0, 1.0)
    else:
        times = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64).tolist()
        for source, target in zip(times[:-1], times[1:]):
            x = x + (target - source) * velocity(x, source, target)
    if not torch.isfinite(x).all():
        raise FloatingPointError("Student produced non-finite latents")
    return x.to(torch.float32)
