"""The active consistency + flow-matching loss used by checkpoint-best.

Extracted from the frozen training source. Model time runs from noise (0) to
data (1). Student guidance is fixed inside the model; the teacher uses CFG.
"""

import math
from numbers import Real

import torch
from torch import nn
import torch.nn.functional as F


def validate_m_ratio(value):
    """Probability of selecting the independently noised training state."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not 0.0 <= value <= 1.0 or not math.isfinite(value)):
        raise ValueError("m_ratio must be a finite number in [0, 1]")
    return float(value)


def sample_times(batch_size, device):
    """Sorted logit-normal draws, preserving the reference RNG consumption."""
    t = torch.sigmoid(torch.randn(batch_size, device=device) - 0.4).clamp(0, 1)
    r = torch.sigmoid(torch.randn(batch_size, device=device) + 1.2).clamp(0, 1)
    t, r = torch.minimum(t, r), torch.maximum(t, r)
    # Preserve the draw consumed by the original ratio=1 (always false) mask.
    torch.rand(batch_size, device=device)
    return t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)


def teacher_velocity(teacher, x, t, labels, num_classes=1000):
    """Teacher CFG 1.75 on native time [0.125, 1], otherwise CFG 1."""
    guidance = torch.where((t >= 0.125) & (t <= 1.0), 1.75, 1.0)
    combined = teacher(
        torch.cat([x, x]), torch.cat([t, t]),
        y=torch.cat([labels, torch.full_like(labels, num_classes)]),
    )
    conditional, unconditional = combined.chunk(2)
    guided = unconditional + guidance.view(-1, 1, 1, 1) * (conditional - unconditional)
    return guided[:, :x.shape[1]]


def anchored_loss(u, fm_prediction, velocity, dudt, t, r, epsilon=0.001):
    """Clipped detached consistency target and diagonal FM MSE + cosine loss."""
    u, fm_prediction = u.float(), fm_prediction.float()
    velocity, dudt = velocity.detach().float(), dudt.detach().float()
    delta = (r - t).float()
    residual = (u.detach() - velocity - delta * dudt).detach()
    target = (u.detach() - residual.clamp(-1.0, 1.0)).detach()
    error = (u - target).square().flatten(1).mean(1)
    cm = error / torch.sqrt(error.detach() + epsilon)
    fm = (fm_prediction - velocity).square().flatten(1).mean(1)
    fm = fm + (1.0 - F.cosine_similarity(fm_prediction, velocity, dim=1)).flatten(1).mean(1)
    return fm + cm, fm, cm, residual


class ConsistentDMDLoss(nn.Module):
    """Data-free objective: current-student states and a frozen LDiT teacher."""

    def __init__(self, teacher, *, bf16=True, num_classes=1000, m_ratio=0.5):
        super().__init__()
        self.m_ratio = validate_m_ratio(m_ratio)
        self.teacher = teacher.eval().requires_grad_(False)
        self.bf16 = bf16
        self.num_classes = num_classes

    def autocast(self, device):
        return torch.amp.autocast(
            device.type, dtype=torch.bfloat16, enabled=self.bf16, cache_enabled=False,
        )

    def forward(self, student, noise, labels):
        raw_student = student.module if hasattr(student, "module") else student
        batch_size = noise.shape[0]
        t, r = sample_times(batch_size, noise.device)
        zeros = torch.zeros(batch_size, device=noise.device)
        ones = torch.ones(batch_size, device=noise.device)
        with torch.no_grad(), self.autocast(noise.device):
            latents = noise + raw_student(noise, zeros, r=ones, y=labels)[:, :noise.shape[1]]
            xt_new = ((1.0 - t) * torch.randn_like(noise) + t * latents).detach()
            xt_same = noise + t * raw_student(noise, zeros, r=t.flatten(), y=labels)[:, :noise.shape[1]]
            mixture_mask = torch.rand(batch_size, device=noise.device) < self.m_ratio
            xt = torch.where(mixture_mask.view(-1, 1, 1, 1), xt_new, xt_same)
            velocity = teacher_velocity(self.teacher, xt, t.flatten(), labels, self.num_classes)

        def student_fn(x, source_time, target_time):
            return raw_student(x, source_time, r=target_time, y=labels)[:, :noise.shape[1]]

        with torch.no_grad(), self.autocast(noise.device):
            _, dudt = torch.func.jvp(
                student_fn, (xt.float(), t.flatten(), r.flatten()),
                (velocity.float(), torch.ones_like(t.flatten()), torch.zeros_like(r.flatten())),
            )
        # One differentiable DDP forward covers the interval and diagonal tasks.
        with self.autocast(noise.device):
            predictions = student(
                torch.cat([xt, xt]), torch.cat([t.flatten(), t.flatten()]),
                r=torch.cat([r.flatten(), t.flatten()]), y=torch.cat([labels, labels]),
            )[:, :noise.shape[1]]
        u, fm_prediction = predictions.chunk(2)
        per_example, fm, cm, residual = anchored_loss(u, fm_prediction, velocity, dudt, t, r)
        return per_example.mean(), {
            "loss": per_example.mean().detach(), "loss_fm": fm.mean().detach(),
            "loss_cm": cm.mean().detach(), "sampled_t": t.detach(), "sampled_r": r.detach(),
            "per_example_loss": per_example.detach(),
            "consistency_residual_rms": residual.square().mean().sqrt(),
            "consistency_clip_fraction": (residual.abs() > 1).float().mean(),
            "jvp_term_rms": ((r - t) * dudt.float()).square().mean().sqrt().detach(),
            "teacher_velocity_rms": velocity.float().square().mean().sqrt().detach(),
        }
