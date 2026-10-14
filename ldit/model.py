"""LightningDiT backbone and its dual-time CDMD conditioning wrapper.

This module keeps the released FACM LightningDiT checkpoint layout while
remaining independent of the baseline package. Ordinary attention uses
PyTorch SDPA; forward-mode AD uses either portable manual attention or the
lazily imported FlashAttention-2 JVP adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import TimestepEmbedder, get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed


__all__ = ["LightningDiT", "CDMD_LDiT", "create_student", "validate_ldit_attention_backend"]

FIXED_GUIDANCE = 2.0


def _normalize_attention_backend(backend: str) -> str:
    if backend not in ("base", "fa2"):
        raise ValueError(
            f"unknown LightningDiT attention backend {backend!r}; "
            "expected base or fa2"
        )
    return "manual" if backend == "base" else "fa2"


def _has_forward_ad_tangent(*tensors: torch.Tensor) -> bool:
    return any(fwAD.unpack_dual(tensor)[1] is not None for tensor in tensors)


@lru_cache(maxsize=1)
def _load_flash_attention_2_jvp():
    try:
        from .flash_attention_2_jvp import flash_attn_func
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "could not import the shared FA2 JVP adapter; install the pinned "
            "training dependencies from requirements-train.txt"
        ) from error
    return flash_attn_func


def _flash_attention_2_jvp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the shared FA2 primal/JVP adapter on ``[B, H, N, D]`` inputs."""

    if torch.is_grad_enabled():
        raise RuntimeError(
            "FA2 JVP is only supported for the detached torch.no_grad target JVP"
        )
    if not _has_forward_ad_tangent(q, k, v):
        raise RuntimeError("FA2 JVP was selected without a forward-AD tangent")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("FA2 JVP requires CUDA query, key, and value tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("FA2 JVP requires query, key, and value on one CUDA device")
    if not (q.ndim == k.ndim == v.ndim == 4):
        raise ValueError("FA2 JVP expects rank-four [B,H,N,D] tensors")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            "FA2 JVP currently supports self-attention with identical Q/K/V shapes"
        )
    if q.shape[2] != 256:
        raise ValueError(
            "FA2 JVP is gated for the XL/1 sequence length 256; "
            f"got {q.shape[2]}"
        )
    if q.shape[1] != 16 or q.shape[-1] != 72:
        raise ValueError(
            "FA2 JVP is gated for LightningDiT-XL with 16 heads of dimension "
            f"72; got heads={q.shape[1]}, head_dim={q.shape[-1]}"
        )

    output_dtype = q.dtype
    output = _load_flash_attention_2_jvp()(
        q.transpose(1, 2).to(torch.bfloat16),
        k.transpose(1, 2).to(torch.bfloat16),
        v.transpose(1, 2).to(torch.bfloat16),
        softmax_scale=softmax_scale,
    )
    return output.to(output_dtype).transpose(1, 2)


@lru_cache(maxsize=None)
def _run_flash_jvp_smoke(device_index: int) -> None:
    """Compile and numerically gate the production XL/1 attention geometry."""

    device = torch.device("cuda", device_index)
    shape = (1, 16, 256, 72)
    values = torch.arange(
        math.prod(shape), device=device, dtype=torch.float32
    ).reshape(shape)
    q = (((values % 97) - 48) / 97).to(torch.bfloat16)
    k = (((values.roll(7, -1) % 89) - 44) / 89).to(torch.bfloat16)
    v = (((values.roll(13, -1) % 83) - 41) / 83).to(torch.bfloat16)
    tq = (((values.roll(17, -1) % 79) - 39) / 79).to(torch.bfloat16)
    tk = (((values.roll(23, -1) % 73) - 36) / 73).to(torch.bfloat16)
    tv = (((values.roll(29, -1) % 71) - 35) / 71).to(torch.bfloat16)
    scale = shape[-1] ** -0.5

    def fused(query, key, value):
        return _flash_attention_2_jvp(query, key, value, softmax_scale=scale)

    def reference(query, key, value):
        query = query.float()
        key = key.float()
        value = value.float()
        probabilities = (
            torch.matmul(query, key.transpose(-2, -1)) * scale
        ).softmax(dim=-1)
        return torch.matmul(probabilities, value).to(torch.bfloat16)

    with torch.no_grad():
        fused_primal, fused_tangent = torch.func.jvp(
            fused, (q, k, v), (tq, tk, tv)
        )
        reference_primal, reference_tangent = torch.func.jvp(
            reference, (q, k, v), (tq, tk, tv)
        )

    for label, actual, expected, limit in (
        ("primal", fused_primal, reference_primal, 0.02),
        ("tangent", fused_tangent, reference_tangent, 0.03),
    ):
        if not torch.isfinite(actual).all().item():
            raise RuntimeError(
                f"FA2 JVP startup smoke produced non-finite {label}"
            )
        relative_error = (
            (actual - expected).float().norm()
            / expected.float().norm().clamp_min(1e-12)
        ).item()
        if relative_error >= limit:
            raise RuntimeError(
                f"FA2 JVP startup {label} parity failed: "
                f"relative_error={relative_error:.6g}, limit={limit}"
            )


def validate_ldit_attention_backend(
    attn_func: str = "base",
    device: torch.device | str | None = None,
) -> None:
    """Validate the selected LightningDiT continuous-JVP attention backend."""

    backend = _normalize_attention_backend(attn_func)
    if backend == "manual":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("FA2 JVP requires CUDA")

    resolved_device = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if resolved_device.type != "cuda":
        raise RuntimeError(
            f"FA2 JVP requires a CUDA device, got {resolved_device}"
        )
    major, minor = torch.cuda.get_device_capability(resolved_device)
    if major < 8:
        raise RuntimeError(
            "FA2 JVP requires an Ampere or newer GPU (compute capability 8.x+), "
            f"got {major}.{minor}"
        )
    _load_flash_attention_2_jvp()

    device_index = (
        torch.cuda.current_device()
        if resolved_device.index is None
        else resolved_device.index
    )
    _run_flash_jvp_smoke(device_index)


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + 1e-6
        )
        return normalized.type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    """The EVA-style 2-D rotary embedding used by LightningDiT."""

    def __init__(self, dim, seq_len):
        super().__init__()
        freqs = 1.0 / (
            10000 ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
        )
        positions = torch.arange(seq_len) / seq_len * seq_len
        freqs = torch.einsum("i,j->ij", positions, freqs).repeat_interleave(2, -1)
        height = freqs[:, None, :].expand(seq_len, seq_len, -1)
        width = freqs[None, :, :].expand(seq_len, seq_len, -1)
        freqs = torch.cat((height, width), dim=-1).reshape(seq_len * seq_len, -1)
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, x):
        return x * self.freqs_cos + _rotate_half(x) * self.freqs_sin


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        # The pretrained teacher has a null-class entry for classifier-free
        # guidance. Student training never drops labels.
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.jvp_attention_backend = "manual"
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope):
        batch_size, sequence_length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))

        has_tangent = _has_forward_ad_tangent(q, k, v)
        if has_tangent and self.jvp_attention_backend == "fa2":
            x = _flash_attention_2_jvp(q, k, v, softmax_scale=self.scale)
        elif has_tangent:
            attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
            x = attention @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch_size, sequence_length, channels)
        return self.proj(x)


class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, condition, feat_rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), feat_rope
        )
        return x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


def _as_batch_vector(value, name, batch_size):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    value = value.expand(batch_size) if value.ndim == 0 else value.reshape(-1)
    if value.shape[0] != batch_size:
        raise ValueError(f"{name} must contain one value per sample (batch={batch_size})")
    return value


class LightningDiT(nn.Module):
    """The released LightningDiT architecture; dimensions can shrink for tests."""

    def __init__(
        self, input_size=16, patch_size=1, in_channels=32, hidden_size=1152,
        depth=28, num_heads=16, mlp_ratio=4.0, num_classes=1000,
    ):
        super().__init__()
        if input_size % patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False
        )
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=hidden_size // num_heads // 2, seq_len=input_size // patch_size
        )
        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        patch_size = self.x_embedder.patch_size[0]
        height = width = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], height, width, patch_size, patch_size, self.in_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_channels, height * patch_size, width * patch_size)

    def forward(self, x, t, y):
        t = _as_batch_vector(t, "t", x.shape[0])
        y = _as_batch_vector(y, "y", x.shape[0])
        tokens = self.x_embedder(x) + self.pos_embed
        condition = self.t_embedder(t) + self.y_embedder(y)
        for block in self.blocks:
            tokens = block(tokens, condition, self.feat_rope)
        return self.unpatchify(self.final_layer(tokens, condition))


def _zero_initialized_residual(hidden_size):
    residual = nn.Sequential(
        nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
    )
    nn.init.xavier_uniform_(residual[0].weight)
    nn.init.zeros_(residual[0].bias)
    nn.init.zeros_(residual[2].weight)
    nn.init.zeros_(residual[2].bias)
    return residual


class CDMD_LDiT(nn.Module):
    """Source-time early blocks, target-time later blocks, fixed guidance input."""

    def __init__(self, dit, dmf_depth=10):
        super().__init__()
        if not 0 <= dmf_depth <= len(dit.blocks):
            raise ValueError(f"dmf_depth must be between 0 and {len(dit.blocks)}")
        self.dit = dit
        self.dmf_depth = dmf_depth
        self.r_map = _zero_initialized_residual(dit.hidden_size)
        # Retain the trained guidance modules and their exact parameter names.
        self.gw_embedder = TimestepEmbedder(dit.hidden_size)
        nn.init.normal_(self.gw_embedder.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.gw_embedder.mlp[0].bias)
        nn.init.normal_(self.gw_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.gw_embedder.mlp[2].bias)
        self.gw_map = _zero_initialized_residual(dit.hidden_size)

    def load_teacher_core_state_dict_(self, state_or_checkpoint):
        """Load the pretrained teacher's EMA, preserving student conditioning."""
        if not isinstance(state_or_checkpoint, Mapping):
            raise TypeError("Teacher checkpoint must be a state-dict mapping")
        state = state_or_checkpoint.get("ema", state_or_checkpoint)
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
        ):
            raise TypeError("Teacher state must be a string-to-tensor mapping")
        expected = self.dit.state_dict()
        missing = sorted(expected.keys() - state.keys())
        unexpected = sorted(state.keys() - expected.keys())
        mismatched = [
            key for key in expected.keys() & state.keys()
            if state[key].shape != expected[key].shape
        ]
        if missing or unexpected or mismatched:
            raise RuntimeError(
                f"Teacher core state is incompatible with LightningDiT: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"shape_mismatches={mismatched[:8]}"
            )
        return self.dit.load_state_dict(state, strict=True)

    def forward(self, x, t, r, y):
        t = _as_batch_vector(t, "t", x.shape[0])
        r = _as_batch_vector(r, "r", x.shape[0])
        y = _as_batch_vector(y, "y", x.shape[0])
        tokens = self.dit.x_embedder(x) + self.dit.pos_embed
        source_condition = self.dit.t_embedder(t) + self.dit.y_embedder(y)
        target_condition = source_condition + self.r_map(self.dit.t_embedder(t - r))
        guidance_condition = self.gw_map(self.gw_embedder(torch.full_like(t, FIXED_GUIDANCE)))
        source_condition = source_condition + guidance_condition
        target_condition = target_condition + guidance_condition
        for block in self.dit.blocks[:self.dmf_depth]:
            tokens = block(tokens, source_condition, self.dit.feat_rope)
        for block in self.dit.blocks[self.dmf_depth:]:
            tokens = block(tokens, target_condition, self.dit.feat_rope)
        return self.dit.unpatchify(self.dit.final_layer(tokens, target_condition))


def create_student(attn_func="base"):
    """Build the released 16×16×32 LightningDiT-XL/1 student (fixed gw=2)."""
    model = CDMD_LDiT(LightningDiT(), dmf_depth=10)
    backend = _normalize_attention_backend(attn_func)
    for block in model.dit.blocks:
        block.attn.jvp_attention_backend = backend
    return model
