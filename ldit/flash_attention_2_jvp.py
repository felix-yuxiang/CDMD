"""Detached forward-AD adapter used by the released FA2 training option.

Adapted from DMF's Python FlashAttention-2 JVP implementation:
https://github.com/kyungmnlee/dmf/blob/main/models/flash_attention_2_jvp.py

Ordinary attention and its backward pass use PyTorch SDPA in model.py. The
consistency target uses only the original split-head Triton primal/JVP kernel.
"""

import torch
import torch.autograd.forward_ad as fwAD

from .triton_utils import _attn_fwd_dual_split


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v, tq, tk, tv, softmax_scale):
        block_q, block_kv, num_stages, num_warps = 128, 128, 2, 8
        b, l_q, h, d = q.shape
        l_kv = k.shape[1]
        tile_dim, tail_dim = 64, 16
        o, to = torch.empty_like(q), torch.empty_like(tq)
        grid = (l_q // block_q, b * h, 1)
        m = torch.empty((b, h, l_q), device=q.device, dtype=torch.float32)
        _attn_fwd_dual_split[grid](
            q, k, v,  #
            tq, tk, tv,  #
            m, o, to,  #
            softmax_scale,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            tq.stride(0), tq.stride(1), tq.stride(2), tq.stride(3),  #
            tk.stride(0), tk.stride(1), tk.stride(2), tk.stride(3),  #
            tv.stride(0), tv.stride(1), tv.stride(2), tv.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            to.stride(0), to.stride(1), to.stride(2), to.stride(3),
            h, l_q, l_kv, d, tile_dim, tail_dim,  #
            block_q=block_q, block_kv=block_kv,  #
            num_stages=num_stages, num_warps=num_warps,  #
        )
        return o, m, to

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        _, _, tangent = outputs
        ctx.save_for_forward(tangent)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("The attention JVP adapter supports detached targets only")

    @staticmethod
    def jvp(ctx, *tangents):
        saved = getattr(ctx, "saved_forward", ())
        return (saved[0] if saved else None), None, None


def flash_attn_func(q, k, v, softmax_scale=None):
    """Run the fixed XL/1 target JVP; all other attention goes through SDPA."""
    if torch.is_grad_enabled():
        raise RuntimeError("The attention JVP adapter requires torch.no_grad()")
    if q.ndim != 4 or q.shape[1:] != (256, 16, 72) or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("The attention JVP adapter requires identical [B,256,16,72] Q/K/V")
    unpacked = [fwAD.unpack_dual(tensor) for tensor in (q, k, v)]
    if all(tangent is None for _, tangent in unpacked):
        raise RuntimeError("The attention JVP adapter requires a forward-AD tangent")
    primals = [primal for primal, _ in unpacked]
    tangents = [torch.zeros_like(primal) if tangent is None else tangent
                for primal, tangent in unpacked]
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    out, _, tangent = FlashAttnFunc.apply(*primals, *tangents, scale)
    return fwAD.make_dual(out, tangent)
