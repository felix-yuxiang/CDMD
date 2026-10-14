# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math

import numpy as np
import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    """Embed scalar times with the checkpoint's 256 sinusoidal frequencies."""

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(128, dtype=torch.float32) / 128
        ).to(device=t.device)
        angles = t[:, None].float() * frequencies[None]
        return self.mlp(torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1))


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Return the checkpoint's fixed square-grid positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed(embed_dim, positions):
    frequencies = np.arange(embed_dim // 2, dtype=np.float64)
    frequencies /= embed_dim / 2.0
    frequencies = 1.0 / 10000**frequencies
    angles = np.einsum("m,d->md", positions.reshape(-1), frequencies)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
