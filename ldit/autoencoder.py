"""The released VA-VAE decoder, adapted from FACM; see THIRD_PARTY_NOTICES.md."""

import torch
from torch import nn


def nonlinearity(x):
    return x * torch.sigmoid(x)


def normalize(channels):
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest"))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(nonlinearity(self.norm2(h)))
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = normalize(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        normalized = self.norm(x)
        q, k, v = self.q(normalized), self.k(normalized), self.v(normalized)
        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch, channels, height * width)
        attention = torch.bmm(q, k) * (int(channels) ** -0.5)
        attention = torch.nn.functional.softmax(attention, dim=2)
        v = v.reshape(batch, channels, height * width)
        h = torch.bmm(v, attention.permute(0, 2, 1)).reshape(batch, channels, height, width)
        return x + self.proj_out(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 512
        resolution = 16
        self.conv_in = nn.Conv2d(32, channels, kernel_size=3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)
        self.up = nn.ModuleList()
        # Preserve the checkpoint's up.0 ... up.4 module names.
        for level, multiplier in reversed(list(enumerate((1, 1, 2, 2, 4)))):
            up = nn.Module()
            up.block, up.attn = nn.ModuleList(), nn.ModuleList()
            out_channels = 128 * multiplier
            for _ in range(3):
                up.block.append(ResnetBlock(channels, out_channels))
                channels = out_channels
                if resolution == 16:
                    up.attn.append(AttnBlock(channels))
            if level != 0:
                up.upsample = Upsample(channels)
                resolution *= 2
            self.up.insert(0, up)
        self.norm_out = normalize(channels)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for up in reversed(self.up):
            for index, block in enumerate(up.block):
                h = block(h)
                if up.attn:
                    h = up.attn[index](h)
            if hasattr(up, "upsample"):
                h = up.upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class VAEDecoder(nn.Module):
    """Decode 32-channel latents; the encoder is unnecessary for data-free training."""

    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.post_quant_conv = nn.Conv2d(32, 32, kernel_size=1)

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))
