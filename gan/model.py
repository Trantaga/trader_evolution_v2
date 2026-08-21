"""1D-conv WGAN-GP generator/critic over [4, 168] window representations.

IMPORTANT: the 4 channels are NOT raw (open, high, low, close) prices -- the
generator never outputs 4 independent prices. They are the structural
parametrization from gan/data.py: (r, gap, h_off, l_off) = (close-to-close
log return, open gap, high-wick offset, low-wick offset), with h_off/l_off
passed through softplus so they can never be negative. That non-negativity
is what makes the *reconstructed* OHLC (gan.data.reconstruct_ohlc) valid by
construction: high >= max(open,close), low <= min(open,close), high >= low,
for any generator weights, at any point in training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW_LEN = 168  # 168 4h-bars = 28 days (NOT 7 days -- see README "window length note")
N_CHANNELS = 4  # (r, gap, h_off, l_off) -- see module docstring


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, base_channels: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.base_channels = base_channels
        self.start_len = 21  # 21 * 2 * 2 * 2 = 168

        self.fc = nn.Linear(z_dim, base_channels * self.start_len)

        def up_block(in_ch: int, out_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            up_block(base_channels, base_channels // 2),  # 21 -> 42
            up_block(base_channels // 2, base_channels // 4),  # 42 -> 84
            up_block(base_channels // 4, base_channels // 4),  # 84 -> 168
        )
        self.head = nn.Conv1d(base_channels // 4, N_CHANNELS, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.base_channels, self.start_len)
        x = self.net(x)
        raw = self.head(x)  # [B, 4, WINDOW_LEN], unconstrained

        r, gap, h_off_raw, l_off_raw = raw.unbind(dim=1)
        h_off = F.softplus(h_off_raw)  # >= 0, always
        l_off = F.softplus(l_off_raw)  # >= 0, always
        return torch.stack([r, gap, h_off, l_off], dim=1)


class Critic(nn.Module):
    """No BatchNorm anywhere: WGAN-GP's gradient penalty is defined per-sample,
    and BatchNorm couples samples within a batch, which invalidates it. Use
    GroupNorm(1, C) (= LayerNorm over channels+time, still per-sample) instead.
    """

    def __init__(self, base_channels: int = 128):
        super().__init__()

        def down_block(in_ch: int, out_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(1, out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            down_block(N_CHANNELS, base_channels // 4),  # 168 -> 84
            down_block(base_channels // 4, base_channels // 2),  # 84 -> 42
            down_block(base_channels // 2, base_channels),  # 42 -> 21
        )
        self.out = nn.Linear(base_channels * 21, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.flatten(1)
        return self.out(h).squeeze(-1)  # no sigmoid: Wasserstein critic, unbounded scalar
