"""1D-conv WGAN-GP generator/critic over [4, WINDOW_LEN] window representations.

IMPORTANT: the 4 channels are NOT raw (open, high, low, close) prices -- the
generator never outputs 4 independent prices. They are the structural
parametrization from gan/data.py: (r, gap, h_off, l_off) = (close-to-close
log return, open gap, high-wick offset, low-wick offset), with h_off/l_off
passed through softplus so they can never be negative. That non-negativity
is what makes the *reconstructed* OHLC (gan.data.reconstruct_ohlc) valid by
construction: high >= max(open,close), low <= min(open,close), high >= low,
for any generator weights, at any point in training.

WINDOW_LEN is the only thing that varies between branches comparing window
lengths (e.g. main=168 bars/28d vs windows-42bar=42 bars/7d). The conv stack
below is generic over WINDOW_LEN so that changing it is a one-line edit: it
always builds START_LEN=21 and exactly 3 resampling blocks (matching the
depth/channel schedule of the original 168-bar model), applying stride-2
resampling to only as many of those 3 blocks as needed to reach WINDOW_LEN
and leaving the rest as stride-1 same-length refinement blocks -- so
architecture depth and width stay comparable across window lengths, and the
168-bar case is bit-for-bit identical to the original hand-written model
(WINDOW_LEN=168 needs all 3 blocks to resample, same as before).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW_LEN = 42  # 42 4h-bars = 7 days (true 7-day window; main branch uses 168 = 28 days)
N_CHANNELS = 4  # (r, gap, h_off, l_off) -- see module docstring

START_LEN = 21  # base sequence length before/after resampling; WINDOW_LEN must be START_LEN * 2**k
N_BLOCKS = 3  # fixed depth, matching the original 168-bar model


def _n_resample_blocks() -> int:
    factor = WINDOW_LEN / START_LEN
    n_steps = round(math.log2(factor))
    assert START_LEN * (2**n_steps) == WINDOW_LEN, (
        f"WINDOW_LEN={WINDOW_LEN} must equal START_LEN({START_LEN}) * a power of 2"
    )
    assert 0 <= n_steps <= N_BLOCKS, f"need {n_steps} resampling blocks but only have {N_BLOCKS}"
    return n_steps


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, base_channels: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.base_channels = base_channels
        self.start_len = START_LEN

        self.fc = nn.Linear(z_dim, base_channels * self.start_len)

        def block(in_ch: int, out_ch: int, upsample: bool) -> nn.Module:
            conv = (
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
                if upsample
                else nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            )
            return nn.Sequential(conv, nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))

        n_upsample = _n_resample_blocks()  # first n_upsample blocks double length; rest hold it
        channels = [base_channels, base_channels // 2, base_channels // 4, base_channels // 4]
        self.net = nn.Sequential(
            *[block(channels[i], channels[i + 1], upsample=i < n_upsample) for i in range(N_BLOCKS)]
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

        def block(in_ch: int, out_ch: int, downsample: bool) -> nn.Module:
            conv = (
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
                if downsample
                else nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            )
            return nn.Sequential(conv, nn.GroupNorm(1, out_ch), nn.LeakyReLU(0.2, inplace=True))

        n_downsample = _n_resample_blocks()  # first n_downsample blocks halve length; rest hold it
        channels = [N_CHANNELS, base_channels // 4, base_channels // 2, base_channels]
        self.net = nn.Sequential(
            *[block(channels[i], channels[i + 1], downsample=i < n_downsample) for i in range(N_BLOCKS)]
        )
        self.out = nn.Linear(base_channels * START_LEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.flatten(1)
        return self.out(h).squeeze(-1)  # no sigmoid: Wasserstein critic, unbounded scalar
