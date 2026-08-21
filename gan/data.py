"""Data pipeline: BTC 4h OHLCV -> structurally-safe log-space window features.

Every real OHLC bar is losslessly re-expressed as four log-space quantities
relative to the *previous* bar's close:

  r      = log(close_t)  - log(close_{t-1})        close-to-close log return
  gap    = log(open_t)   - log(close_{t-1})         open gap from prior close
  h_off  = log(high_t)   - log(max(open_t, close_t))  high wick, always >= 0
  l_off  = log(min(open_t, close_t)) - log(low_t)     low wick,  always >= 0

`r` and `gap` are z-scored (mean/std). `h_off`/`l_off` are scale-only
normalized (divide by their mean magnitude) so they stay >= 0 after
normalization -- this is what lets the generator guarantee valid OHLC just
by keeping its raw h_off/l_off outputs non-negative (see gan/model.py).

Reconstruction (`reconstruct_ohlc`) inverts this exactly: cumulative-sum the
returns to get the close log-price path, add the gap for opens, and push
high/low out from the bar body by the (non-negative) wick offsets. Because
the offsets are guaranteed >= 0, high >= max(open,close) and
low <= min(open,close) hold by construction, for ANY generator output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "BTCUSDT_4h_full.csv"

FEATURE_NAMES = ["r", "gap", "h_off", "l_off"]


@dataclass
class Normalizer:
    mean_r: float
    std_r: float
    mean_gap: float
    std_gap: float
    scale_h: float
    scale_l: float

    def standardize(self, feats: np.ndarray) -> np.ndarray:
        """feats: [..., 4] in raw log-space order (r, gap, h_off, l_off)."""
        out = np.empty_like(feats)
        out[..., 0] = (feats[..., 0] - self.mean_r) / self.std_r
        out[..., 1] = (feats[..., 1] - self.mean_gap) / self.std_gap
        out[..., 2] = feats[..., 2] / self.scale_h  # scale-only: preserves >= 0
        out[..., 3] = feats[..., 3] / self.scale_l  # scale-only: preserves >= 0
        return out

    def destandardize(self, z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z)
        out[..., 0] = z[..., 0] * self.std_r + self.mean_r
        out[..., 1] = z[..., 1] * self.std_gap + self.mean_gap
        out[..., 2] = z[..., 2] * self.scale_h
        out[..., 3] = z[..., 3] * self.scale_l
        return out


def load_ohlc() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def ohlc_to_features(df: pd.DataFrame) -> np.ndarray:
    """Per-bar (r, gap, h_off, l_off) in log space, shape [N-1, 4].

    Row j corresponds to df row j+1; its "previous close" is df row j's
    close (the first bar of the file has no previous close and is dropped).
    """
    o, h, l, c = (df[col].to_numpy(dtype=np.float64) for col in ["open", "high", "low", "close"])
    log_o, log_h, log_l, log_c = np.log(o), np.log(h), np.log(l), np.log(c)

    prev_log_c = log_c[:-1]
    log_o, log_h, log_l, log_c = log_o[1:], log_h[1:], log_l[1:], log_c[1:]

    r = log_c - prev_log_c
    gap = log_o - prev_log_c
    body_hi = np.maximum(log_o, log_c)
    body_lo = np.minimum(log_o, log_c)
    h_off = log_h - body_hi
    l_off = body_lo - log_l

    assert (h_off >= -1e-9).all() and (l_off >= -1e-9).all(), "real data violates OHLC ordering"
    return np.stack([r, gap, h_off, l_off], axis=-1)


def fit_normalizer(feats: np.ndarray) -> Normalizer:
    r, gap, h_off, l_off = feats[..., 0], feats[..., 1], feats[..., 2], feats[..., 3]
    return Normalizer(
        mean_r=float(r.mean()),
        std_r=float(r.std()),
        mean_gap=float(gap.mean()),
        std_gap=float(gap.std()),
        scale_h=float(h_off.mean() + 1e-8),
        scale_l=float(l_off.mean() + 1e-8),
    )


def make_windows(z_feats: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    """Sliding windows over the standardized feature series.

    Overlapping (stride < window_len) is intentional data augmentation for
    GAN training -- unlike the reality-floor measurement, which needs
    *non-overlapping* windows for statistical independence, the GAN just
    needs many varied training examples. Returns [n_windows, 4, window_len].
    """
    n = len(z_feats)
    starts = range(0, n - window_len + 1, stride)
    windows = np.stack([z_feats[s : s + window_len] for s in starts], axis=0)  # [n, window_len, 4]
    return np.transpose(windows, (0, 2, 1))  # [n, 4, window_len]


def window_anchor_prices(df: pd.DataFrame, window_len: int, stride: int) -> np.ndarray:
    """The real close price immediately before each window's first bar
    (window i's first feature row is df index i*stride + 1, so its anchor
    is df.close[i*stride]). Used to reconstruct windows back into real
    dollar-scale OHLC for validation/plotting."""
    close = df["close"].to_numpy(dtype=np.float64)
    n_feats = len(df) - 1
    starts = list(range(0, n_feats - window_len + 1, stride))
    return close[starts]


def reconstruct_ohlc(z: np.ndarray, normalizer: Normalizer, anchor_price) -> dict[str, np.ndarray]:
    """Inverse transform: standardized (r,gap,h_off,l_off) of shape [...,4,T]
    -> OHLC price paths of shape [...,T]. `anchor_price` is the real close
    price the window starts from (scalar or array broadcastable to [...]).

    Structurally guaranteed valid (high >= max(o,c), low <= min(o,c),
    high >= low) whenever the h_off/l_off channels of `z` are >= 0 --
    exactly what the generator's softplus output guarantees.
    """
    feats = np.moveaxis(z, -2, -1)  # [...,T,4]
    raw = normalizer.destandardize(feats)
    r, gap, h_off, l_off = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
    h_off = np.maximum(h_off, 0.0)
    l_off = np.maximum(l_off, 0.0)

    log_close = np.cumsum(r, axis=-1)
    prev_log_close = np.concatenate([np.zeros_like(log_close[..., :1]), log_close[..., :-1]], axis=-1)
    log_open = prev_log_close + gap
    body_hi = np.maximum(log_open, log_close)
    body_lo = np.minimum(log_open, log_close)
    log_high = body_hi + h_off
    log_low = body_lo - l_off

    anchor = np.asarray(anchor_price)
    if anchor.ndim > 0:
        anchor = anchor.reshape(anchor.shape + (1,) * (log_close.ndim - anchor.ndim))

    return {
        "open": anchor * np.exp(log_open),
        "high": anchor * np.exp(log_high),
        "low": anchor * np.exp(log_low),
        "close": anchor * np.exp(log_close),
    }
