"""Windowing and stylized-fact featurization of OHLCV price series."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "ann_vol",
    "skew",
    "excess_kurtosis",
    "acf_ret_1",
    "acf_ret_2",
    "acf_ret_3",
    "acf_absret_1",
    "acf_absret_2",
    "acf_absret_3",
    "acf_absret_5",
    "q01",
    "q05",
    "q95",
    "q99",
    "max_drawdown",
    "longest_up_run",
    "longest_down_run",
]

BARS_PER_YEAR_4H = 6 * 365  # 6 bars/day * 365 days


def slice_windows(closes: np.ndarray, window_len: int) -> list[np.ndarray]:
    """Split a 1D array of close prices into non-overlapping windows of `window_len`
    bars. Any leftover tail shorter than a full window is dropped."""
    n_windows = len(closes) // window_len
    usable = n_windows * window_len
    trimmed = closes[:usable]
    return list(trimmed.reshape(n_windows, window_len))


def _acf(x: np.ndarray, lag: int) -> float:
    if lag >= len(x):
        return 0.0
    x0 = x[:-lag]
    x1 = x[lag:]
    if x0.std() == 0 or x1.std() == 0:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def _longest_run(signs: np.ndarray, positive: bool) -> int:
    target = 1 if positive else -1
    best = cur = 0
    for s in signs:
        if s == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _max_drawdown(prices: np.ndarray) -> float:
    running_max = np.maximum.accumulate(prices)
    drawdown = (prices - running_max) / running_max
    return float(drawdown.min())


def featurize_window(prices: np.ndarray) -> np.ndarray:
    """Compute the stylized-fact feature vector for one window of close prices.
    Returns a vector in the order given by FEATURE_NAMES."""
    log_ret = np.diff(np.log(prices))

    ann_vol = float(log_ret.std(ddof=1) * np.sqrt(BARS_PER_YEAR_4H))

    mean = log_ret.mean()
    std = log_ret.std(ddof=1)
    if std > 0:
        skew = float(np.mean(((log_ret - mean) / std) ** 3))
        excess_kurtosis = float(np.mean(((log_ret - mean) / std) ** 4) - 3.0)
    else:
        skew = 0.0
        excess_kurtosis = 0.0

    abs_ret = np.abs(log_ret)

    acf_ret_1 = _acf(log_ret, 1)
    acf_ret_2 = _acf(log_ret, 2)
    acf_ret_3 = _acf(log_ret, 3)
    acf_absret_1 = _acf(abs_ret, 1)
    acf_absret_2 = _acf(abs_ret, 2)
    acf_absret_3 = _acf(abs_ret, 3)
    acf_absret_5 = _acf(abs_ret, 5)

    q01, q05, q95, q99 = np.quantile(log_ret, [0.01, 0.05, 0.95, 0.99])

    max_dd = _max_drawdown(prices)

    signs = np.sign(log_ret)
    longest_up = _longest_run(signs, positive=True)
    longest_down = _longest_run(signs, positive=False)

    return np.array(
        [
            ann_vol,
            skew,
            excess_kurtosis,
            acf_ret_1,
            acf_ret_2,
            acf_ret_3,
            acf_absret_1,
            acf_absret_2,
            acf_absret_3,
            acf_absret_5,
            q01,
            q05,
            q95,
            q99,
            max_dd,
            float(longest_up),
            float(longest_down),
        ]
    )


def build_feature_matrix(closes: np.ndarray, window_len: int) -> pd.DataFrame:
    """Slice the price series into non-overlapping windows of `window_len` and
    return a DataFrame of stylized-fact features, one row per window."""
    windows = slice_windows(closes, window_len)
    rows = [featurize_window(w) for w in windows]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
