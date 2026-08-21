"""Reality-floor measurement: what AUC does real BTC get against itself?

For each window length, slice BTC 4h closes into non-overlapping windows,
featurize each window by stylized facts, then repeatedly split windows
50/50 at random into two groups and measure how well a classifier tells
the groups apart (5-fold CV AUC). Since the split is pure noise, the
resulting AUC distribution is the "reality floor": how distinguishable
real market windows are from other real market windows of the same length.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reality_floor.features import build_feature_matrix
from reality_floor.classify import repeated_real_vs_real_auc

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "BTCUSDT_4h_full.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

WINDOW_LENGTHS_DAYS = {"7d": 42, "14d": 84, "30d": 180}
N_REPEATS = 20
BASE_SEED = 42


def load_closes() -> tuple[np.ndarray, pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df["close"].to_numpy(), df["timestamp"].iloc[0], df["timestamp"].iloc[-1]


def main() -> None:
    closes, start, end = load_closes()
    print(f"BTC 4h data: {len(closes)} bars, {start} -> {end}")
    print()

    rows = []
    for label, window_len in WINDOW_LENGTHS_DAYS.items():
        features = build_feature_matrix(closes, window_len)
        n_windows = len(features)

        aucs = repeated_real_vs_real_auc(features, N_REPEATS, seed=BASE_SEED + window_len)

        rows.append(
            {
                "window": label,
                "bars": window_len,
                "n_windows": n_windows,
                "mean_auc": aucs.mean(),
                "std_auc": aucs.std(ddof=1),
                "min_auc": aucs.min(),
                "max_auc": aucs.max(),
            }
        )
        print(
            f"L={label:>4} ({window_len:>3} bars): n_windows={n_windows:>4}  "
            f"AUC mean={aucs.mean():.4f} std={aucs.std(ddof=1):.4f} "
            f"min={aucs.min():.4f} max={aucs.max():.4f}"
        )

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "reality_floor_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
