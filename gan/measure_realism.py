"""Step 2: measure how distinguishable synthetic 168-bar GAN windows are from
real ones, reusing the EXACT SAME classifier + stylized-fact features as
`reality_floor/`, so the AUC here is directly comparable to the real-vs-real
floor measured there.

Real windows for this test are NON-OVERLAPPING 168-bar slices of the full
history (reality_floor's requirement, for statistical independence between
windows) -- deliberately different from the overlapping stride-8 windows
`gan/train.py` trains on. Mixing those two would bias the AUC.

Because 168 bars wasn't one of the three window lengths reality_floor
originally measured (42/84/180), this script also (re)computes the
real-vs-real floor at L=168 using the identical `reality_floor.classify`
code, so there's an apples-to-apples number to compare against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reality_floor.classify import cv_auc, repeated_real_vs_real_auc
from reality_floor.features import FEATURE_NAMES, build_feature_matrix, featurize_window

from gan.data import fit_normalizer, load_ohlc, ohlc_to_features, reconstruct_ohlc
from gan.model import Generator, WINDOW_LEN
from gan.train import GEN_BASE_CHANNELS, Z_DIM, pick_device

# ----------------------------------- CONFIG -----------------------------------
N_REPEATS = 20  # fresh synthetic batches / CV splits per checkpoint (same discipline as Step 0)
BASE_SEED = 4242
FLOOR_SEED = 42  # matches reality_floor/run.py's seed, for a like-for-like floor at L=168

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "gan"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

N_IMPORTANCE_REPEATS = 30  # permutation-importance repeats for the feature-importance diagnostic
# --------------------------------------------------------------------------------


def load_generator(checkpoint_path: Path, device: torch.device) -> Generator:
    G = Generator(z_dim=Z_DIM, base_channels=GEN_BASE_CHANNELS).to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()
    return G


def synthesize_close_paths(G: Generator, n: int, normalizer, device: torch.device, rng_seed: int) -> np.ndarray:
    """Sample n synthetic windows and reconstruct their close-price paths
    (anchor=1.0 -- arbitrary, since all reality_floor features are scale-
    invariant log-return statistics)."""
    g = torch.Generator()
    g.manual_seed(rng_seed)
    z = torch.randn(n, Z_DIM, generator=g).to(device)
    with torch.no_grad():
        fake = G(z).cpu().numpy()
    ohlc = reconstruct_ohlc(fake, normalizer, anchor_price=1.0)
    return ohlc["close"]


def featurize_batch(price_paths: np.ndarray) -> pd.DataFrame:
    rows = [featurize_window(p) for p in price_paths]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def real_vs_fake_auc(real_feats: pd.DataFrame, fake_feats: pd.DataFrame, seed: int) -> float:
    X = pd.concat([real_feats, fake_feats], ignore_index=True)
    y = np.concatenate([np.ones(len(real_feats)), np.zeros(len(fake_feats))])
    rng = np.random.default_rng(seed)
    clf_seed = int(rng.integers(0, 2**31 - 1))
    cv_seed = int(rng.integers(0, 2**31 - 1))
    return cv_auc(X, y, clf_seed, cv_seed)


def measure_checkpoint(
    ckpt_path: Path, real_feats: pd.DataFrame, normalizer, device: torch.device, base_seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    G = load_generator(ckpt_path, device)
    n_real = len(real_feats)
    aucs = np.empty(N_REPEATS)
    last_fake_feats = None
    for i in range(N_REPEATS):
        seed = base_seed + i
        close_paths = synthesize_close_paths(G, n_real, normalizer, device, rng_seed=seed)
        fake_feats = featurize_batch(close_paths)
        aucs[i] = real_vs_fake_auc(real_feats, fake_feats, seed=seed)
        last_fake_feats = fake_feats
    return aucs, last_fake_feats


def feature_importance_diagnostic(
    real_feats: pd.DataFrame, fake_feats: pd.DataFrame, seed: int
) -> tuple[list[tuple[str, float]], float]:
    """Fit once on a train split, then use permutation importance (model-
    agnostic -- HistGradientBoostingClassifier has no built-in
    feature_importances_) on a held-out test split to see which stylized
    facts most separate real from synthetic."""
    X = pd.concat([real_feats, fake_feats], ignore_index=True)
    y = np.concatenate([np.ones(len(real_feats)), np.zeros(len(fake_feats))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(X_train.values, y_train)
    test_auc = float(roc_auc_score(y_test, clf.predict_proba(X_test.values)[:, 1]))

    perm = permutation_importance(
        clf, X_test.values, y_test, scoring="roc_auc", n_repeats=N_IMPORTANCE_REPEATS, random_state=seed
    )
    ranked = sorted(zip(FEATURE_NAMES, perm.importances_mean), key=lambda kv: -kv[1])
    return ranked, test_auc


def main() -> None:
    device = pick_device()

    df = load_ohlc()
    feats = ohlc_to_features(df)
    normalizer = fit_normalizer(feats)
    closes = df["close"].to_numpy()

    real_feats = build_feature_matrix(closes, WINDOW_LEN)
    n_real = len(real_feats)
    print(f"Real NON-OVERLAPPING {WINDOW_LEN}-bar windows for validation: {n_real}")

    floor_aucs = repeated_real_vs_real_auc(real_feats, n_repeats=N_REPEATS, seed=FLOOR_SEED)
    print(
        f"Reality floor at L={WINDOW_LEN} (real-vs-real, for comparison): "
        f"mean={floor_aucs.mean():.4f} std={floor_aucs.std(ddof=1):.4f} "
        f"min={floor_aucs.min():.4f} max={floor_aucs.max():.4f}\n"
    )

    checkpoints = sorted(CHECKPOINT_DIR.glob("gen_step_*.pt"))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {CHECKPOINT_DIR} -- run gan/train.py first.")
    print(f"Found {len(checkpoints)} checkpoints: {[c.stem for c in checkpoints]}\n")

    results = {}
    for ckpt in checkpoints:
        step = int(ckpt.stem.split("_")[-1])
        aucs, _ = measure_checkpoint(ckpt, real_feats, normalizer, device, base_seed=BASE_SEED + step)
        results[step] = dict(
            mean=float(aucs.mean()),
            std=float(aucs.std(ddof=1)),
            min=float(aucs.min()),
            max=float(aucs.max()),
            aucs=aucs.tolist(),
        )
        print(
            f"step {step:6d}  AUC mean={aucs.mean():.4f} std={aucs.std(ddof=1):.4f} "
            f"min={aucs.min():.4f} max={aucs.max():.4f}"
        )

    # ---------------- feature-importance diagnostic on the FINAL checkpoint ----------------
    final_step = max(results.keys())
    final_ckpt = [c for c in checkpoints if int(c.stem.split("_")[-1]) == final_step][0]
    G_final = load_generator(final_ckpt, device)
    close_paths = synthesize_close_paths(G_final, n_real, normalizer, device, rng_seed=999)
    fake_feats_final = featurize_batch(close_paths)

    ranked, diag_test_auc = feature_importance_diagnostic(real_feats, fake_feats_final, seed=42)
    print(f"\nFeature importance diagnostic (final checkpoint, held-out test AUC={diag_test_auc:.4f}):")
    print("Top features by permutation importance (mean ROC-AUC drop when shuffled):")
    for name, val in ranked[:8]:
        print(f"  {name:20s} {val:+.4f}")

    # ---------------- save everything ----------------
    out = {
        "window_len": WINDOW_LEN,
        "n_real_windows": n_real,
        "n_repeats": N_REPEATS,
        "reality_floor_at_L": {
            "mean": float(floor_aucs.mean()),
            "std": float(floor_aucs.std(ddof=1)),
            "min": float(floor_aucs.min()),
            "max": float(floor_aucs.max()),
            "aucs": floor_aucs.tolist(),
        },
        "checkpoints": results,
        "final_checkpoint_step": final_step,
        "feature_importance": {
            "test_auc": diag_test_auc,
            "ranked": [(name, float(val)) for name, val in ranked],
        },
    }
    out_path = RESULTS_DIR / "step2_realism_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")

    try:
        import matplotlib.pyplot as plt

        steps = sorted(results.keys())
        means = [results[s]["mean"] for s in steps]
        stds = [results[s]["std"] for s in steps]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(steps, means, yerr=stds, marker="o", capsize=3, label="real vs. synthetic AUC")
        ax.axhline(floor_aucs.mean(), color="gray", linestyle="--", label=f"reality floor (mean={floor_aucs.mean():.3f})")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="AUC = 0.5")
        ax.set_xlabel("generator training step")
        ax.set_ylabel("ROC-AUC (real vs. synthetic)")
        ax.set_title("Step 2: realism (real-vs-synthetic AUC) over training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "step2_auc_over_training.png", dpi=120)
        print(f"Saved plot to {RESULTS_DIR / 'step2_auc_over_training.png'}")
    except ImportError:
        print("matplotlib not available, skipped plot")


if __name__ == "__main__":
    main()
