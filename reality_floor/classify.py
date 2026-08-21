"""Real-vs-real classifiability: random A/B split of windows + CV AUC, repeated."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

N_CV_FOLDS = 5


def random_ab_auc(features: pd.DataFrame, rng: np.random.Generator) -> float:
    """One trial: randomly split windows 50/50 into group A / group B (label),
    then measure how well a classifier tells them apart via 5-fold CV AUC on
    out-of-fold predictions. Since A/B membership is pure noise, this AUC
    isolates how distinguishable real windows are from *other real windows*
    of the same length, regardless of era."""
    n = len(features)
    labels = np.zeros(n, dtype=int)
    labels[: n // 2] = 1
    rng.shuffle(labels)

    clf = HistGradientBoostingClassifier(random_state=int(rng.integers(0, 2**31 - 1)))
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))

    oof_proba = cross_val_predict(clf, features.values, labels, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, oof_proba))


def repeated_real_vs_real_auc(features: pd.DataFrame, n_repeats: int, seed: int) -> np.ndarray:
    """Repeat the random-split + CV-AUC trial `n_repeats` times with a
    deterministic seed sequence, returning the array of AUC values."""
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_repeats)
    aucs = np.empty(n_repeats)
    for i, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        aucs[i] = random_ab_auc(features, rng)
    return aucs
