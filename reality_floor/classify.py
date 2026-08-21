"""Real-vs-real classifiability: random A/B split of windows + CV AUC, repeated."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

N_CV_FOLDS = 5


def cv_auc(features: pd.DataFrame, labels: np.ndarray, clf_seed: int, cv_seed: int) -> float:
    """5-fold CV ROC-AUC of a HistGradientBoostingClassifier distinguishing the
    two groups in `labels`, via out-of-fold predictions. This is the single
    classifier definition shared by the real-vs-real reality-floor
    measurement (this module) and the GAN's real-vs-synthetic realism
    measurement (gan/measure_realism.py) -- same model, same CV scheme, so
    the two AUC numbers are directly comparable."""
    clf = HistGradientBoostingClassifier(random_state=clf_seed)
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=cv_seed)
    oof_proba = cross_val_predict(clf, features.values, labels, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, oof_proba))


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

    clf_seed = int(rng.integers(0, 2**31 - 1))
    cv_seed = int(rng.integers(0, 2**31 - 1))
    return cv_auc(features, labels, clf_seed, cv_seed)


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
