# trader_evolution_v2

## Goal

Build a trading model that beats buy-and-hold and other trivial strategies on
**unseen** data. The intended path there: train the model on a synthetic-data
*generator*, not directly on the limited real history. For that to be a valid
shortcut, the generator has to be validated as indistinguishable from reality
first — otherwise the model just learns to exploit generator artifacts.

"Indistinguishable" needs a numeric target, though. A classifier trained to
tell generated data from real data should get an AUC no better than a
classifier trained to tell two random halves of *real* data apart from each
other. That second number — how well real BTC can be told apart from itself
— is the **reality floor**, and it's what this first module measures.

## Module 1: reality floor (`reality_floor/`)

### Data

`data/BTCUSDT_4h_full.csv` — BTC/USDT 4h OHLCV, the full continuous history
available: **2017-08-17 04:00 UTC to 2026-04-16 08:00 UTC, 18,970 bars**
(one coin, no resampling gaps worth excluding — 18,961 of the 18,969
consecutive bar gaps are exactly 4h; the rest are a handful of isolated
missing bars).

### Method

For window length L in {7 days (42 bars), 14 days (84 bars), 30 days (180
bars)}:

1. Slice the full close-price series into non-overlapping windows of L bars
   (any leftover tail shorter than L is dropped).
2. Featurize each window with stylized-fact summary stats only (no raw
   prices): annualized vol, skew, excess kurtosis, return autocorrelation at
   lags 1-3, |return| autocorrelation (vol clustering) at lags 1, 2, 3, 5,
   return quantiles q01/q05/q95/q99, max drawdown, longest up-run and
   down-run. See `reality_floor/features.py`.
3. Randomly split the windows 50/50 into group A / group B — a shuffle over
   *all* windows, not a time split, so both groups draw from every era (2021
   bull, 2022 crash, 2023 chop, ...). The question is "real vs. real", not
   "bull era vs. calm era".
4. Train a `HistGradientBoostingClassifier` to tell A from B, evaluated by
   5-fold stratified cross-validation (out-of-fold ROC-AUC).
5. Repeat steps 3-4 twenty times with fresh random splits (seeded, so the run
   is deterministic) and report the AUC distribution instead of one number,
   since the random split itself is a noise source.

Run it:

```
python3 reality_floor/run.py
```

Results are written to `results/reality_floor_results.csv`.

### Results

| window | bars | n_windows | mean AUC | std AUC | min AUC | max AUC |
|--------|-----:|----------:|---------:|--------:|--------:|--------:|
| 7d     |   42 |       451 |   0.5117 |  0.0448 |  0.4140 |  0.5892 |
| 14d    |   84 |       225 |   0.4841 |  0.0584 |  0.4140 |  0.5990 |
| 30d    |  180 |       105 |   0.5122 |  0.0804 |  0.3462 |  0.6676 |

(20 random-split repeats per window length, base seed 42.)

### Reading

At all three scales, real-vs-real AUC sits **around 0.5** — real BTC 4h
price action is not internally distinguishable across eras by these
stylized-fact features. There's no evidence of an elevated floor here: 0.5 is
the target, not e.g. 0.65.

What differs across window lengths is the *noise* in that estimate, driven
by how many non-overlapping windows the history yields:

- **7d (n=451)**: tightest distribution (std 0.045), mean 0.512, range
  [0.414, 0.589] — a clean, reliable ~0.5.
- **14d (n=225)**: mean actually dips slightly below 0.5 (0.484), std 0.058
  — still consistent with 0.5 given the noise, but the estimate is looser.
- **30d (n=105)**: mean 0.512 but std 0.080 and a wide range [0.346, 0.668]
  — individual splits swing far from 0.5 purely from having only ~105
  windows to split and cross-validate over. The mean is fine; any single
  run is not trustworthy at this scale.

**7-day windows give the cleanest ~0.5** — most windows, lowest variance,
tightest bracket around 0.5 — making it the best scale to build and validate
the generator against. 30-day windows would need a longer history (or
overlapping/bootstrapped windows) before their AUC estimate is trustworthy
enough to use as a target.
