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

## Module 2: WGAN-GP generator (`gan/`)

Target: a generator for 7-day (168-bar) BTC OHLC windows whose synthetic
windows score ~0.5 AUC against real ones on the reality-floor test above.
This is **Step 1 only** — build the skeleton and prove the training loop
runs stably and produces structurally valid OHLC. Not tuned for realism yet
(no reality-floor AUC run against it here — that's Step 2).

### How to run

**Locally (Mac, MPS)**:

```
pip install -r requirements.txt
python3 gan/train.py
```

Device is auto-selected in order MPS → CUDA → CPU, printed at the start of
the run. On a fanless M1 (e.g. MacBook Air), sustained long runs may
thermally throttle and slow down mid-run — that's expected, not a bug.

**Later, on Colab (GPU)**: upload or `git clone` this repo and run the exact
same `python gan/train.py` — no code changes needed. The device-selection
code will pick up CUDA automatically since MPS won't be available there.

All hyperparameters (`Z_DIM`, batch size, learning rate, `N_CRITIC`,
`LAMBDA_GP`, window stride, step count, ...) live in a config block at the
top of `gan/train.py`.

### OHLC parametrization (structural validity by construction)

The generator never outputs 4 independent prices. Per bar it outputs, in log
space relative to the previous bar's close:

- `r` — close-to-close log return (unconstrained)
- `gap` — open's log gap from the previous close (unconstrained)
- `h_off` — high-wick offset above the bar body, passed through `softplus`
  so it's **always ≥ 0**
- `l_off` — low-wick offset below the bar body, also `softplus`'d to
  **always ≥ 0**

OHLC is then reconstructed deterministically: cumulative-sum `r` for the
close path, add `gap` to the previous close for opens, then push high above
`max(open,close)` and low below `min(open,close)` by the non-negative
offsets. Because `h_off, l_off ≥ 0` is guaranteed by `softplus` for *any*
network weights, `high ≥ max(open,close)`, `low ≤ min(open,close)`, and
`high ≥ low` hold structurally — not something the GAN has to learn. See
`gan/data.py` (transform + inverse transform) and `gan/model.py`
(`Generator.forward`).

Real data is losslessly re-expressed the same way for training (so the
critic sees the same 4-channel representation for real and fake): `r`/`gap`
are z-scored, `h_off`/`l_off` are scale-only normalized (divided by their
mean magnitude, no centering) so real data's non-negativity is preserved
too. Training windows are overlapping (stride 8 bars, i.e. 1-day steps)
for data augmentation — unlike the reality-floor module, which needs
non-overlapping windows for statistical independence, the GAN just needs
many varied examples to train on.

### Architecture

- **Generator**: `z ~ N(0,I)`, dim 128 → linear → reshape → 3×
  `ConvTranspose1d` upsampling blocks (21 → 42 → 84 → 168, BatchNorm + ReLU)
  → 1×1-ish conv head → 4 raw channels → `softplus` on the two offset
  channels. ~393K params.
- **Critic**: `[4, 168]` → 3× `Conv1d` downsampling blocks (168 → 84 → 42 →
  21, GroupNorm + LeakyReLU — **no BatchNorm**, since WGAN-GP's gradient
  penalty is per-sample and BatchNorm couples samples in a batch, which
  breaks it) → flatten → linear → scalar (no sigmoid). ~45K params.
- **WGAN-GP loss**: gradient penalty on random real/fake interpolates,
  `n_critic=5` critic updates per generator step, Adam(β=(0.0, 0.9)),
  lr 1e-4, λ_gp=10, batch size 32.

### Step-1 results

Ran `gan/train.py` unmodified (300 generator steps × 5 critic steps = 1,500
critic updates, batch size 32, seed 42):

```
Using device: mps
Loaded 18970 bars -> 18969 feature bars -> 2351 training windows (window=168 bars = 7d, stride=8 bars)
Real-data round-trip check (reconstruct_ohlc vs. actual prices): max relative error = 1.45e-15
Generator params: 392,580  Critic params: 44,833
step    1  critic_loss= +1.7533  W_dist= +0.3261  gen_loss= +0.6288  gp=0.2079
step   20  critic_loss= -7.5647  W_dist= +9.2927  gen_loss= +9.0827  gp=0.1728
step  100  critic_loss=-13.3628  W_dist=+18.7493  gen_loss=+32.1792  gp=0.5387
step  200  critic_loss=-11.7413  W_dist=+14.0106  gen_loss=+44.0872  gp=0.2269
step  300  critic_loss=-12.4748  W_dist=+14.2416  gen_loss=+52.1502  gp=0.1767

Training (300 generator steps x 5 critic steps) took 16.8s on mps (56.0 ms/gen-step)
All logged losses finite (no NaN/Inf): True

Structural validity on 512 generated windows:
  high >= max(open, close): True
  low  <= min(open, close): True
  high >= low             : True
  all OHLC values finite  : True
```

Full log: `results/gan/step1_loss_log.json`. Validity check:
`results/gan/step1_validity_check.json`. Sample generated close-price paths
(anchor = 1.0): `results/gan/step1_sample_paths.png` — plausibly-shaped
random walks, no blow-up, no flat lines, not realistic yet (expected — no
realism tuning has happened).

**Reading the loss trace**: everything stays finite throughout, and the
gradient penalty stays small and bounded (~0.05–0.5, i.e. the critic's
gradient norm stays close to the target of 1) — the Lipschitz constraint is
being enforced correctly, no explosion. The Wasserstein distance estimate
grows rather than shrinks over this short run; that's expected, not a bug:
with `n_critic=5` the critic quickly gets good at separating real windows
from an as-yet-undertrained generator's output, and 300 generator steps is
far too short to expect convergence — Step 1 only needs to show the loop is
numerically stable, which it is.

**Timing**: 16.8s of training for 300 generator steps on an M1 (MPS) — about
56ms/step. At that rate, even a much longer realism-tuning run (e.g. 20,000
generator steps) would be roughly 20 minutes locally, so **local M1 training
looks practical**; Colab likely isn't necessary unless Step 2 needs
substantially more capacity or a much longer run runs into thermal
throttling on the fanless Air.
