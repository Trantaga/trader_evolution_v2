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

Target: a generator for 168-bar BTC OHLC windows whose synthetic windows
score ~0.5 AUC against real ones on the reality-floor test above.

**Window length note**: 168 4h-bars is **28 days**, not 7 — a labeling slip
from Step 1 (6 bars/day × 7 = 42, which is what `reality_floor/` actually
calls "7d"). The architecture was already built and trained around
`WINDOW_LEN=168` before this was caught, so Step 2 keeps it and instead
computes the reality floor *at L=168* directly (see Results below) for a
correct apples-to-apples comparison, rather than misusing the 7d/42-bar
floor number. Worth fixing (retrain at 42 bars for a true 7-day window) in
a later step if 7 days specifically matters, e.g. for shaping training runs
downstream.

### How to run

**Locally (Mac, MPS)**:

```
pip install -r requirements.txt
python3 gan/train.py           # train, checkpointing every CHECKPOINT_EVERY steps
python3 gan/measure_realism.py # measure real-vs-synthetic AUC across saved checkpoints
```

Device is auto-selected in order MPS → CUDA → CPU, printed at the start of
the run. On a fanless M1 (e.g. MacBook Air), sustained long runs may
thermally throttle and slow down mid-run — that's expected, not a bug (see
Step 2 below, where it happened).

**Later, on Colab (GPU)**: upload or `git clone` this repo and run the exact
same commands — no code changes needed. The device-selection code will pick
up CUDA automatically since MPS won't be available there.

All hyperparameters (`Z_DIM`, batch size, learning rate, `N_CRITIC`,
`LAMBDA_GP`, window stride, step count, checkpoint interval, `RUN_TAG`, ...)
live in a config block at the top of `gan/train.py`. `gan/train.py` is
general-purpose — `RUN_TAG` picks which output filenames a run writes, so
Step 1's short verification run and Step 2's proper run don't overwrite each
other's artifacts.

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

### Step-1 results (skeleton verification only, not tuned for realism)

Ran `gan/train.py` with `N_GEN_STEPS=300` (300 generator steps × 5 critic
steps = 1,500 critic updates, batch size 32, seed 42):

```
Using device: mps
Loaded 18970 bars -> 18969 feature bars -> 2351 training windows (window=168 bars = 7d, stride=8 bars)
  [verbatim historical log -- the "= 7d" label was wrong, see the window-length note above; fixed in code since]
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

### Step-2 results: proper training run + first realism measurement

Ran `gan/train.py` with `N_GEN_STEPS=20_000`, `CHECKPOINT_EVERY=2000` (same
architecture/hyperparameters as Step 1, `RUN_TAG="step2"`), then measured
realism with `gan/measure_realism.py` against **every** saved checkpoint.

**Training**:

```
Training (20000 generator steps x 5 critic steps) took 1174.0s on mps (58.7 ms/gen-step avg)
Throughput: first 2000-step chunk = 55.2 ms/step, last chunk = 87.0 ms/step (+58% change) -- looks like thermal throttling
All logged losses finite (no NaN/Inf): True

Structural validity on 512 generated windows: all four checks True (as in Step 1)
```

19.6 minutes total, matching the Step-1 extrapolation. **Thermal throttling
did happen** on the fanless M1 Air — per-step time grew 58% from the first
2,000-step chunk to the last — but the run stayed numerically stable
throughout (all losses finite, 10/10 checkpoints valid OHLC). Full log with
per-chunk throughput: `results/gan/step2_loss_log.json`.

The Wasserstein estimate itself is noisy and doesn't show clean monotonic
convergence over this run (bounces roughly 5–30 throughout, gradient penalty
stays bounded so training is stable, just not settling) — which is why the
next section, not the loss curve, is the real signal to watch.

**Realism measurement** (reusing `reality_floor`'s exact classifier +
features, unchanged; real windows are 112 **non-overlapping** 168-bar slices
— disjoint from the overlapping stride-8 windows used for training; 20
repeats of fresh-synthetic-batch + 5-fold CV per checkpoint):

Reality floor at L=168 (real-vs-real, freshly computed for this exact window
length): **mean 0.5106, std 0.0825, min 0.3839, max 0.6604** (20 repeats).

| step | AUC mean | std | min | max |
|-----:|---------:|----:|----:|----:|
| 2,000  | 0.8414 | 0.0214 | 0.7978 | 0.8811 |
| 4,000  | 0.7996 | 0.0270 | 0.7471 | 0.8505 |
| 6,000  | 0.7469 | 0.0369 | 0.6939 | 0.8118 |
| 8,000  | 0.6847 | 0.0279 | 0.6083 | 0.7256 |
| 10,000 | 0.6502 | 0.0452 | 0.5670 | 0.7423 |
| 12,000 | 0.6472 | 0.0552 | 0.5564 | 0.7455 |
| 14,000 | 0.6504 | 0.0385 | 0.5859 | 0.7218 |
| 16,000 | 0.6359 | 0.0366 | 0.5519 | 0.6999 |
| 18,000 | 0.6223 | 0.0372 | 0.5285 | 0.6975 |
| 20,000 (final) | 0.6106 | 0.0444 | 0.5414 | 0.6845 |

Plot: `results/gan/step2_auc_over_training.png`. Full numbers (all 20
AUCs per checkpoint): `results/gan/step2_realism_results.json`.

**Top feature importances** (final checkpoint, permutation importance —
`HistGradientBoostingClassifier` has no built-in `feature_importances_` —
mean ROC-AUC drop on a held-out test split when each feature is shuffled,
30 repeats):

| feature | importance (AUC drop) |
|---|---:|
| `acf_ret_1` (return autocorrelation, lag 1) | +0.0616 |
| `longest_up_run` | +0.0508 |
| `q99` (99th-pct return) | +0.0223 |
| `acf_absret_5` (vol-clustering acf, lag 5) | +0.0099 |
| `q95` (95th-pct return) | +0.0050 |
| `skew` | +0.0005 |
| `excess_kurtosis` | -0.0019 |
| `q05` | -0.0025 |

### Reading

**AUC is dropping steadily, not stuck near 1.0 — we're on the right
track.** It falls from 0.84 at step 2,000 to 0.61 at step 20,000, more than
halving the gap to the 0.51 floor (gap went from ~0.33 to ~0.10). The drop
is fastest early (2k→10k) and slows down 10k→20k but is still trending
down at the last checkpoint, not flat — there's no evidence of a plateau
yet, so more steps (and/or the tuning Step 3 will do) should keep helping
rather than hitting a wall. Also notable: **the run is not monotonic
checkpoint-to-checkpoint** (12k ticks up slightly from 10k) — exactly why
Step 2 saved checkpoints throughout rather than only the final one; the
best checkpoint so far is the final one (20,000), but that could change
with more training.

**Biggest unrealistic feature: `acf_ret_1`, by a wide margin** (more than
double the next-highest importance), with `longest_up_run` close behind.
Both point at the same underlying problem, visible directly in
`results/gan/step2_sample_paths.png`: generated windows show long, smooth,
persistently-directional runs — real BTC log-returns are close to
uncorrelated bar-to-bar, but the generator has learned momentum instead of
noise. The tail-quantile features (`q99`, `q95`) and lag-5 vol-clustering
show up next but much weaker, and `skew`/`excess_kurtosis`/`q05` barely
move the needle at all — the generator's tails and asymmetry are already
roughly plausible; what's off is the trend structure. **Attack lag-1 return
autocorrelation (and run-length) next** — that's the single biggest lever
for the next tuning step.
