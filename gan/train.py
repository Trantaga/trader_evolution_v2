"""WGAN-GP training script for 168-bar BTC OHLC windows.

General-purpose training script, configurable via N_GEN_STEPS / RUN_TAG:
  - Step 1 (committed as `results/gan/step1_*`): a short 300-step run that
    only verified the loop runs stably and output is structurally valid
    OHLC -- not tuned for realism.
  - Step 2 (this default config, RUN_TAG="step2"): a proper ~20k-step run,
    with periodic generator checkpoints (GAN quality isn't monotonic in
    step count -- we want to be able to pick the best checkpoint, not just
    the last) so `gan/measure_realism.py` can track how real-vs-synthetic
    AUC evolves over training.

Runs locally on Apple Silicon (device auto-picks MPS -> CUDA -> CPU) or on
Colab GPU unchanged: upload/clone this repo and run `python gan/train.py`;
no code changes needed, only the device that gets picked.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gan.data import (
    fit_normalizer,
    load_ohlc,
    make_windows,
    ohlc_to_features,
    reconstruct_ohlc,
    window_anchor_prices,
)
from gan.model import Critic, Generator, N_CHANNELS, WINDOW_LEN

# ----------------------------------- CONFIG -----------------------------------
SEED = 42

WINDOW_STRIDE = 8  # sliding-window stride (bars) for training data; overlap = augmentation

Z_DIM = 128
GEN_BASE_CHANNELS = 128
CRITIC_BASE_CHANNELS = 128

BATCH_SIZE = 32  # modest, fits comfortably on an 8GB M1
LR = 1e-4
BETAS = (0.0, 0.9)  # standard WGAN-GP Adam betas
N_CRITIC = 5  # critic updates per generator step
LAMBDA_GP = 10.0

N_GEN_STEPS = 20_000  # proper Step-2 run; Step 1 used 300 (skeleton check only)
LOG_EVERY = 100
N_VALIDITY_CHECK = 512  # generated windows to structurally validate after training
N_SAMPLE_PATHS = 4  # sample generated price paths to plot

CHECKPOINT_EVERY = 2000  # save G periodically -- GAN quality isn't monotonic in step count
THROUGHPUT_CHUNK = 2000  # steps per throughput-timing chunk, to spot thermal throttling

RUN_TAG = "step2"  # prefixes output filenames; keeps Step 1's committed artifacts intact

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "gan"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
# --------------------------------------------------------------------------------


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: GANs are not bit-for-bit deterministic (WGAN-GP's double backward
    # for the gradient penalty, plus MPS/CUDA non-determinism in conv kernels)
    # -- this seeds everything that CAN be seeded, but exact loss values will
    # still vary run to run and especially across devices (Mac vs Colab).


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> torch.Tensor:
    b = real.size(0)
    alpha = torch.rand(b, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.reshape(b, -1)
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def verify_data_roundtrip(df, normalizer, windows: np.ndarray, anchors: np.ndarray) -> None:
    """Sanity check on the pipeline itself: reconstructing a few REAL windows
    from their standardized features should reproduce the original OHLC
    almost exactly (float precision only). This is separate from the
    structural-validity check on GENERATED windows below."""
    idxs = [0, len(windows) // 2, len(windows) - 1]
    max_rel_err = 0.0
    for i in idxs:
        recon = reconstruct_ohlc(windows[i], normalizer, anchor_price=anchors[i])
        first_bar_df_idx = i * WINDOW_STRIDE + 1
        true_open = df["open"].iloc[first_bar_df_idx]
        true_close = df["close"].iloc[first_bar_df_idx + WINDOW_LEN - 1]
        rel_err_open = abs(recon["open"][0] - true_open) / true_open
        rel_err_close = abs(recon["close"][-1] - true_close) / true_close
        max_rel_err = max(max_rel_err, rel_err_open, rel_err_close)
    print(f"Real-data round-trip check (reconstruct_ohlc vs. actual prices): "
          f"max relative error = {max_rel_err:.2e} (float32 precision)")


def main() -> None:
    seed_everything(SEED)
    device = pick_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------- data ----------------------------
    df = load_ohlc()
    feats = ohlc_to_features(df)
    normalizer = fit_normalizer(feats)
    z_feats = normalizer.standardize(feats)
    windows = make_windows(z_feats, WINDOW_LEN, WINDOW_STRIDE)  # [n, 4, 168]
    anchors = window_anchor_prices(df, WINDOW_LEN, WINDOW_STRIDE)
    print(
        f"Loaded {len(df)} bars -> {len(feats)} feature bars -> {len(windows)} training windows "
        f"(window={WINDOW_LEN} bars = 28d, stride={WINDOW_STRIDE} bars)"
    )
    verify_data_roundtrip(df, normalizer, windows, anchors)

    data_t = torch.tensor(windows, dtype=torch.float32, device=device)

    # ---------------------------- models ----------------------------
    G = Generator(z_dim=Z_DIM, base_channels=GEN_BASE_CHANNELS).to(device)
    C = Critic(base_channels=CRITIC_BASE_CHANNELS).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_c = torch.optim.Adam(C.parameters(), lr=LR, betas=BETAS)
    print(
        f"Generator params: {sum(p.numel() for p in G.parameters()):,}  "
        f"Critic params: {sum(p.numel() for p in C.parameters()):,}"
    )

    n = data_t.size(0)

    def sample_real_batch() -> torch.Tensor:
        idx = torch.randint(0, n, (BATCH_SIZE,), device=device)
        return data_t[idx]

    # ---------------------------- training loop ----------------------------
    log = []
    throughput_log = []  # (chunk_end_step, ms_per_gen_step) -- thermal-throttle diagnostic
    t0 = time.time()
    t_chunk_start = t0
    for step in range(1, N_GEN_STEPS + 1):
        for _ in range(N_CRITIC):
            real = sample_real_batch()
            z = torch.randn(BATCH_SIZE, Z_DIM, device=device)
            with torch.no_grad():
                fake = G(z)
            c_real = C(real)
            c_fake = C(fake)
            gp = gradient_penalty(C, real, fake, device)
            c_loss = c_fake.mean() - c_real.mean() + LAMBDA_GP * gp

            opt_c.zero_grad(set_to_none=True)
            c_loss.backward()
            opt_c.step()

        z = torch.randn(BATCH_SIZE, Z_DIM, device=device)
        fake = G(z)
        g_loss = -C(fake).mean()

        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        if step == 1 or step % LOG_EVERY == 0:
            entry = dict(
                step=step,
                critic_loss=float(c_loss.item()),
                wasserstein_dist=float((c_real.mean() - c_fake.mean()).item()),
                gen_loss=float(g_loss.item()),
                gp=float(gp.item()),
            )
            log.append(entry)
            print(
                f"step {step:5d}  critic_loss={entry['critic_loss']:+8.4f}  "
                f"W_dist={entry['wasserstein_dist']:+8.4f}  "
                f"gen_loss={entry['gen_loss']:+8.4f}  gp={entry['gp']:.4f}"
            )

        if step % CHECKPOINT_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f"gen_step_{step:06d}.pt"
            torch.save(G.state_dict(), ckpt_path)
            print(f"  -> saved checkpoint {ckpt_path.name}")

        if step % THROUGHPUT_CHUNK == 0:
            now = time.time()
            ms_per_step = (now - t_chunk_start) / THROUGHPUT_CHUNK * 1000
            throughput_log.append(dict(chunk_end_step=step, ms_per_gen_step=ms_per_step))
            t_chunk_start = now

    elapsed = time.time() - t0
    print(
        f"\nTraining ({N_GEN_STEPS} generator steps x {N_CRITIC} critic steps) took "
        f"{elapsed:.1f}s on {device} ({elapsed / N_GEN_STEPS * 1000:.1f} ms/gen-step avg)"
    )

    first_chunk_ms = throughput_log[0]["ms_per_gen_step"]
    last_chunk_ms = throughput_log[-1]["ms_per_gen_step"]
    slowdown_pct = (last_chunk_ms / first_chunk_ms - 1) * 100
    print(
        f"Throughput: first {THROUGHPUT_CHUNK}-step chunk = {first_chunk_ms:.1f} ms/step, "
        f"last chunk = {last_chunk_ms:.1f} ms/step ({slowdown_pct:+.0f}% change) "
        f"{'-- looks like thermal throttling' if slowdown_pct > 20 else '-- no notable throttling'}"
    )

    all_finite = all(
        np.isfinite(e[k]) for e in log for k in ("critic_loss", "wasserstein_dist", "gen_loss", "gp")
    )
    print(f"All logged losses finite (no NaN/Inf): {all_finite}")

    with open(RESULTS_DIR / f"{RUN_TAG}_loss_log.json", "w") as f:
        json.dump(
            {
                "device": str(device),
                "elapsed_sec": elapsed,
                "n_gen_steps": N_GEN_STEPS,
                "n_critic": N_CRITIC,
                "batch_size": BATCH_SIZE,
                "n_windows": int(n),
                "throughput_log": throughput_log,
                "log": log,
            },
            f,
            indent=2,
        )

    # ---------------------------- post-training checks ----------------------------
    G.eval()
    with torch.no_grad():
        z = torch.randn(N_VALIDITY_CHECK, Z_DIM, device=device)
        fake = G(z).cpu().numpy()
    ohlc = reconstruct_ohlc(fake, normalizer, anchor_price=1.0)
    o, h, l, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]

    cond_h = bool((h >= np.maximum(o, c) - 1e-6).all())
    cond_l = bool((l <= np.minimum(o, c) + 1e-6).all())
    cond_hl = bool((h >= l - 1e-6).all())
    finite = bool(np.isfinite(o).all() and np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all())

    print(f"\nStructural validity on {N_VALIDITY_CHECK} generated windows:")
    print(f"  high >= max(open, close): {cond_h}")
    print(f"  low  <= min(open, close): {cond_l}")
    print(f"  high >= low             : {cond_hl}")
    print(f"  all OHLC values finite  : {finite}")

    validity = dict(
        n_checked=N_VALIDITY_CHECK,
        high_ge_body=cond_h,
        low_le_body=cond_l,
        high_ge_low=cond_hl,
        all_finite=finite,
        close_min=float(c.min()),
        close_max=float(c.max()),
    )
    with open(RESULTS_DIR / f"{RUN_TAG}_validity_check.json", "w") as f:
        json.dump(validity, f, indent=2)

    # a few sample paths for the README/plot
    sample_paths = c[:N_SAMPLE_PATHS].tolist()
    with open(RESULTS_DIR / f"{RUN_TAG}_sample_paths.json", "w") as f:
        json.dump({"close_paths": sample_paths}, f)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(N_SAMPLE_PATHS):
            ax.plot(c[i], label=f"sample {i}")
        ax.set_title(f"{RUN_TAG}: sample generated close-price paths")
        ax.set_xlabel("bar (4h)")
        ax.set_ylabel("price (anchor = 1.0)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{RUN_TAG}_sample_paths.png", dpi=120)
        print(f"\nSaved sample-path plot to {RESULTS_DIR / f'{RUN_TAG}_sample_paths.png'}")
    except ImportError:
        print("matplotlib not available, skipped plot (JSON paths still saved)")


if __name__ == "__main__":
    main()
