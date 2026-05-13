# EEC289A HW2 — InvertedPendulum World Model

## Score (public_scoreboard split, max_horizon=1000)

| Split | VPT80@0.25 | VPT50@0.25 | nMSE@10 | nMSE@100 | nMSE@1000 | nMSE_AUC |
|------:|-----------:|-----------:|--------:|---------:|----------:|---------:|
| test  | **18**     | 22         | 0.0066  | 0.853    | 1.71      | 1.39     |
| ood   | **18**     | 22         | 0.0111  | 0.815    | 1.68      | 1.37     |

Checkpoint selected at update 16,500 of 18,000 by `val/VPT80@0.25` at validation horizon 300.

## What I changed

### `student/model.py` — Residual MLP with LayerNorm

Replaced the 2-layer SiLU MLP with 4 **pre-LN residual blocks** of width 256.
Each block: `LayerNorm → Linear → SiLU → Linear → residual add`. Output is a
`LayerNorm`, a head producing the normalized delta, and the original
`delta_limit * tanh(raw / delta_limit)` bound. Output head is initialized with
`std=1e-3` so the first updates produce small deltas. The `use_gru` flag is
retained but disabled (state is fully Markov).

### `student/losses.py` — Random-horizon rollout + obs noise

- Sample a random rollout horizon per batch uniformly in
  `[rollout_min_horizon, rollout_train_horizon] = [5, 60]`. Random start
  position within the window is preserved. This acts as a free curriculum and
  exposes the model to gradient through varied-length rollouts without needing
  a step-aware scheduler (the locked `train.py` does not expose one).
- Small Gaussian input noise (`obs_noise_std = 0.02`) on normalized
  observations during one-step training, to harden the model against the small
  state-space drift produced by its own predictions during long rollouts.
- A clipped L2 penalty on normalized predicted deltas (weight 1e-4, slack 5.0)
  that only fires when the model tries to predict a delta well outside
  realistic per-step changes. Never activated during training (kept as a
  safety net).

### `student/rollout.py` — Optional truncated BPTT

Added an optional `tbptt_chunk` argument that detaches the rolled-out state
every `tbptt_chunk` steps. Unused in the final submission (TBPTT trained
medium-horizon error down but cost too much single-step accuracy → lower
VPT80) but left in place for future experiments. Default behavior unchanged.

### `configs/student.yaml`

```yaml
model: { hidden_dim: 256, num_layers: 4, use_gru: false }
training:
  batch_size: 256
  updates: 18000
  train_sequence_length: 128
  learning_rate: 5.0e-4
  grad_clip_norm: 2.0
  val_horizon: 300
  checkpoint_metric: val/VPT80@0.25
loss:
  one_step_weight: 3.0          # keeps per-step error tight
  rollout_weight: 1.0
  rollout_train_horizon: 60     # max; random per batch
  rollout_min_horizon: 5
  obs_noise_std: 0.02           # denoising training
```

## What didn't work (and why)

I ran four ablations beyond the v1 baseline before settling on this design.
Public-scoreboard `test` VPT80@0.25 in parentheses.

- **v1** *(17)*. Same model, fixed-horizon rollout=15 (starter setting),
  one-step weight 1.0, 12k updates, no obs noise. Solid baseline.
- **v2** *(7)*. Rollout horizon raised to `[15, 100]` with `rollout_weight=2.0`.
  Hurt single-step accuracy enough to drop VPT80 by 10. Lesson: VPT80@0.25
  punishes any sacrifice of per-step nMSE, even if medium-horizon error
  improves.
- **v3** *(18 — submitted)*. Heavier one-step weight (3.0), modest rollout
  horizon `[5, 60]`, denoising noise 0.02, 18k updates. Best balance.
- **v4** *(≤7, killed at 1 k of 15 k)*. Long sequences (256) + rollout
  horizon `[10, 120]`. Too slow and showed the same v2 trade-off early.
- **v5** *(≤7, killed at 7.5 k)*. Added TBPTT (chunk=20) to allow horizon=150
  rollout training. nMSE@100 dropped to 0.27 (much better than v3's 0.85) but
  nMSE@1 climbed to 0.01 (v3 holds 0.0005). VPT80 again hurt by the
  tighter-on-average / weaker-at-step-1 trade-off.

The recurring lesson: with `tanh`-bounded normalized residuals, **single-step
fidelity dominates VPT80@0.25**. Any loss design that lets the rollout
gradients drag one-step error above ~0.001 loses on the primary metric, even
if it improves the long-horizon curve. v3 keeps one-step rmse at ~0.005 and
that is what produced the best VPT.

## Files modified

- `student/model.py`
- `student/losses.py`
- `student/rollout.py` *(TBPTT plumbing, default behaviour unchanged)*
- `configs/student.yaml`

## How to reproduce

```bash
python -m wm_hw.dataset    --config configs/public_scoreboard.yaml --output-dir data/public_scoreboard
python -m wm_hw.train      --config configs/student.yaml --model student \
                           --dataset-dir data/public_scoreboard \
                           --output-dir artifacts/student
python -m wm_hw.eval_horizon --checkpoint-dir artifacts/student/best_checkpoint \
                             --dataset-dir data/public_scoreboard --split test \
                             --eval-config configs/official_eval.yaml \
                             --output-dir artifacts/student/eval_test
python -m wm_hw.eval_horizon --checkpoint-dir artifacts/student/best_checkpoint \
                             --dataset-dir data/public_scoreboard --split ood \
                             --eval-config configs/official_eval.yaml \
                             --output-dir artifacts/student/eval_ood
python -m wm_hw.plotting --eval-dir artifacts/student/eval_test \
                         --output-dir artifacts/student/plots
```

Training ran on a single RTX 4090 in ~25 minutes (18k updates @ batch 256).
All 13 `pytest -q -m "not slow"` tests pass with the modified student files.
