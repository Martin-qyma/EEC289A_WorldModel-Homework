"""Student loss — scheduled one-step → rollout curriculum.

The kinematic prior in `student/model.py` already makes the position
channels exactly determined by the predicted velocity, so the loss has
to push the velocity prediction in two regimes:

1. **Per-step regime (first ~3000 updates).** Pure one-step MSE.
   Builds an accurate single-step velocity prediction (`nMSE@1 ≈ 0`)
   without any rollout-loss interference. This is what makes `nMSE@10`
   close to zero.

2. **Rollout regime (after `rollout_warmup_steps`).** The one-step
   loss is kept on a *very small* weight, and a long open-loop rollout
   loss with truncated BPTT teaches the model to recover from drifted
   states. This is the only mechanism that breaks the open-loop
   unstable dynamics on the angle channel (eigenvalue ≈ 1.21) — once
   the model can predict correctly *from* drifted obs, the error stops
   compounding in the open-loop rollout.

The schedule is implemented by ramping `rollout_weight` from 0 to its
configured value linearly over `rollout_warmup_steps` updates, plus a
matching ramp of the *rollout horizon* from `rollout_min_horizon` up
to `rollout_train_horizon`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


_GLOBAL_STEP = {"value": 0}


def reset_global_step() -> None:
    _GLOBAL_STEP["value"] = 0


def current_global_step() -> int:
    return _GLOBAL_STEP["value"]


def _ensure_normalizer_synced(model, normalizer) -> None:
    if not hasattr(model, "set_normalizer"):
        return
    if getattr(model, "_normalizer_synced", False):
        return
    device = next(model.parameters()).device
    model.set_normalizer(
        torch.as_tensor(normalizer.obs_mean, dtype=torch.float32, device=device),
        torch.as_tensor(normalizer.obs_std, dtype=torch.float32, device=device),
        torch.as_tensor(normalizer.act_mean, dtype=torch.float32, device=device),
        torch.as_tensor(normalizer.act_std, dtype=torch.float32, device=device),
        torch.as_tensor(normalizer.delta_mean, dtype=torch.float32, device=device),
        torch.as_tensor(normalizer.delta_std, dtype=torch.float32, device=device),
    )
    model._normalizer_synced = True


def one_step_delta_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    *,
    obs_noise_std: float = 0.0,
) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    if obs_noise_std > 0.0 and model.training:
        obs_norm = obs_norm + obs_noise_std * torch.randn_like(obs_norm)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    tbptt_chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for "
            f"warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(
        model,
        sub_states,
        sub_actions,
        normalizer,
        warmup_steps=warmup_steps,
        horizon=horizon,
        tbptt_chunk=tbptt_chunk,
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    loss = F.mse_loss(pred_norm, target_norm)
    return loss, pred_norm


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    _ensure_normalizer_synced(model, normalizer)
    _GLOBAL_STEP["value"] += 1
    step = _GLOBAL_STEP["value"]

    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    obs_noise_std = float(loss_cfg.get("obs_noise_std", 0.0))
    one = one_step_delta_loss(
        model, states, actions, normalizer, obs_noise_std=obs_noise_std
    )

    # Rollout ramp: `rollout_weight = configured * min(1, (step - delay) / warmup)`.
    rollout_delay = int(loss_cfg.get("rollout_delay_steps", 0))
    rollout_warmup = int(loss_cfg.get("rollout_warmup_steps", 0))
    if step < rollout_delay:
        rollout_ramp = 0.0
    elif rollout_warmup > 0:
        rollout_ramp = min(1.0, (step - rollout_delay) / float(rollout_warmup))
    else:
        rollout_ramp = 1.0

    base_rollout_weight = float(loss_cfg.get("rollout_weight", 0.0))
    roll_w = base_rollout_weight * rollout_ramp

    # Horizon ramp.
    max_horizon = int(loss_cfg.get("rollout_train_horizon", 100))
    min_horizon = int(loss_cfg.get("rollout_min_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 10))
    available = int(states.shape[1]) - warmup - 1
    upper_cap = max(1, min(max_horizon, available))
    lower_cap = max(1, min(min_horizon, upper_cap))
    # Ramp current `upper` from `lower_cap` to `upper_cap` with the
    # rollout-weight schedule.
    cur_upper = int(round(lower_cap + rollout_ramp * (upper_cap - lower_cap)))
    cur_upper = max(lower_cap, cur_upper)
    if loss_cfg.get("rollout_fixed_horizon", False):
        horizon = cur_upper
    elif cur_upper > lower_cap:
        horizon = int(
            torch.randint(lower_cap, cur_upper + 1, (), device=states.device).item()
        )
    else:
        horizon = cur_upper

    if roll_w > 0.0:
        tbptt_chunk = loss_cfg.get("tbptt_chunk", None)
        roll, pred_norm = rollout_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps=warmup,
            horizon=horizon,
            tbptt_chunk=tbptt_chunk,
        )
    else:
        roll = torch.zeros((), device=states.device, dtype=states.dtype)
        pred_norm = torch.zeros(1, 1, device=states.device, dtype=states.dtype)

    delta_reg_weight = float(loss_cfg.get("delta_reg_weight", 0.0))
    if delta_reg_weight > 0.0:
        slack = float(loss_cfg.get("delta_reg_slack", 5.0))
        excess = torch.clamp(pred_norm.abs() - slack, min=0.0)
        delta_reg = (excess ** 2).mean()
    else:
        delta_reg = torch.zeros((), device=states.device, dtype=states.dtype)

    one_w = float(loss_cfg.get("one_step_weight", 1.0))
    total = one_w * one + roll_w * roll + delta_reg_weight * delta_reg

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/delta_reg": float(delta_reg.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
        "loss/rollout_weight": float(roll_w),
    }
