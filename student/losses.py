"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


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
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
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
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    obs_noise_std = float(loss_cfg.get("obs_noise_std", 0.0))
    one = one_step_delta_loss(model, states, actions, normalizer, obs_noise_std=obs_noise_std)

    max_horizon = int(loss_cfg.get("rollout_train_horizon", 50))
    min_horizon = int(loss_cfg.get("rollout_min_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 10))

    available = int(states.shape[1]) - warmup - 1
    upper = max(min_horizon, min(max_horizon, available))
    lower = min(min_horizon, upper)
    if upper > lower:
        horizon = int(torch.randint(lower, upper + 1, (), device=states.device).item())
    else:
        horizon = upper

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

    delta_reg_weight = float(loss_cfg.get("delta_reg_weight", 0.0))
    if delta_reg_weight > 0.0:
        slack = float(loss_cfg.get("delta_reg_slack", 5.0))
        excess = torch.clamp(pred_norm.abs() - slack, min=0.0)
        delta_reg = (excess ** 2).mean()
    else:
        delta_reg = torch.zeros((), device=states.device, dtype=states.dtype)

    one_w = float(loss_cfg.get("one_step_weight", 1.0))
    roll_w = float(loss_cfg.get("rollout_weight", 1.0))
    total = one_w * one + roll_w * roll + delta_reg_weight * delta_reg

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/delta_reg": float(delta_reg.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
    }
