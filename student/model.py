"""Student world model — trapezoidal kinematic MLP.

Two structural facts about MuJoCo's `InvertedPendulum-v5` integrator,
verified on the public-scoreboard training split:

* the position/angle channels obey the trapezoidal identity exactly
  (residual std 7e-6 / 4e-5):

      pos_{t+1}   = pos_t   + dt/2 · (cart_vel_t  + cart_vel_{t+1})
      angle_{t+1} = angle_t + dt/2 · (pole_vel_t  + pole_vel_{t+1})

  with `dt = 0.04`; and

* the *velocity* dynamics are mostly linear in `(obs, action)`
  (residual std ≈5 % of delta-std for both `Δcart_vel` and `Δpole_vel`).

We therefore use a tiny architecture: an MLP predicts the two velocity
deltas, and the position-angle update is produced analytically by the
trapezoidal identity. The benefit: per-step error in *position* grows
linearly (not exponentially) given a correct velocity prediction.

The buffers (`obs_mean_buf`, …) are populated on the first
`compute_loss` call so the forward can un-normalise to real units;
they serialise into the state-dict so eval-time `load_state_dict`
restores them automatically.
"""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_DT = 0.04
IDX_POS, IDX_ANGLE, IDX_VEL, IDX_ANG_VEL = 0, 1, 2, 3


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_gru: bool = False,
        delta_limit: float = 6.0,
        dropout: float = 0.0,
        dt: float = DEFAULT_DT,
    ):
        super().__init__()
        if obs_dim != 4 or act_dim != 1:
            raise ValueError(
                "kinematic StudentWorldModel only supports "
                "(obs_dim=4, act_dim=1) — got "
                f"({obs_dim=}, {act_dim=})"
            )
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.dt = float(dt)

        in_dim = obs_dim + act_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.act_in = nn.SiLU()
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_dim, dropout=dropout) for _ in range(int(num_layers))]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        # Output is the two velocity-delta channels in normalised-delta
        # space; position-angle deltas are computed analytically below.
        self.head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

        self.register_buffer("obs_mean_buf", torch.zeros(obs_dim))
        self.register_buffer("obs_std_buf", torch.ones(obs_dim))
        self.register_buffer("act_mean_buf", torch.zeros(act_dim))
        self.register_buffer("act_std_buf", torch.ones(act_dim))
        self.register_buffer("delta_mean_buf", torch.zeros(obs_dim))
        self.register_buffer("delta_std_buf", torch.ones(obs_dim))

    def set_normalizer(
        self,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        act_mean: torch.Tensor,
        act_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.obs_mean_buf.copy_(obs_mean.to(self.obs_mean_buf))
            self.obs_std_buf.copy_(obs_std.to(self.obs_std_buf))
            self.act_mean_buf.copy_(act_mean.to(self.act_mean_buf))
            self.act_std_buf.copy_(act_std.to(self.act_std_buf))
            self.delta_mean_buf.copy_(delta_mean.to(self.delta_mean_buf))
            self.delta_std_buf.copy_(delta_std.to(self.delta_std_buf))

    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        obs_raw = obs_norm * self.obs_std_buf + self.obs_mean_buf
        vel_t = obs_raw[..., IDX_VEL]
        ang_vel_t = obs_raw[..., IDX_ANG_VEL]

        feat = self.input_proj(torch.cat([obs_norm, act_norm], dim=-1))
        feat = self.act_in(feat)
        for block in self.blocks:
            feat = block(feat)
        feat = self.out_norm(feat)
        raw_d_vel = self.head(feat)
        clamp = self.delta_limit * torch.tanh(raw_d_vel / self.delta_limit)
        d_vel_norm = clamp[..., 0]
        d_ang_norm = clamp[..., 1]
        d_vel_raw = (
            d_vel_norm * self.delta_std_buf[IDX_VEL] + self.delta_mean_buf[IDX_VEL]
        )
        d_ang_raw = (
            d_ang_norm * self.delta_std_buf[IDX_ANG_VEL]
            + self.delta_mean_buf[IDX_ANG_VEL]
        )

        # Trapezoidal kinematic update for the position channels.
        half_dt = 0.5 * self.dt
        d_pos_raw = self.dt * vel_t + half_dt * d_vel_raw
        d_angle_raw = self.dt * ang_vel_t + half_dt * d_ang_raw

        delta_raw = torch.stack(
            [d_pos_raw, d_angle_raw, d_vel_raw, d_ang_raw], dim=-1
        )
        delta_norm = (delta_raw - self.delta_mean_buf) / self.delta_std_buf
        return delta_norm, hidden
