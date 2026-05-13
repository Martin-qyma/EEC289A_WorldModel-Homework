"""Student open-loop rollout implementation."""

from __future__ import annotations

import torch

from wm_hw.model_utils import predict_next


def open_loop_rollout(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    tbptt_chunk: int | None = None,
):
    """Roll out `horizon` steps after a ground-truth warmup.

    Future ground-truth states after `warmup_steps` must not be read.

    When `tbptt_chunk` is set, gradients are detached every `tbptt_chunk` steps
    of the rollout. This caps backprop length to stabilize long-horizon training
    without leaking ground truth into the prediction trajectory.
    """
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)
    for t in range(int(warmup_steps)):
        _, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)
    cur = states[:, int(warmup_steps)]
    preds = []
    chunk = int(tbptt_chunk) if tbptt_chunk is not None and int(tbptt_chunk) > 0 else 0
    for h in range(int(horizon)):
        if chunk > 0 and h > 0 and h % chunk == 0:
            cur = cur.detach()
            if hidden is not None:
                hidden = hidden.detach()
        cur, hidden = predict_next(model, cur, actions[:, int(warmup_steps) + h], hidden, normalizer)
        preds.append(cur)
    return torch.stack(preds, dim=1)
