#!/usr/bin/env python
"""Pick the best checkpoint among candidate runs and package as the final
submission.

Usage:
  python scripts/pick_best.py \
      --candidate artifacts/run_final/best_checkpoint \
      --candidate artifacts/run_seed42/best_checkpoint \
      --candidate artifacts/submission_v1_backup/best_checkpoint \
      --dataset-dir data/public_scoreboard \
      --eval-config configs/official_eval.yaml \
      --output-dir artifacts/submission_final
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def evaluate(checkpoint_dir: Path, dataset_dir: Path, eval_config: Path, split: str) -> dict:
    out = Path("/tmp") / f"pick_best_eval_{checkpoint_dir.parent.name}_{split}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wm_hw.eval_horizon",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            split,
            "--eval-config",
            str(eval_config),
            "--output-dir",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    with (out / "scoreboard_summary.json").open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    eval_config = Path(args.eval_config)
    candidates = []
    for cand in args.candidate:
        cand_path = Path(cand)
        if not (cand_path / "checkpoint.pt").exists():
            print(f"  [skip] {cand_path}: no checkpoint.pt")
            continue
        try:
            test_metrics = evaluate(cand_path, dataset_dir, eval_config, "test")
        except subprocess.CalledProcessError as e:
            print(f"  [skip] {cand_path}: eval failed:\n{e.stderr.decode() if e.stderr else e}")
            continue
        candidates.append((cand_path, test_metrics))
        print(
            f"  [cand] {cand_path.parent.name}: VPT80={test_metrics['VPT80@0.25']} "
            f"nMSE@10={test_metrics['nMSE@10']:.4f} "
            f"nMSE@100={test_metrics['nMSE@100']:.4f} "
            f"nMSE@1000={test_metrics['nMSE@1000']:.4f}"
        )

    if not candidates:
        raise RuntimeError("No usable candidate checkpoints.")

    # Sort by (VPT80 desc, nMSE_AUC asc)
    candidates.sort(
        key=lambda x: (-int(x[1]["VPT80@0.25"]), float(x[1]["nMSE_AUC"]))
    )
    best_path, best_metrics = candidates[0]
    print(f"\n>>> best: {best_path.parent.name} VPT80={best_metrics['VPT80@0.25']}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copytree(best_path, output_dir / "best_checkpoint")
    nz = best_path.parent / "normalizer.json"
    if nz.exists():
        shutil.copy2(nz, output_dir / "normalizer.json")
    train_log = best_path.parent / "train.log"
    if train_log.exists():
        shutil.copy2(train_log, output_dir / "train.log")
    train_sum = best_path.parent / "train_summary.json"
    if train_sum.exists():
        shutil.copy2(train_sum, output_dir / "train_summary.json")

    # Re-run final eval to populate output_dir/eval_{test,ood}
    for split in ("test", "ood"):
        eval_out = output_dir / f"eval_{split}"
        if eval_out.exists():
            shutil.rmtree(eval_out)
        eval_out.mkdir(parents=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wm_hw.eval_horizon",
                "--checkpoint-dir",
                str(output_dir / "best_checkpoint"),
                "--dataset-dir",
                str(dataset_dir),
                "--split",
                split,
                "--eval-config",
                str(eval_config),
                "--output-dir",
                str(eval_out),
            ],
            check=True,
        )

    # Plot
    plot_out = output_dir / "plots"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wm_hw.plotting",
            "--eval-dir",
            str(output_dir / "eval_test"),
            "--output-dir",
            str(plot_out),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
