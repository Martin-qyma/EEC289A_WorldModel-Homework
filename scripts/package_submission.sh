#!/usr/bin/env bash
# Build the final submission bundle from a trained run directory.
#
# Usage:
#   bash scripts/package_submission.sh <run_dir> <submission_dir>
# Example:
#   bash scripts/package_submission.sh artifacts/run_v2 artifacts/submission_final

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <run_dir> <submission_dir>" >&2
    exit 1
fi

RUN_DIR=$(realpath "$1")
SUB_DIR=$(realpath -m "$2")
DATA_DIR=${DATA_DIR:-data/public_scoreboard}
EVAL_CFG=${EVAL_CFG:-configs/official_eval.yaml}

echo "[package] run_dir=$RUN_DIR"
echo "[package] sub_dir=$SUB_DIR"
echo "[package] data=$DATA_DIR"

mkdir -p "$SUB_DIR"
cp -r "$RUN_DIR/best_checkpoint" "$SUB_DIR/best_checkpoint"
cp "$RUN_DIR/normalizer.json" "$SUB_DIR/normalizer.json"
[ -f "$RUN_DIR/train.log" ] && cp "$RUN_DIR/train.log" "$SUB_DIR/train.log"
[ -f "$RUN_DIR/train_summary.json" ] && cp "$RUN_DIR/train_summary.json" "$SUB_DIR/train_summary.json"

python -m wm_hw.eval_horizon \
    --checkpoint-dir "$SUB_DIR/best_checkpoint" \
    --dataset-dir "$DATA_DIR" --split test \
    --eval-config "$EVAL_CFG" \
    --output-dir "$SUB_DIR/eval_test"
python -m wm_hw.eval_horizon \
    --checkpoint-dir "$SUB_DIR/best_checkpoint" \
    --dataset-dir "$DATA_DIR" --split ood \
    --eval-config "$EVAL_CFG" \
    --output-dir "$SUB_DIR/eval_ood"
python -m wm_hw.plotting \
    --eval-dir "$SUB_DIR/eval_test" \
    --output-dir "$SUB_DIR/plots"

python <<PY
import json
with open("${SUB_DIR}/eval_test/scoreboard_summary.json") as f:
    test = json.load(f)
with open("${SUB_DIR}/eval_ood/scoreboard_summary.json") as f:
    ood = json.load(f)
print("=== TEST ===")
print(json.dumps(test, indent=2))
print("=== OOD ===")
print(json.dumps(ood, indent=2))
PY

# Zip the final artifacts.
(cd "$(dirname "$SUB_DIR")" && zip -r "$(basename "$SUB_DIR")/final_artifacts.zip" \
    "$(basename "$SUB_DIR")/best_checkpoint" \
    "$(basename "$SUB_DIR")/eval_test" \
    "$(basename "$SUB_DIR")/eval_ood" \
    "$(basename "$SUB_DIR")/plots" \
    "$(basename "$SUB_DIR")/normalizer.json" \
    "$(basename "$SUB_DIR")/train.log" \
    "$(basename "$SUB_DIR")/train_summary.json" 2>/dev/null || true)

echo "[package] Done -> $SUB_DIR"
