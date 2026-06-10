#!/usr/bin/env bash
# Evaluate one (model, track) on the frozen SafeSight-Bench scenes, then score.
#
# Prereqs:
#   - pip install -r requirements.txt
#   - the frozen scene set placed under data/scenes/ and data/safe_scenes/
#     (released on acceptance; paths are configured in benchmark/config.py)
#   - a CUDA-capable GPU for local model inference
#
# Usage:
#   bash scripts/run_eval.sh <model_short_name> [track]
#   track ∈ {text_only, image_only, text_and_image}   (default: text_and_image)

set -euo pipefail

MODEL="${1:?usage: run_eval.sh <model> [track]}"
TRACK="${2:-text_and_image}"

cd "$(dirname "$0")/.."

echo "[1/2] Evaluating ${MODEL} on track=${TRACK} ..."
python -m benchmark.evaluate.baseline_eval --model "${MODEL}" --track "${TRACK}"

echo "[2/2] Scoring ..."
python -m benchmark.score.scorer_v2

echo "Done. See data/scores/ for FAR / Recall / F1 / HA / PC."
