#!/usr/bin/env bash
# Re-run all offline scoring from existing inference results (no GPU needed).
#
# Produces the aggregate tables under data/scores/ and data/supplementary/.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Main scoring (FAR / Recall / F1 / Hazard Alignment / Phantom Contamination) ..."
python -m benchmark.score.scorer_v2

echo "[2/3] EGAV mitigation (Table 3 / Table 4) ..."
python -m supplementary_experiments.mitigation.score_mitigation

echo "[3/3] Visual-noise probe (Appendix C) ..."
python -m supplementary_experiments.visual_noise.score_visual_noise

echo "Done."
