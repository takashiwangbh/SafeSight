#!/usr/bin/env bash
# Supplementary experiments runner (mitigation + visual noise).
#
# Run from the repository root (GPU required for stage 2):
#     bash supplementary_experiments/run_all.sh            # full
#     bash supplementary_experiments/run_all.sh --smoke    # tiny sanity run
#
# Stages:
#   1. offline sample selection (no GPU)
#   2. GPU re-inference (self-correction + corrupted-image eval)
#   3. offline scoring → Table 4 + Appendix C csvs
#
# Offline stages (1 & 3) work anywhere; stage 2 needs the models + GPUs.

set -euo pipefail

SMOKE=0
N_FP=100
N_TP=50
N_VIS=50
EXTRA_RUN_FLAGS=""

for arg in "$@"; do
  case "$arg" in
    --smoke)
      SMOKE=1; N_FP=10; N_TP=5; N_VIS=5
      EXTRA_RUN_FLAGS="--max-per-model 10"
      ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

mkdir -p logs
TS="$(date +%Y%m%d_%H%M)"

echo "================================================================"
echo "[1/3] Offline sample selection"
echo "================================================================"
python -m supplementary_experiments.mitigation.select_fp_samples \
  --n-fp "$N_FP" --n-tp "$N_TP"
python -m supplementary_experiments.visual_noise.select_visual_samples \
  --n "$N_VIS"

echo
echo "================================================================"
echo "[2/3] GPU re-inference"
echo "================================================================"
echo ">> Mitigation: cognitive self-correction"
python -u -m supplementary_experiments.mitigation.run_self_correction \
  ${EXTRA_RUN_FLAGS} 2>&1 | tee "logs/supp_selfcorrect_${TS}.log"

VIS_FLAGS=""
if [[ "$SMOKE" -eq 1 ]]; then VIS_FLAGS="--max-scenes 5"; fi
echo ">> Visual noise: corrupt + re-eval image track"
python -u -m supplementary_experiments.visual_noise.run_visual_noise_eval \
  ${VIS_FLAGS} 2>&1 | tee "logs/supp_visualnoise_${TS}.log"

echo
echo "================================================================"
echo "[3/3] Offline scoring"
echo "================================================================"
python -m supplementary_experiments.mitigation.score_mitigation
python -m supplementary_experiments.visual_noise.score_visual_noise

echo
echo "All done. Artefacts under data/supplementary/"
