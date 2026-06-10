#!/usr/bin/env bash
# Cross-Verification pipeline runner.
#
# Usage (run from the repository root):
#     bash cross_verify/run_pipeline.sh                 # full default
#     bash cross_verify/run_pipeline.sh --smoke         # 5-scene sanity run
#
# Environment expectations:
#   - HF_HOME / HUGGINGFACE_HUB_CACHE set to a writable cache (/data/...)
#   - bitsandbytes installed (needed for unsloth bnb-4bit jurors)
#   - dual GPU recommended; one A6000 is enough since jurors load one
#     at a time.
#   - OPENAI_API_KEY exported when the registry includes any
#     provider="openai" juror (e.g. gpt-5.5).  The script warns and
#     skips the OpenAI step if the key is missing.

set -euo pipefail

# ─── Args ───────────────────────────────────────────────────────────────
SMOKE=0
N_HAZARD=250
N_SAFE=250
SEED=42

for arg in "$@"; do
  case "$arg" in
    --smoke)
      SMOKE=1
      N_HAZARD=10
      N_SAFE=10
      ;;
    --seed=*)
      SEED="${arg#*=}"
      ;;
    --n-hazard=*)
      N_HAZARD="${arg#*=}"
      ;;
    --n-safe=*)
      N_SAFE="${arg#*=}"
      ;;
    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

mkdir -p logs

# ─── Env ───────────────────────────────────────────────────────────────
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HOME/.cache/huggingface_modules}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================================================================"
echo "[1/3] Stratified scene sampling"
echo "================================================================"
python -u -m cross_verify.sample_scenes \
  --n-hazard "$N_HAZARD" --n-safe "$N_SAFE" --seed "$SEED" \
  2>&1 | tee "logs/cv_sample_$(date +%Y%m%d_%H%M).log"

echo
echo "================================================================"
echo "[2/3] Jury evaluation"
echo "================================================================"
JURY_FLAGS=()
if [[ "$SMOKE" -eq 1 ]]; then
  JURY_FLAGS+=(--max-scenes 20)
fi

# Detect OpenAI jurors in the registry and warn if the API key is missing.
HAS_OPENAI_JURY=$(python - <<'PY'
import sys
sys.path.insert(0, '.')
from cross_verify.config import JURY_MODELS
print("yes" if any(j.get("provider") == "openai" for j in JURY_MODELS) else "no")
PY
)
if [[ "$HAS_OPENAI_JURY" == "yes" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "⚠  OPENAI_API_KEY is not set, but the registry contains an"
    echo "   OpenAI juror.  The OpenAI step will be skipped.  To run it:"
    echo "       export OPENAI_API_KEY=sk-..."
    echo "       python -m cross_verify.jury_eval --jury gpt-5.5"
  else
    echo "✓  OPENAI_API_KEY detected (length=${#OPENAI_API_KEY})."
  fi
fi

python -u -m cross_verify.jury_eval "${JURY_FLAGS[@]}" \
  2>&1 | tee "logs/cv_jury_$(date +%Y%m%d_%H%M).log"

echo
echo "================================================================"
echo "[3/3] Consensus metrics"
echo "================================================================"
python -u -m cross_verify.consensus \
  2>&1 | tee "logs/cv_consensus_$(date +%Y%m%d_%H%M).log"

echo
echo "Pipeline complete. Results in data/cross_verify/"
