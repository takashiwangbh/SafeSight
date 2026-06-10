# Supplementary Experiments

Two focused studies that reuse the existing inference results and rendered
scenes — no full re-run of the benchmark is required. All outputs are written
under `data/supplementary/`; the main pipeline is untouched.

| Study | Paper | Summary |
|-------|-------|---------|
| `mitigation/` | **Table 3 / Table 4** | **EGAV** evidence-grounded alarm verification: a training-free, inference-time post-hoc gate that roughly halves FAR while *improving* F1. |
| `visual_noise/` | **Appendix C** | RGB-level physical-corruption feasibility probe (Gaussian sensor noise, low illumination, motion blur) on the image-only track. |

## Mitigation (EGAV)

Three defences are compared; the first two are deliberately weak baselines
whose failure motivates the proposed method:

- **Confidence gate (baseline).** Downgrade a *dangerous* verdict whose
  self-reported confidence is below a threshold. Fails because the audited
  yes-man alarms are high-confidence (≈0.88–0.90).
- **Self-correction re-prompt (baseline).** A strict adversarial second pass.
  Fails because it suppresses real hazards too, collapsing recall.
- **EGAV (proposed).** Keep a *dangerous* verdict only when it is grounded in
  observable object-state evidence (abnormal state flag, or in the balanced
  `state_object` mode a pinned sharp/breakable danger object); otherwise
  overturn to *safe*. Pure post-processing, no GPU, no retraining.

```bash
# Offline scoring (reads existing first-pass results + scene metadata)
python -m supplementary_experiments.mitigation.score_mitigation
```

Headline outputs: `data/supplementary/mitigation/table4_egav.csv` (full-population
FAR / Recall / F1 before→after) and `table4_sample_pools.csv` (baselines vs EGAV).

## Visual Noise (Appendix C)

```bash
# Stage 1 (offline): pick clean-correct hazardous image-only scenes
python -m supplementary_experiments.visual_noise.select_visual_samples --n 50
# Stage 2 (GPU): corrupt images and re-run image-only inference
python -m supplementary_experiments.visual_noise.run_visual_noise_eval
# Stage 3 (offline): score recall vs the clean control
python -m supplementary_experiments.visual_noise.score_visual_noise
```

The injector (`visual_noise_injector.py`) uses only numpy + Pillow (no OpenCV).

## Notes

- `common.py` reuses the main benchmark's scoring helpers so all numbers stay
  consistent with `benchmark/score/scorer_v2.py`.
- `run_self_correction.py` (GPU) only exists to quantify the self-correction
  baseline's failure; EGAV itself needs no GPU.
