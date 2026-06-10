# Supplementary Experiments

Two focused studies that reuse the existing inference results and rendered
scenes rather than re-running the full benchmark. All outputs are written under
`data/supplementary/`; the main pipeline is untouched.

| Study | Paper | Summary |
|-------|-------|---------|
| `mitigation/` | **Table 3 / Table 4** | **EGAV** evidence-grounded alarm verification: a training-free, inference-time post-hoc gate that roughly halves FAR while *improving* F1. |
| `visual_noise/` | **Appendix C** | RGB-level physical-corruption feasibility probe (Gaussian sensor noise, low illumination, motion blur) on the image-only track. |

## Mitigation (EGAV)

Three defences are compared; the first two are deliberately weak baselines whose
failure motivates the proposed method:

- **Confidence gate (baseline).** Downgrade a *dangerous* verdict whose
  self-reported confidence is below a threshold. Fails because the audited
  yes-man alarms are high-confidence (≈0.88–0.90).
- **Self-correction re-prompt (baseline).** A strict adversarial second pass.
  Fails because it suppresses real hazards too, collapsing recall.
- **EGAV (proposed).** Keeps a *dangerous* verdict only when it is grounded in
  observable object-state evidence (an abnormal state flag, or in the balanced
  `state_object` mode a pinned sharp/breakable danger object); otherwise it
  overturns the verdict to *safe*. Pure post-processing — no GPU, no retraining.

Headline artifacts: `data/supplementary/mitigation/table4_egav.csv`
(full-population FAR / Recall / F1 before→after) and `table4_sample_pools.csv`
(baselines vs EGAV).

## Visual Noise (Appendix C)

A small probe corrupts clean-correct hazardous image-only scenes with three
physical sensor degradations and re-measures recall against the clean control.
The injector uses only numpy + Pillow. Headline artifacts:
`appendixC_recall_by_condition.csv` and `appendixC_recall_by_severity.csv`.

## Notes

- `common.py` reuses the main benchmark's scoring helpers so all numbers stay
  consistent with `benchmark/score/scorer_v2.py`.
- EGAV itself needs no GPU; the self-correction baseline requires re-inference
  only to quantify its failure.
