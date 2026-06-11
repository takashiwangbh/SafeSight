# SafeSight: Auditing False Alarms in Embodied Hazard Perception

This repository is the **anonymized code-and-results artifact** accompanying the
paper *“SafeSight: Auditing False Alarms in Embodied Hazard Perception.”* It is
released for peer review and contains no author-identifying information.

---

## 1. Introduction

Embodied agents increasingly rely on large language and vision-language models
(LLMs/VLMs) to perceive household hazards such as running faucets, broken glass,
or unattended stoves. Benchmarks that evaluate this ability mainly through
**recall on hazardous scenes** reward a degenerate strategy: predicting
*“dangerous”* for every input. SafeSight measures the complementary failure —
**over-prediction (false alarms) on matched safe scenes** — and several
diagnostic behaviors that recall alone cannot capture.

The project has two clearly separated layers :

- **SafeSight (framework).** An extensible, AI2-THOR-grounded pipeline:
  simulator-grounded hazard operationalization, safe-control construction,
  semantic sensor-noise injection, model-adapter evaluation, and scoring. This
  is the layer for defining **new hazards, rooms, or noise models**.
- **SafeSight-Bench (frozen instantiation).** The fixed, all-in-one scene set
  produced by the framework (33 hazard templates yielding 891 hazardous + 356
  safe clean scenes, plus pre-generated noisy variants). This is the layer for
  **reproducible, out-of-the-box comparison**.

In short: evaluating a new model uses the frozen bench; studying new
hazards/noise uses the framework.

---

## 2. Code Contributions

SafeSight contributes a complete, AI2-THOR-grounded auditing stack: an
affordance-grounded hazard-operationalization layer that turns 33 manually
curated hazard templates into simulator-executable, metadata-verifiable
hazardous scenes with matched safe controls; a reusable semantic sensor-noise
engine with eight perturbation primitives across six intensity levels; a
three-track evaluation harness (text-only, image-only, text+image) built on a
unified output schema and a robust multi-stage parser; and a scoring layer that
introduces three diagnostic metrics beyond recall — False-Alarm Rate (FAR),
Hazard Alignment (HA), and Phantom Contamination (PC). It further includes
EGAV, a training-free post-hoc gate that suppresses ungrounded false alarms; an
RGB-level physical-corruption probe; and a machine-consensus cross-verification
of the simulator labels.

---

## 3. Dataset Volume (SafeSight-Bench)

| Quantity | Value |
|----------|-------|
| Hazard templates | **33** (15 kitchen, 7 bathroom, 6 living room, 5 bedroom) |
| Room types / floor plans | 4 /  120 iTHOR floor plans |
| Clean hazardous scenes | **891** |
| Clean safe control scenes | **356** (≈3 viewpoints per floor plan, 119 plans) |
| Clean base observations | **1,247** |
| Noisy hazardous variants | **16,038** (891 × 6 levels × 3 seeds) |
| Noisy safe variants | **1,424** (356 × 4 levels × 1 seed) |
| Total clean + noisy scene files | **18,709** |
| Noise primitives / intensity levels | 8 / 6 |
| Audited models / (model, track) configs | 12 / 22 |
| Raw model-inference JSON results | **142,542** |
| Standardized scoring records | **137,182** |
| Cross-verification subset | 500 scenes (250 hazardous + 250 safe); 456 sensor-observable |

Diagnostic metrics reported: Precision, Recall, F1, FAR, Hazard Alignment, and
Phantom Contamination.

---

## 4. Repository Distribution

This repository ships all source code together with the aggregate result tables
and a curated set of example renders. The full scene corpus and per-record
outputs are withheld at review time and will be released upon acceptance.

| Path | Status | Notes |
|------|--------|-------|
| `benchmark/`, `supplementary_experiments/`, `cross_verify/` | Included | the SafeSight framework source |
| `main.py`, `run_benchmark.py`, `llm_client.py`, `thor_wrapper.py` | Included | entry points / shared utilities |
| `data/scores/` | Included | aggregate CSVs backing the main tables/figures |
| `data/supplementary/` | Included | aggregate CSVs for Table 3/4 (EGAV) and Appendix C |
| `data/figures/` | Included | ~60 curated renders (hazards, safe controls, visual-noise examples) |
| Full scene corpus (18,709 files) | Withheld | regenerable; released on acceptance |
| Raw / per-record model outputs (142,542 / 137,182) | Withheld | released on acceptance |

```
safesight-anon/
├── benchmark/                 # SafeSight framework (harvest / noise / evaluate / score)
├── supplementary_experiments/ # EGAV mitigation + RGB-noise probe
├── cross_verify/             # machine-consensus label cross-verification
├── data/
│   ├── scores/               # aggregate scoring CSVs (main results)
│   ├── supplementary/        # Table 3/4 and Appendix C CSVs
│   └── figures/              # curated example renders
├── main.py  run_benchmark.py  llm_client.py  thor_wrapper.py
├── requirements.txt  LICENSE  .gitignore  README.md
```

---

## Notes

- Hazard labels are operationalized through AI2-THOR object states, affordances,
  and action outcomes, giving every scene executable, metadata-verifiable
  grounding.
- The complete corpus is large: the 18,709 rendered scene files plus the 142,542
  raw model-output JSON files total over **2 GB**, which is why the full
  dataset is not shipped inside this review repository. The aggregate CSVs under
  `data/` reproduce every number reported in the paper, and the full corpus will
  be released upon acceptance.
- No secrets are stored in code. The optional OpenAI-backed juror in
  `cross_verify/` reads its key only from the `OPENAI_API_KEY` environment
  variable.
