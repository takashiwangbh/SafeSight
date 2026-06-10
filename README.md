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

The project has two clearly separated layers (this distinction is the answer to
the common “is this a dataset or a framework?” question):

- **SafeSight (framework).** An extensible, AI2-THOR-grounded pipeline:
  simulator-grounded hazard operationalization → safe-control construction →
  semantic sensor-noise injection → model-adapter evaluation → scoring.
  Use this if you want to define **new hazards, rooms, or noise models**.
- **SafeSight-Bench (frozen instantiation).** The fixed, all-in-one scene set
  produced by the framework (33 hazard templates → 891 hazardous + 356 safe
  clean scenes, plus pre-generated noisy variants). Use this if you just want
  **reproducible, out-of-the-box comparison**: implement a model adapter and run
  evaluation + scoring. No regeneration required.

In short: **evaluating a new model needs only the frozen bench + an adapter;
studying new hazards/noise needs the framework.**

---

## 2. Code Contributions

| Module | What it provides |
|--------|------------------|
| `benchmark/harvest/` | Affordance-grounded hazard operationalization. `danger_recipes.py` defines the 33 hazard templates as formal tuples ⟨required objects, setup actions, danger labels, severity, safe/unsafe responses⟩; `harvest.py` / `safe_harvest.py` instantiate, validate (Algorithm 1: each setup action accepted only if the affordance exists and `lastActionSuccess` is true), render, and serialize hazardous / matched-safe scenes. |
| `benchmark/noise/` | `noise_engine.py` — eight semantic sensor-noise primitives (label swap, state flip, info drop, position jitter, distance warp, phantom inject, property corrupt, sensor blackout) over six intensity levels λ ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}. |
| `benchmark/evaluate/` | Three input tracks (text-only / image-only / text+image), a unified output schema (`assessment`, `confidence`, `hazards_detected`, `reasoning`, `action`), and a multi-stage parser with keyword fallback (`prompts.py`, `vlm_client.py`, `evaluator.py`, `baseline_eval.py`, `noisy_eval.py`). |
| `benchmark/score/` | Scoring with three diagnostic metrics beyond P/R/F1: **False-Alarm Rate (FAR)** on safe scenes, **Hazard Alignment (HA)**, and **Phantom Contamination (PC)** (`scorer_v2.py`). |
| `supplementary_experiments/mitigation/` | **EGAV** (Embodied/Evidence-Grounded Alarm Verification): a training-free, inference-time post-hoc gate that keeps a *dangerous* verdict only when it is grounded in observable object-state evidence. Roughly halves FAR while improving F1 (paper Table 3/4). |
| `supplementary_experiments/visual_noise/` | RGB-level physical-corruption feasibility probe (Gaussian sensor noise, low illumination, motion blur) using numpy + Pillow only (paper Appendix C). |
| `cross_verify/` | Machine-consensus cross-verification: an independent large-model jury checks the simulator-derived labels (paper Section 7). See `cross_verify/README.md`. |

---

## 3. Dataset Volume (SafeSight-Bench)

| Quantity | Value |
|----------|-------|
| Hazard templates | **33** (15 kitchen, 7 bathroom, 6 living room, 5 bedroom) |
| Room types / floor plans | 4 / up to 120 iTHOR floor plans |
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

## 4. Repository Distribution (what is / isn’t in this anonymous repo)

To keep the review artifact lightweight and shareable, this repository ships
**all source code** plus the **aggregate result tables** and a **curated set of
example renders**. The full multi-gigabyte scene corpus and per-record outputs
are withheld at review time and **will be released upon acceptance**.

| Path | Included? | Notes |
|------|-----------|-------|
| `benchmark/`, `supplementary_experiments/`, `cross_verify/` | ✅ full source | the SafeSight framework |
| `main.py`, `run_benchmark.py`, `llm_client.py`, `thor_wrapper.py` | ✅ | entry points / shared utilities |
| `data/scores/` | ✅ aggregate CSVs | source numbers for the main tables/figures |
| `data/supplementary/` | ✅ aggregate CSVs | Table 3/4 (EGAV) and Appendix C |
| `data/figures/` | ✅ ~60 curated renders | hazards (1 per template), safe controls, visual-noise examples |
| Full scene corpus (18,709 files) | ⛔ withheld | regenerable via `benchmark/harvest/`; released on acceptance |
| Raw / per-record model outputs (142,542 / 137,182) | ⛔ withheld | released on acceptance |

```
safesight-anon/
├── benchmark/                 # SafeSight framework (harvest / noise / evaluate / score)
├── supplementary_experiments/ # EGAV mitigation + RGB-noise probe
├── cross_verify/             # machine-consensus label cross-verification
├── scripts/                  # convenience runners
├── data/
│   ├── scores/               # aggregate scoring CSVs (main results)
│   ├── supplementary/        # Table 3/4 and Appendix C CSVs
│   └── figures/              # curated example renders
├── main.py  run_benchmark.py  llm_client.py  thor_wrapper.py
├── requirements.txt  LICENSE  .gitignore  README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

### Evaluate your own model on SafeSight-Bench (no regeneration)

1. Obtain the frozen scene set (released on acceptance) and place it under
   `data/scenes/`, `data/safe_scenes/`, and (optionally) `data/noisy/`,
   `data/safe_noisy/`. Paths are configured in `benchmark/config.py`
   (`PROJECT_ROOT`-relative; no absolute paths).
2. Wrap your model behind the same call signature used in
   `benchmark/evaluate/vlm_client.py` (input: serialized scene metadata and/or
   the egocentric PNG; output: the unified JSON schema in §2). An OpenAI-style
   HTTP endpoint can be used directly via `llm_client.py`.
3. Run evaluation, then score:
   ```bash
   python -m benchmark.evaluate.baseline_eval --model <your-model> --track text_and_image
   python -m benchmark.score.scorer_v2
   ```
   Outputs land in `data/scores/` with the same FAR / Recall / F1 / HA / PC
   columns used in the paper.

### Extend the framework (new hazards / rooms / noise)

Add templates in `benchmark/harvest/danger_recipes.py`, then regenerate scenes
and noise with `benchmark/harvest/` and `benchmark/noise/`. See `scripts/`.

---

## Notes

- The framework is **simulation-only**; labels operationalize household risks
  through AI2-THOR states, affordances, and action outcomes.
- No secrets are stored in code. The optional OpenAI-backed juror in
  `cross_verify/` reads its key **only** from the `OPENAI_API_KEY` environment
  variable.
