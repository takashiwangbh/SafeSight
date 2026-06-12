# `cross_verify/` — Machine-Consensus Cross-Verification

Purpose: audit the simulator-derived hazard labels with an **independent
large-model jury**, producing agreement statistics that support **Section 7** of
the paper. This is an *auxiliary validation* protocol, not a replacement for
human annotation; results are described as *"agreement with independent
large-model judgments,"* never as *"objectively verified."*

## Structure

```
cross_verify/
├── config.py          # paths, juror registry, sampling params, generation kwargs
├── sample_scenes.py   # Stage 1: stratified sampling
├── jury_prompt.py     # neutral judgment prompt (omits any "when in doubt, flag" bias)
├── jury_eval.py       # Stage 2: load each juror, evaluate sampled scenes
├── consensus.py       # Stage 3: agreement / Cohen's kappa / keyword overlap
└── run_pipeline.sh    # three-stage runner
```

The pipeline reuses model loading, VRAM preflight, hazard-keyword alignment, and
JSON parsing from `benchmark/` and `llm_client.py`.

## Verifier panel

`config.py` defines the panel. Each entry carries a `provider` field
(`huggingface` or `openai`); the evaluator dispatches on it. The released setup
uses **three independent verifiers** spanning two access regimes and three model
families:

| Verifier | Family | Access |
| --- | --- | --- |
| Qwen2.5-72B-Instruct | Qwen | open-weight, local (4-bit) |
| Llama-3.1-70B-Instruct | Llama | open-weight, local (4-bit) |
| GPT-5.5 | GPT | proprietary, API |

HuggingFace verifiers are loaded one at a time, so a single 48 GiB GPU suffices;
the API verifier uses no local GPU and is read only from the `OPENAI_API_KEY`
environment variable (the pipeline skips it when the key is unset). Spanning
open-weight and proprietary models from different providers ensures the
agreement signal is not specific to any one model family.

## Released outputs

The aggregate results that back the tables below are included under
`data/cross_verify/consensus/`:

| File | Content |
| --- | --- |
| `summary.json` | headline agreement statistics (full sample and observable subset) |
| `per_scene.csv` | one row per scene: simulator label, each verifier's verdict, keyword-overlap flags, observability flag |
| `by_segment.csv` | agreement broken down by severity / room / recipe |
| `by_observability.csv` | per-recipe metadata-observability rate |

Per-verifier raw response files and the scene sampling manifest are produced by
the pipeline at run time; only the aggregate tables are distributed here.

## Reporting

Each verifier is scored independently against the simulator label (accuracy and
Cohen's κ). Inter-verifier agreement is reported pairwise. Derived
consensus columns (`dual`, `majority`, `unanimous`) are also provided in
`per_scene.csv` for completeness, but the primary signal is the per-verifier and
pairwise agreement, which makes no voting assumption.

## Results

All numbers below are computed from the files in
`data/cross_verify/consensus/` over a fixed stratified sample of **500 scenes**
(250 hazardous, 250 safe). The **sensor-observable subset** (N=456) is all
safe scenes plus the hazardous scenes whose metadata exposes at least one
abnormal state flag (see definition below). Agreement is reported as accuracy
and Cohen's κ.

**Table 1 — Agreement of each independent verifier with the simulator label.**

| Verifier | Observable Acc (N=456) | κ | Full Acc (N=500) | κ |
| --- | :---: | :---: | :---: | :---: |
| Qwen2.5-72B-Instruct | 0.908 | 0.811 | 0.828 | 0.656 |
| Llama-3.1-70B-Instruct | 0.871 | 0.732 | 0.794 | 0.588 |
| GPT-5.5 | 0.917 | 0.831 | 0.844 | 0.688 |

All three verifiers independently reproduce the simulator label on the
observable subset with substantial-to-strong agreement. The proprietary
verifier (different provider, no shared weights with the local pair) agrees most
strongly, indicating the labels are not an artifact of any single model family.

**Table 2 — Pairwise inter-verifier agreement.**

| Pair | Observable Acc | κ | Full Acc | κ |
| --- | :---: | :---: | :---: | :---: |
| Qwen2.5-72B ↔ Llama-3.1-70B | 0.954 | 0.899 | 0.958 | 0.903 |
| Qwen2.5-72B ↔ GPT-5.5 | 0.921 | 0.835 | 0.920 | 0.827 |
| Llama-3.1-70B ↔ GPT-5.5 | 0.897 | 0.781 | 0.898 | 0.775 |

**Table 3 — Agreement by template severity (two local verifiers' consensus, N=500).**

| Severity tier | Scenes | Agreement |
| --- | :---: | :---: |
| Critical hazard | 35 | 1.000 |
| High hazard | 88 | 0.739 |
| Medium hazard | 84 | 0.545 |
| Low hazard | 43 | 0.279 |
| Safe baseline | 250 | 1.000 |

Agreement is highest for critical hazards and safe baselines and decreases for
lower-severity templates, which describe genuinely borderline situations (e.g.
an unattended lamp). Among verifier-confirmed hazardous scenes, **82.5%** mention
a keyword aligned with the template's danger label, indicating the agreement is
grounded in the named hazard rather than incidental. The three-model majority
column is available in `by_segment.csv` for readers who prefer a panel vote.

The observable-vs-full gap (e.g. GPT-5.5 0.917 → 0.844) is itself a finding: a
minority of recipes render a hazard visually but do not propagate it into the
object metadata, so the verifiers — like the audited models — read "clean"
sensor data. `by_observability.csv` lists the per-recipe observability rate.

## Sensor-observability

A hazardous scene is **sensor-observable** iff at least one object in its
`visible_objects` has one of these flags set to `True`:

```
isToggled | isOpen | isBroken | isSliced | isFilledWithLiquid
```

Safe scenes are always observable (their safety is the observable fact that no
abnormal flag is set). Some recipes render the hazard visually but do not
propagate it into the object metadata, so both the audited models and the jurors
see "clean" sensor data. Reporting the observable subset separately (1) honestly
measures juror–simulator agreement when the hazard is exposed to the sensor
channel, and (2) surfaces which recipes fail to propagate state — itself an
audit finding. Both the subset and full-sample figures are disclosed, along with
the observability definition.

## Wording guidance (Section 7)

- Prefer *"agreement under this auxiliary validation protocol"* over
  *"establishes objective safety."*
- The protocol is **not** fully independent (it still relies on scene metadata
  and keyword rules); it is described as *"not used by the audited models and
  run outside the main scoring loop."*
- Hazard-keyword overlap is *"keyword-level overlap with template danger
  labels,"* not a *"semantic understanding score."*
- The conclusion is *"supporting evidence that many template labels are
  recoverable by independent large-model judgments,"* not *"confirmed correct."*
