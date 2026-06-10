# `cross_verify/` — Machine-Consensus Cross-Verification

Purpose: audit the simulator-derived hazard labels with an **independent
large-model jury**, producing agreement statistics that support **Section 7** of
the paper. This is an *auxiliary validation* protocol, not a replacement for
human annotation; results should be described as *"agreement with independent
large-model judgments,"* never as *"objectively verified."*

## Structure

```
cross_verify/
├── config.py          # paths, juror registry, sampling params, generation kwargs
├── sample_scenes.py   # Stage 1: stratified sampling -> sampled_scenes.json
├── jury_prompt.py     # neutral judgment prompt (omits any "when in doubt, flag" bias)
├── jury_eval.py       # Stage 2: load each juror, evaluate all sampled scenes
├── consensus.py       # Stage 3: agreement / Cohen's kappa / keyword overlap
└── run_pipeline.sh    # three-stage runner
```

Shared with `benchmark/`:

- `benchmark.evaluate.vlm_client` — model loading + greedy generation
- `benchmark.evaluate.gpu_utils.preflight_check` — VRAM preflight
- `benchmark.score.scorer_v2.DANGER_LABEL_KEYWORDS` — hazard-keyword alignment
- `llm_client.parse_llm_response` — strict-JSON + free-text fallback parsing

## Juror registry

`config.py` defines the jury. Each entry carries a `provider` field
(`huggingface` or `openai`); `jury_eval.py` dispatches on it.

| provider | example model | loading |
|----------|---------------|---------|
| huggingface | a 4-bit-quantized 70B-class instruct model | loaded once, then unloaded |
| huggingface | a second 70B-class instruct model from a different family | loaded once, then unloaded |
| openai | an API reasoning model (Responses API) | one API call per scene |

HuggingFace jurors are loaded one at a time (load → run → unload → next). An
optional API juror uses no local GPU and can run in parallel.

### API key

The code reads the key **only** from the environment — never hard-code it:

```bash
export OPENAI_API_KEY=...   # required only if an openai juror is configured
```

`run_pipeline.sh` warns and skips the API juror if the key is unset; it never
crashes the pipeline.

## Usage

```bash
# Smoke test (a few scenes)
bash cross_verify/run_pipeline.sh --smoke

# Full run: 500 scenes (250 hazardous + 250 safe)
bash cross_verify/run_pipeline.sh

# Re-sample with a different seed
bash cross_verify/run_pipeline.sh --seed=7
```

Stage-by-stage:

```bash
python -m cross_verify.sample_scenes --n-hazard 250 --n-safe 250 --seed 42
python -m cross_verify.jury_eval                 # all jurors (resumable; skips done scenes)
python -m cross_verify.consensus                 # agreement / kappa / overlap
```

Outputs (under `data/cross_verify/`):

```
sampled_scenes.json                     # Stage 1 manifest
jury_results/<juror>/<basename>.json    # Stage 2 per-juror, per-scene results
consensus/per_scene.csv                 # Stage 3 per-scene table (+ sensor_observable flag)
consensus/summary.json                  # Stage 3 headline numbers (Section 7)
consensus/by_segment.csv                # by severity / room / recipe
consensus/by_observability.csv          # per-recipe observability rate
```

## Consensus modes

- **2 jurors:** unanimous = majority (report as *"dual-model consensus"*).
- **3 jurors:** majority = ≥2/3 agree; unanimous = 3/3 agree. Report the
  majority figure as the headline and disclose the unanimous figure alongside.

## Sensor-observability

A hazardous scene is **sensor-observable** iff at least one object in its
`visible_objects` has one of these flags set to `True`:

```
isToggled | isOpen | isBroken | isSliced | isFilledWithLiquid
```

Safe scenes are always observable (their safety is the observable fact that no
abnormal flag is set). Some recipes render the hazard visually but do **not**
propagate it into the object metadata, so both the audited models and the jurors
see "clean" sensor data. Reporting the observable subset separately (1) honestly
measures juror–simulator agreement when the hazard is exposed to the sensor
channel, and (2) surfaces which recipes fail to propagate state — itself an
audit finding. Always disclose both the subset and full-sample figures plus the
observability definition.

## Wording guidance (Section 7)

- Prefer *"agreement under this auxiliary validation protocol"* over
  *"establishes objective safety."*
- The protocol is **not** fully independent (it still relies on scene metadata
  and keyword rules); describe it as *"not used by the audited models and run
  outside the main scoring loop."*
- Hazard-keyword overlap is *"keyword-level overlap with template danger
  labels,"* not a *"semantic understanding score."*
- Conclude with *"supporting evidence that many template labels are recoverable
  by independent large-model judgments,"* not *"confirmed correct."*
