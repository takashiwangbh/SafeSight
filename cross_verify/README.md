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

## Juror registry

`config.py` defines the jury. Each entry carries a `provider` field
(`huggingface` or `openai`); the evaluator dispatches on it. The default setup
uses two 70B-class instruct models from different families plus an optional
API-served reasoning model. HuggingFace jurors are loaded one at a time; an
optional API juror uses no local GPU. The key for the API juror is read only
from the `OPENAI_API_KEY` environment variable, and the pipeline simply skips
that juror when the key is unset.

## Outputs

Results are written under `data/cross_verify/`: a sampling manifest, per-juror
per-scene results, a per-scene consensus table (with a `sensor_observable`
flag), headline numbers for Section 7, and breakdowns by severity / room /
recipe and by per-recipe observability rate.

## Consensus modes

- **2 jurors:** unanimous equals majority (reported as *"dual-model consensus"*).
- **3 jurors:** majority means ≥2/3 agree; unanimous means 3/3 agree. The
  majority figure is the headline, with the unanimous figure disclosed alongside.

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
