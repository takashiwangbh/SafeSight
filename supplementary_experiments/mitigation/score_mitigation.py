"""Score the mitigation study → paper Table 4 (offline, no GPU).

Produces artefacts under data/supplementary/mitigation/:

1. table4_egav.csv  ★ HEADLINE / proposed method
   Evidence-Grounded Alarm Verification on the FULL clean populations
   (every baseline_safe + baseline text+image result), for both gate modes
   (``state`` and ``state_object``).  Columns: far_before/after,
   recall_before/after, f1_before/after, counts.
   → the positive result: FAR roughly halved while F1 improves.

2. table4_population_confidence_gate.csv  (weak baseline 1)
   Confidence gate on the FULL clean populations, swept over thresholds.
   → demonstrates (honestly) how little a calibration gate helps when the
     yes-man alarms are high-confidence.

3. table4_sample_pools.csv
   Apples-to-apples comparison on the sampled FP / TP pools for:
     · confidence gate      (weak baseline 1, default threshold)
     · self-correction      (weak baseline 2, from run_self_correction)
     · EGAV state / state_object (proposed)
   Metrics:
     · fp_suppression_rate = fraction of original false alarms now 'safe'
       (higher = FAR reduced more)
     · tp_retention_rate   = fraction of original true detections still
       'dangerous' (higher = recall preserved)

4. table4_summary.json — machine-readable roll-up.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from supplementary_experiments.common import (  # noqa: E402
    MITIGATION_DIR, MITIGATION_MODELS, MITIGATION_TRACK,
    SCENES_DIR, SAFE_SCENES_DIR, SCENARIO_DIRS,
    classify_confusion, parsed_of, iter_results, load_result,
    _assessment_polarity,
)
from supplementary_experiments.mitigation.mitigation_baseline import (  # noqa: E402
    confidence_gated_mitigation, egav_mitigation,
)

MANIFEST_PATH = os.path.join(MITIGATION_DIR, "sample_manifest.json")
SC_OUT_DIR = os.path.join(MITIGATION_DIR, "self_correction")

THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]
EGAV_MODES = ["state", "state_object"]


def _safe_div(a: int, b: int) -> float | None:
    return None if b == 0 else round(a / b, 4)


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denom_p = tp + fp
    denom_r = tp + fn
    if denom_p == 0 or denom_r == 0:
        return None
    p = tp / denom_p
    r = tp / denom_r
    return None if (p + r) == 0 else round(2 * p * r / (p + r), 4)


def _scene_gt_path(scenario: str, basename: str) -> str:
    d = SAFE_SCENES_DIR if scenario.endswith("safe") else SCENES_DIR
    return os.path.join(d, f"{basename}_gt.json")


def _result_path(model: str, track: str, scenario: str, basename: str) -> str:
    """Reconstruct a first-pass result path from config (portable across
    local / server, unlike absolute paths baked into the manifest)."""
    return os.path.join(SCENARIO_DIRS[scenario], f"{model}_{track}",
                        f"{basename}_result.json")


# ─── Strategy 1: full-population confidence-gate sweep ───────────────────

def population_sweep(models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        # Gather clean safe (for FAR) and clean hazard (for recall).
        safe = [(parsed_of(r)) for _p, r in
                iter_results("baseline_safe", model, MITIGATION_TRACK)]
        hazard = [(parsed_of(r)) for _p, r in
                  iter_results("baseline", model, MITIGATION_TRACK)]
        if not safe and not hazard:
            continue

        def is_danger(parsed):
            return _assessment_polarity(parsed) == "dangerous"

        n_safe = len(safe)
        n_haz = len(hazard)
        fp_before = sum(1 for p in safe if is_danger(p))
        tp_before = sum(1 for p in hazard if is_danger(p))
        far_before = _safe_div(fp_before, n_safe)
        # recall_before = tp / (tp + fn); fn = hazard not flagged
        recall_before = _safe_div(tp_before, n_haz)

        for th in THRESHOLDS:
            fp_after = 0
            for p in safe:
                v, _ = confidence_gated_mitigation(p, th)
                if is_danger(v):
                    fp_after += 1
            tp_after = 0
            for p in hazard:
                v, _ = confidence_gated_mitigation(p, th)
                if is_danger(v):
                    tp_after += 1
            rows.append({
                "model":          model,
                "track":          MITIGATION_TRACK,
                "threshold":      th,
                "n_safe":         n_safe,
                "n_hazard":       n_haz,
                "far_before":     far_before,
                "far_after":      _safe_div(fp_after, n_safe),
                "recall_before":  recall_before,
                "recall_after":   _safe_div(tp_after, n_haz),
                "fp_overturned":  fp_before - fp_after,
                "tp_lost":        tp_before - tp_after,
            })
    return rows


# ─── PROPOSED: EGAV full-population evaluation (headline Table 4) ─────────

def _collect_population(model: str, scenario: str):
    """Return [(parsed, scene_dict_or_None), ...] for a config."""
    out = []
    for path, res in iter_results(scenario, model, MITIGATION_TRACK):
        base = os.path.basename(path).replace("_result.json", "")
        scene = load_result(_scene_gt_path(scenario, base))
        out.append((parsed_of(res), scene))
    return out


def egav_population(models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        safe = _collect_population(model, "baseline_safe")
        hazard = _collect_population(model, "baseline")
        if not safe and not hazard:
            continue

        def is_danger(parsed):
            return _assessment_polarity(parsed) == "dangerous"

        n_safe = len(safe)
        n_haz = len(hazard)
        fp_before = sum(1 for p, _ in safe if is_danger(p))
        tp_before = sum(1 for p, _ in hazard if is_danger(p))
        far_before = _safe_div(fp_before, n_safe)
        recall_before = _safe_div(tp_before, n_haz)
        f1_before = _f1(tp_before, fp_before, n_haz - tp_before)

        for mode in EGAV_MODES:
            fp_after = 0
            for p, sc in safe:
                v, _ = egav_mitigation(p, sc, mode)
                if is_danger(v):
                    fp_after += 1
            tp_after = 0
            for p, sc in hazard:
                v, _ = egav_mitigation(p, sc, mode)
                if is_danger(v):
                    tp_after += 1
            rows.append({
                "model":         model,
                "track":         MITIGATION_TRACK,
                "egav_mode":     mode,
                "n_safe":        n_safe,
                "n_hazard":      n_haz,
                "far_before":    far_before,
                "far_after":     _safe_div(fp_after, n_safe),
                "recall_before": recall_before,
                "recall_after":  _safe_div(tp_after, n_haz),
                "f1_before":     f1_before,
                "f1_after":      _f1(tp_after, fp_after, n_haz - tp_after),
                "fp_overturned": fp_before - fp_after,
                "tp_lost":       tp_before - tp_after,
            })
    return rows


# ─── Sample-pool comparison (baselines vs EGAV) ──────────────────────────

def _load_sc_output(model: str, track: str, scenario: str, basename: str):
    path = os.path.join(SC_OUT_DIR, f"{model}_{track}",
                        f"{scenario}__{basename}.json")
    return load_result(path)


def sample_pools(manifest: dict, s1_threshold: float) -> list[dict]:
    samples = manifest["samples"]
    track = manifest.get("track", MITIGATION_TRACK)
    rows: list[dict] = []
    models = manifest["models"]

    def _pool_parsed(s):
        res = load_result(_result_path(s["model"], track, s["scenario"], s["basename"]))
        return parsed_of(res) if res else {}

    def _pool_scene(s):
        return load_result(_scene_gt_path(s["scenario"], s["basename"]))

    for model in models:
        fp = [s for s in samples if s["model"] == model and s["confusion"] == "FP"]
        tp = [s for s in samples if s["model"] == model and s["confusion"] == "TP"]

        # ---- Baseline 1 (confidence gate) on the pools ----
        s1_fp_suppressed = 0
        for s in fp:
            v, _ = confidence_gated_mitigation(_pool_parsed(s), s1_threshold)
            if _assessment_polarity(v) != "dangerous":
                s1_fp_suppressed += 1
        s1_tp_retained = 0
        for s in tp:
            v, _ = confidence_gated_mitigation(_pool_parsed(s), s1_threshold)
            if _assessment_polarity(v) == "dangerous":
                s1_tp_retained += 1

        rows.append({
            "model": model, "method": f"confidence_gate@{s1_threshold}",
            "n_fp": len(fp), "n_tp": len(tp),
            "fp_suppression_rate": _safe_div(s1_fp_suppressed, len(fp)),
            "tp_retention_rate":   _safe_div(s1_tp_retained, len(tp)),
        })

        # ---- Strategy 2 (self-correction) on the pools ----
        sc_missing = 0
        s2_fp_suppressed = s2_fp_seen = 0
        for s in fp:
            out = _load_sc_output(model, track, s["scenario"], s["basename"])
            if out is None:
                sc_missing += 1
                continue
            s2_fp_seen += 1
            if _assessment_polarity(out.get("second_pass_parsed", {})) != "dangerous":
                s2_fp_suppressed += 1
        s2_tp_retained = s2_tp_seen = 0
        for s in tp:
            out = _load_sc_output(model, track, s["scenario"], s["basename"])
            if out is None:
                sc_missing += 1
                continue
            s2_tp_seen += 1
            if _assessment_polarity(out.get("second_pass_parsed", {})) == "dangerous":
                s2_tp_retained += 1

        rows.append({
            "model": model, "method": "self_correction",
            "n_fp": s2_fp_seen, "n_tp": s2_tp_seen,
            "fp_suppression_rate": _safe_div(s2_fp_suppressed, s2_fp_seen),
            "tp_retention_rate":   _safe_div(s2_tp_retained, s2_tp_seen),
            "sc_outputs_missing":  sc_missing,
        })

        # ---- PROPOSED: EGAV on the pools (both modes) ----
        for mode in EGAV_MODES:
            egav_fp_suppressed = 0
            for s in fp:
                v, _ = egav_mitigation(_pool_parsed(s), _pool_scene(s), mode)
                if _assessment_polarity(v) != "dangerous":
                    egav_fp_suppressed += 1
            egav_tp_retained = 0
            for s in tp:
                v, _ = egav_mitigation(_pool_parsed(s), _pool_scene(s), mode)
                if _assessment_polarity(v) == "dangerous":
                    egav_tp_retained += 1
            rows.append({
                "model": model, "method": f"egav_{mode}",
                "n_fp": len(fp), "n_tp": len(tp),
                "fp_suppression_rate": _safe_div(egav_fp_suppressed, len(fp)),
                "tp_retention_rate":   _safe_div(egav_tp_retained, len(tp)),
            })
    return rows


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        print(f"  (no rows for {path})")
        return
    keys = list({k for r in rows for k in r})
    # stable column order: put common identifiers first
    head = [k for k in ("model", "track", "method", "egav_mode", "threshold") if k in keys]
    rest = sorted(k for k in keys if k not in head)
    fields = head + rest
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s1-threshold", type=float, default=0.60,
                   help="confidence gate threshold for the pool comparison")
    p.add_argument("--models", nargs="+", default=None)
    args = p.parse_args()

    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"Missing manifest: {MANIFEST_PATH} (run select_fp_samples first)")
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    models = args.models or manifest["models"] or MITIGATION_MODELS

    print("Scoring mitigation study ...")
    egav = egav_population(models)
    pop = population_sweep(models)
    pools = sample_pools(manifest, args.s1_threshold)

    _write_csv(os.path.join(MITIGATION_DIR, "table4_egav.csv"), egav)
    _write_csv(os.path.join(MITIGATION_DIR,
               "table4_population_confidence_gate.csv"), pop)
    _write_csv(os.path.join(MITIGATION_DIR,
               "table4_sample_pools.csv"), pools)

    summary = {
        "track": MITIGATION_TRACK,
        "s1_threshold": args.s1_threshold,
        "egav_population": egav,
        "population_confidence_gate": pop,
        "sample_pools": pools,
    }
    with open(os.path.join(MITIGATION_DIR, "table4_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  wrote {os.path.join(MITIGATION_DIR, 'table4_summary.json')}")

    # Console headline — the positive result.
    print("\n=== PROPOSED: EGAV on full population (Table 4) ===")
    for r in egav:
        print(f"  {r['model']:22s} mode={r['egav_mode']:13s} "
              f"FAR {r['far_before']}->{r['far_after']}  "
              f"Recall {r['recall_before']}->{r['recall_after']}  "
              f"F1 {r['f1_before']}->{r['f1_after']}")

    print("\n=== Sample-pool comparison (baselines vs EGAV) ===")
    for r in pools:
        print(f"  {r['model']:22s} {r['method']:22s} "
              f"FAR-suppress={r['fp_suppression_rate']}  "
              f"recall-retain={r['tp_retention_rate']}  "
              f"(n_fp={r['n_fp']}, n_tp={r['n_tp']})")


if __name__ == "__main__":
    main()
