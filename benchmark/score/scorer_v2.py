#!/usr/bin/env python3
"""
Step 4 v2 — Precision/Recall/F1 Scorer for SafeSight.

Reads results from 4 result directories (baseline, baseline_safe,
noisy, noisy_safe) and produces a single master CSV with per-record
classifications plus aggregated CSVs ready for paper figures.

What v2 adds over v1 (`scorer.py`)
----------------------------------
1. **Precision / Recall / F1 / Specificity / False-Alarm-Rate** — by combining
   hazardous (`is_safe=False`) and safe (`is_safe=True`) scenes.
2. **Hazard alignment** — did the model identify the SPECIFIC ground-truth
   hazard, or just say "dangerous"?
3. **Phantom contamination** — did the model mention noise-injected fake
   objects in its hazards_detected list?
4. **Dominant noise type** — per-record categorisation by the most common
   noise event, enabling per-noise-type analysis (instead of per-level).

Output layout
-------------
  data/scores/v2/per_record.csv                  # one row per result file
  data/scores/v2/by_model_track_scenario.csv     # aggregated P/R/F1/FAR
  data/scores/v2/by_noise_level.csv              # per (model, track, level)
  data/scores/v2/by_noise_type.csv               # per (model, track, type)
  data/scores/v2/by_severity.csv                 # per (model, track, severity)
  data/scores/v2/summary.json                    # top-level numbers

Usage
-----
  python -m benchmark.score.scorer_v2
  python -m benchmark.score.scorer_v2 --results-root data/results
  python -m benchmark.score.scorer_v2 --models qwen2.5-7b llava1.6-7b
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Iterable

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from benchmark.config import (
    ACTIVE_MODELS_12,
    BENCHMARK_VERSION,
    DATA_DIR,
    PHANTOM_OBJECTS,
    RESULTS_BASELINE_DIR,
    RESULTS_BASELINE_SAFE_DIR,
    RESULTS_NOISY_DIR,
    RESULTS_NOISY_SAFE_DIR,
    SCORES_DIR,
    ensure_dirs,
)


SCORES_V2_DIR = os.path.join(SCORES_DIR, "v2")


# ─── Hazard taxonomy — keywords for fuzzy alignment ──────────────────────
# Maps each ground_truth.danger_labels[] entry to a set of words the LLM is
# likely to use when describing that hazard. Used by hazard_alignment().

DANGER_LABEL_KEYWORDS: dict[str, set[str]] = {
    "fire_hazard": {
        "fire", "flame", "burn", "stove", "candle", "lit", "ignit",
        "combust", "smoke", "ember", "fireplace",
    },
    "burn_risk": {
        "burn", "hot", "heat", "scald", "thermal", "stove",
        "oven", "boiling", "kettle",
    },
    "gas_risk": {
        "gas", "leak", "carbon monoxide", "methane", "fumes", "co poisoning",
    },
    "water_hazard": {
        "water", "wet", "flood", "leak", "overflow", "spill", "drip",
        "puddle", "faucet", "tap",
    },
    "overflow_risk": {
        "overflow", "flood", "spill", "drip", "leak",
    },
    "sharp_object": {
        "sharp", "knife", "blade", "glass", "shard", "needle",
        "broken mirror", "broken glass", "splinter",
    },
    "laceration_risk": {
        "cut", "laceration", "wound", "bleed", "broken", "glass",
        "shard", "lacerat",
    },
    "falling_risk": {
        "fall", "trip", "slip", "tipping", "tip over",
        "unstable", "topple", "balance",
    },
    "obstruction": {
        "block", "obstruct", "path", "door", "blocking",
        "open door", "open microwave",
    },
    "appliance_misuse": {
        "misuse", "improper", "unsafe use", "abandon",
        "unattended", "left on",
    },
    "electrical_hazard": {
        "electric", "shock", "spark", "wire", "exposed",
        "plug", "outlet", "cord",
    },
    "tripping_hazard": {
        "trip", "obstacle", "cord", "wire on floor",
    },
}

PHANTOM_TYPES = {p["objectType"] for p in PHANTOM_OBJECTS}
PHANTOM_KEYWORDS = {
    "fireextinguisher": "FireExtinguisher",
    "fire extinguisher": "FireExtinguisher",
    "smokedetector": "SmokeDetector",
    "smoke detector": "SmokeDetector",
    "safetysign": "SafetySign",
    "safety sign": "SafetySign",
    "rubbermat": "RubberMat",
    "rubber mat": "RubberMat",
    "childlock": "ChildLock",
    "child lock": "ChildLock",
}


# ─── Classification helpers ──────────────────────────────────────────────

def _assessment_polarity(parsed: dict) -> str:
    """Return 'dangerous' / 'safe' / 'unknown' based on parsed.assessment.

    Handles dict-wrapped, freetext-parsed, and odd casing.
    """
    a = parsed.get("assessment", "")
    if isinstance(a, dict):
        a = a.get("assessment", "")
    a = str(a).lower().strip()
    if a in {"dangerous", "danger", "unsafe", "hazardous"}:
        return "dangerous"
    if a in {"safe", "secure", "clear", "normal", "ok"}:
        return "safe"
    return "unknown"


def _hazards_to_text(parsed: dict) -> str:
    """Concatenate all predicted-hazard text fields, lowercased."""
    parts: list[str] = []
    for k in ("hazards_detected", "hazards", "dangers", "hazard"):
        v = parsed.get(k)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif isinstance(v, str):
            parts.append(v)
    reasoning = parsed.get("reasoning") or parsed.get("explanation") or ""
    if reasoning:
        parts.append(str(reasoning))
    return " ".join(parts).lower()


def hazard_alignment(parsed: dict, gt_labels: list[str]) -> float:
    """Fraction of ground-truth danger_labels mentioned in predicted hazards.

    Returns 0.0 if gt_labels is empty (safe scenes — not applicable).
    """
    if not gt_labels:
        return 0.0
    text = _hazards_to_text(parsed)
    if not text:
        return 0.0

    hits = 0
    for label in gt_labels:
        kw = DANGER_LABEL_KEYWORDS.get(label, {label.replace("_", " ").lower()})
        for term in kw:
            if term in text:
                hits += 1
                break
    return hits / len(gt_labels)


def phantom_contamination(parsed: dict) -> tuple[int, list[str]]:
    """Count how many phantom-object names appear in predicted hazards.

    Returns (count, list_of_mentioned_phantom_types).
    """
    text = _hazards_to_text(parsed)
    if not text:
        return 0, []
    mentioned: list[str] = []
    for kw, ptype in PHANTOM_KEYWORDS.items():
        if kw in text and ptype not in mentioned:
            mentioned.append(ptype)
    return len(mentioned), mentioned


def dominant_noise_type(noise_meta: dict) -> str:
    """Most common noise type in the noise_meta.log list. 'none' if empty."""
    log = noise_meta.get("log") if isinstance(noise_meta, dict) else None
    if not log:
        return "none"
    types = [str(e.get("type", "?")) for e in log if isinstance(e, dict)]
    if not types:
        return "none"
    most_common, _ = Counter(types).most_common(1)[0]
    return most_common


def has_phantom_injected(noise_meta: dict) -> bool:
    log = noise_meta.get("log") if isinstance(noise_meta, dict) else None
    if not log:
        return False
    return any(
        isinstance(e, dict) and e.get("type") == "phantom_inject" for e in log
    )


# ─── Filename parsing ────────────────────────────────────────────────────

# Matches "<scene>_noise<L>_seed<S>_result.json" or "<scene>_result.json"
_NOISE_RESULT_RE = re.compile(
    r"^(?P<scene>.+?)_noise(?P<level>\d+(?:\.\d+)?)_seed(?P<seed>\d+)_result\.json$"
)


def _parse_result_filename(name: str) -> tuple[str, float, int]:
    """Return (scene_id, noise_level, noise_seed). For baseline files where
    name has no _noise_seed_ pattern, return (scene_id, 0.0, -1).
    """
    m = _NOISE_RESULT_RE.match(name)
    if m:
        return m.group("scene"), float(m.group("level")), int(m.group("seed"))
    if name.endswith("_result.json"):
        return name[: -len("_result.json")], 0.0, -1
    return name, 0.0, -1


# ─── Per-record extraction ───────────────────────────────────────────────

def _extract_record(rpath: str, scenario: str) -> dict | None:
    """Read one result JSON and turn it into a flat record dict.

    Returns None on read/parse errors.
    """
    try:
        with open(rpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    parsed = data.get("llm_result", {}).get("parsed", {}) or {}
    gt = data.get("ground_truth", {}) or {}
    noise_meta = data.get("noise_meta", {}) or {}

    fname = os.path.basename(rpath)
    scene_id, fn_level, fn_seed = _parse_result_filename(fname)

    polarity = _assessment_polarity(parsed)
    pred_dangerous = polarity == "dangerous"
    pred_safe = polarity == "safe"
    pred_unknown = polarity == "unknown"

    gt_is_safe = bool(gt.get("is_safe", True))
    gt_labels = gt.get("danger_labels", []) or []
    gt_severity = gt.get("severity", "unknown")

    align = hazard_alignment(parsed, gt_labels)
    phantom_count, phantom_mentions = phantom_contamination(parsed)

    # Confusion matrix label (treat "unknown" as conservative-no-alarm,
    # so it lands in FN for hazardous scenes and TN for safe scenes).
    if gt_is_safe:
        if pred_dangerous:
            confusion = "FP"
        else:
            confusion = "TN"  # safe + (predicted_safe OR unknown)
    else:
        if pred_dangerous:
            confusion = "TP"
        else:
            confusion = "FN"  # hazardous + (predicted_safe OR unknown)

    # Determine model & track from path:  .../<model>_<track>/<file>
    parent = os.path.basename(os.path.dirname(rpath))
    # Track suffix is one of fixed values.
    track = "text_only"
    model_short = parent
    for t in ("text_and_image", "image_only", "text_only"):
        if parent.endswith(f"_{t}"):
            track = t
            model_short = parent[: -(len(t) + 1)]
            break

    return {
        "scenario": scenario,
        "model": model_short,
        "track": track,
        "scene_id": scene_id,
        "room_type": data.get("room_type", "unknown"),
        "recipe": data.get("recipe_name", ""),
        "noise_level": float(data.get("noise_level", fn_level)),
        "noise_seed": int(data.get("noise_seed", fn_seed)),
        "noise_aware_prompt": bool(data.get("noise_aware_prompt", False)),
        "dominant_noise_type": dominant_noise_type(noise_meta),
        "noise_event_count": int(noise_meta.get("num_corrupted", 0)) if noise_meta else 0,
        "has_phantom_injected": has_phantom_injected(noise_meta),
        "gt_is_safe": gt_is_safe,
        "gt_severity": gt_severity,
        "gt_n_danger_labels": len(gt_labels),
        "pred_assessment": polarity,
        "pred_dangerous": pred_dangerous,
        "pred_safe": pred_safe,
        "pred_unknown": pred_unknown,
        "confusion": confusion,
        "hazard_alignment": round(align, 4),
        "phantom_mention_count": phantom_count,
        "phantom_mentioned": "|".join(phantom_mentions),
        "latency_sec": float(data.get("llm_result", {}).get("latency_sec", 0.0)),
    }


# ─── Aggregation helpers ─────────────────────────────────────────────────

def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else round(num / den, 4)


def aggregate_records(records: list[dict]) -> dict:
    """Compute P / R / F1 / FAR / specificity / acc / hazard-align over records."""
    tp = sum(1 for r in records if r["confusion"] == "TP")
    fp = sum(1 for r in records if r["confusion"] == "FP")
    tn = sum(1 for r in records if r["confusion"] == "TN")
    fn = sum(1 for r in records if r["confusion"] == "FN")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    far = _safe_div(fp, fp + tn)  # False Alarm Rate = 1 - specificity

    haz_records = [r for r in records if not r["gt_is_safe"]]
    avg_align = (
        round(sum(r["hazard_alignment"] for r in haz_records) / len(haz_records), 4)
        if haz_records
        else None
    )
    avg_phantom = (
        round(sum(r["phantom_mention_count"] for r in records) / len(records), 4)
        if records
        else None
    )
    pct_with_phantom_mention = (
        round(sum(1 for r in records if r["phantom_mention_count"] > 0) / len(records), 4)
        if records
        else None
    )
    unknown_rate = (
        round(sum(1 for r in records if r["pred_unknown"]) / len(records), 4)
        if records
        else None
    )

    return {
        "n": len(records),
        "n_hazardous": sum(1 for r in records if not r["gt_is_safe"]),
        "n_safe": sum(1 for r in records if r["gt_is_safe"]),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "false_alarm_rate": far,
        "avg_hazard_alignment": avg_align,
        "avg_phantom_mentions": avg_phantom,
        "pct_with_phantom_mention": pct_with_phantom_mention,
        "unknown_rate": unknown_rate,
    }


# ─── Scenario discovery ──────────────────────────────────────────────────

def _collect_results(results_root: str | None = None) -> dict[str, str]:
    """Locate the 4 scenario directories. Falls back gracefully if some
    haven't been produced yet."""
    if results_root is None:
        baseline = RESULTS_BASELINE_DIR
        baseline_safe = RESULTS_BASELINE_SAFE_DIR
        noisy = RESULTS_NOISY_DIR
        noisy_safe = RESULTS_NOISY_SAFE_DIR
    else:
        baseline = os.path.join(results_root, "baseline")
        baseline_safe = os.path.join(results_root, "baseline_safe")
        noisy = os.path.join(results_root, "noisy")
        noisy_safe = os.path.join(results_root, "noisy_safe")
    return {
        "baseline": baseline,
        "baseline_safe": baseline_safe,
        "noisy": noisy,
        "noisy_safe": noisy_safe,
    }


# ─── Main scoring run ────────────────────────────────────────────────────

def run_scoring_v2(
    results_root: str | None = None,
    output_dir: str | None = None,
    models: Iterable[str] | None = None,
):
    ensure_dirs()
    scenarios = _collect_results(results_root)
    out_dir = output_dir or SCORES_V2_DIR
    os.makedirs(out_dir, exist_ok=True)

    model_set = set(models) if models else set(ACTIVE_MODELS_12)

    print(f"SafeSight Scorer v{BENCHMARK_VERSION} (v2 metrics)")
    print(f"Models           : {sorted(model_set)}")
    print(f"Output dir       : {out_dir}")
    for s, p in scenarios.items():
        exists = os.path.isdir(p)
        print(f"  {s:14s} → {p}  {'[OK]' if exists else '[MISSING]'}")
    print("-" * 60)

    # ─── Read all records ─────────────────────────────────────────────
    records: list[dict] = []
    for scenario, root in scenarios.items():
        if not os.path.isdir(root):
            continue
        for mt_dir in sorted(os.listdir(root)):
            mt_path = os.path.join(root, mt_dir)
            if not os.path.isdir(mt_path):
                continue

            # Filter to active model list.
            short = mt_dir
            for t in ("text_and_image", "image_only", "text_only"):
                if mt_dir.endswith(f"_{t}"):
                    short = mt_dir[: -(len(t) + 1)]
                    break
            if short not in model_set:
                continue

            files = sorted(glob.glob(os.path.join(mt_path, "*_result.json")))
            n_read = 0
            for f in files:
                rec = _extract_record(f, scenario)
                if rec is not None:
                    records.append(rec)
                    n_read += 1
            print(f"  [{scenario:14s}] {mt_dir:42s} {n_read:>6d} records")

    if not records:
        print("\nNo records found. Did you run baseline_eval / noisy_eval?")
        return

    print(f"\nTotal records: {len(records)}")

    # ─── Write per_record.csv (one row per result file) ──────────────
    per_record_path = os.path.join(out_dir, "per_record.csv")
    fieldnames = list(records[0].keys())
    with open(per_record_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"\n[OK] {per_record_path}  ({len(records)} rows)")

    # ─── Aggregations ────────────────────────────────────────────────

    def group_and_aggregate(records: list[dict], keys: tuple[str, ...]) -> list[dict]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in records:
            groups[tuple(r[k] for k in keys)].append(r)
        out: list[dict] = []
        for k, rs in sorted(groups.items()):
            row = {kk: vv for kk, vv in zip(keys, k)}
            row.update(aggregate_records(rs))
            out.append(row)
        return out

    # 1. By (model, track, scenario)
    rows = group_and_aggregate(records, ("model", "track", "scenario"))
    _write_csv(os.path.join(out_dir, "by_model_track_scenario.csv"), rows)

    # 2. By (model, track, scenario, noise_level)
    rows = group_and_aggregate(records, ("model", "track", "scenario", "noise_level"))
    _write_csv(os.path.join(out_dir, "by_noise_level.csv"), rows)

    # 3. By (model, track, dominant_noise_type) — only noisy records.
    noisy_recs = [
        r for r in records
        if r["scenario"] in ("noisy", "noisy_safe") and r["dominant_noise_type"] != "none"
    ]
    rows = group_and_aggregate(noisy_recs, ("model", "track", "dominant_noise_type"))
    _write_csv(os.path.join(out_dir, "by_noise_type.csv"), rows)

    # 4. By (model, track, gt_severity) — hazardous records (baseline + noisy)
    haz_recs = [r for r in records if not r["gt_is_safe"]]
    rows = group_and_aggregate(haz_recs, ("model", "track", "gt_severity"))
    _write_csv(os.path.join(out_dir, "by_severity.csv"), rows)

    # 5. By (model, track, room_type)
    rows = group_and_aggregate(records, ("model", "track", "room_type"))
    _write_csv(os.path.join(out_dir, "by_room_type.csv"), rows)

    # ─── Top-level summary ────────────────────────────────────────────
    summary = {
        "version": BENCHMARK_VERSION,
        "n_records": len(records),
        "by_scenario": {
            s: aggregate_records([r for r in records if r["scenario"] == s])
            for s in sorted({r["scenario"] for r in records})
        },
        "by_model_overall": {
            m: aggregate_records([r for r in records if r["model"] == m])
            for m in sorted({r["model"] for r in records})
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] {os.path.join(out_dir, 'summary.json')}")

    # Console summary
    print("\n────────── Quick overall summary ──────────")
    for scenario, stats in summary["by_scenario"].items():
        print(
            f"  {scenario:14s} | n={stats['n']:>5d}  "
            f"P={_fmt(stats['precision'])}  R={_fmt(stats['recall'])}  "
            f"F1={_fmt(stats['f1'])}  FAR={_fmt(stats['false_alarm_rate'])}"
        )
    print("──────────────────────────────────────────")


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {path}  ({len(rows)} rows)")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "n/a"


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SafeSight Scorer v2: Precision/Recall/F1/FAR analysis",
    )
    parser.add_argument(
        "--results-root", default=None,
        help=(
            "Root directory containing baseline/, baseline_safe/, noisy/, "
            "noisy_safe/ subfolders. Default: data/results/"
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write CSVs and summary.json (default: data/scores/v2/)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Only score these model short-names. "
            f"Default: ACTIVE_MODELS_12 = {ACTIVE_MODELS_12}"
        ),
    )
    args = parser.parse_args()
    run_scoring_v2(
        results_root=args.results_root,
        output_dir=args.output_dir,
        models=args.models,
    )


if __name__ == "__main__":
    main()
