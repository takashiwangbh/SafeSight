"""Score the RGB-corruption study → paper Appendix C (offline, no GPU).

All selected scenes are hazards, so on the image_only track:
    recall = (# predicted dangerous) / (# scenes)

We report, per model × condition:
    recall, and Δrecall vs the matched CLEAN control run.

Outputs under data/supplementary/visual_noise/:
    appendixC_recall_by_condition.csv
    appendixC_recall_by_severity.csv
    appendixC_summary.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from supplementary_experiments.common import (  # noqa: E402
    VISUAL_NOISE_DIR, load_result, _assessment_polarity,
)

RESULTS_DIR = os.path.join(VISUAL_NOISE_DIR, "results")


def _safe_div(a: int, b: int):
    return None if b == 0 else round(a / b, 4)


def _collect(models: list[str]):
    """records[model][condition] = list of (basename, severity, is_danger)."""
    records: dict = defaultdict(lambda: defaultdict(list))
    for model in models:
        mdir = os.path.join(RESULTS_DIR, model)
        if not os.path.isdir(mdir):
            continue
        for path in glob.glob(os.path.join(mdir, "*.json")):
            r = load_result(path)
            if r is None:
                continue
            cond = r.get("condition", "?")
            is_danger = _assessment_polarity(r.get("parsed", {})) == "dangerous"
            records[model][cond].append(
                (r["basename"], r.get("severity", "unknown"),
                 r.get("noise_type", "?"), r.get("intensity", 0.0), is_danger)
            )
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=None)
    args = p.parse_args()

    manifest_path = os.path.join(VISUAL_NOISE_DIR, "sample_manifest.json")
    models = args.models
    if models is None:
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                models = json.load(f)["models"]
        else:
            models = sorted(os.listdir(RESULTS_DIR)) if os.path.isdir(RESULTS_DIR) else []

    records = _collect(models)

    by_cond_rows: list[dict] = []
    by_sev_rows: list[dict] = []
    summary: dict = {}

    for model in models:
        conds = records.get(model, {})
        clean = conds.get("clean", [])
        clean_recall = _safe_div(sum(1 for *_x, d in clean if d), len(clean))
        summary.setdefault(model, {})["clean_recall"] = clean_recall

        for cond, rows in sorted(conds.items()):
            n = len(rows)
            n_danger = sum(1 for *_x, d in rows if d)
            recall = _safe_div(n_danger, n)
            noise_type = rows[0][2] if rows else "?"
            intensity = rows[0][3] if rows else 0.0
            delta = (None if (recall is None or clean_recall is None)
                     else round(recall - clean_recall, 4))
            by_cond_rows.append({
                "model": model, "condition": cond,
                "noise_type": noise_type, "intensity": intensity,
                "n": n, "recall": recall,
                "delta_vs_clean": delta,
            })

            # severity breakdown
            sev_buckets: dict[str, list] = defaultdict(list)
            for basename, sev, *_rest, d in rows:
                sev_buckets[sev].append(d)
            for sev, ds in sorted(sev_buckets.items()):
                by_sev_rows.append({
                    "model": model, "condition": cond, "severity": sev,
                    "n": len(ds),
                    "recall": _safe_div(sum(1 for d in ds if d), len(ds)),
                })

    def _write(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  wrote {path}  ({len(rows)} rows)")

    if by_cond_rows:
        _write(os.path.join(VISUAL_NOISE_DIR, "appendixC_recall_by_condition.csv"),
               by_cond_rows,
               ["model", "condition", "noise_type", "intensity",
                "n", "recall", "delta_vs_clean"])
    if by_sev_rows:
        _write(os.path.join(VISUAL_NOISE_DIR, "appendixC_recall_by_severity.csv"),
               by_sev_rows, ["model", "condition", "severity", "n", "recall"])

    with open(os.path.join(VISUAL_NOISE_DIR, "appendixC_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump({"by_condition": by_cond_rows, "clean_recall": summary},
                  f, indent=2, ensure_ascii=False)

    print("\n=== Appendix C headline (image_only recall) ===")
    for r in by_cond_rows:
        if r["condition"] == "clean":
            print(f"  {r['model']:22s} clean recall = {r['recall']}")
    for r in by_cond_rows:
        if r["condition"] != "clean":
            print(f"  {r['model']:22s} {r['condition']:22s} "
                  f"recall={r['recall']}  Δ={r['delta_vs_clean']}")


if __name__ == "__main__":
    main()
