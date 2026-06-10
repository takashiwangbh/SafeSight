"""Select the local mitigation evaluation set from EXISTING results (offline).

For each mitigation model (text+image track) we sample two disjoint pools
from the clean baseline so the second-pass study can show BOTH effects:

  * FP pool  — false alarms on clean SAFE scenes (pred=dangerous, gt=safe).
               These are what the mitigation should overturn → drives FAR↓.
  * TP pool  — correct detections on clean HAZARD scenes (pred=dangerous,
               gt=hazard).  These are the control: a good defence must keep
               them dangerous → proves recall is retained.

Sampling is stratified by room_type and deterministic (fixed seed) so the
manifest is reproducible.  No GPU, no torch — pure stdlib.

Output: data/supplementary/mitigation/sample_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from supplementary_experiments.common import (  # noqa: E402
    MITIGATION_MODELS, MITIGATION_TRACK, MITIGATION_DIR, MODEL_HF_ID,
    SCENES_DIR, SAFE_SCENES_DIR, ensure_supp_dirs,
    classify_confusion, parsed_of, confidence_of, iter_results,
)

MANIFEST_PATH = os.path.join(MITIGATION_DIR, "sample_manifest.json")


def _room_of(result: dict) -> str:
    return result.get("room_type") or "unknown"


def _stratified(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Even split across room_type, deterministic."""
    if n <= 0 or len(pool) <= n:
        return pool
    by_room: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_room[r["room_type"]].append(r)
    rng = random.Random(seed)
    for v in by_room.values():
        rng.shuffle(v)
    rooms = sorted(by_room)
    picked: list[dict] = []
    i = 0
    while len(picked) < n:
        advanced = False
        for room in rooms:
            bucket = by_room[room]
            if i < len(bucket):
                picked.append(bucket[i])
                advanced = True
                if len(picked) >= n:
                    break
        if not advanced:
            break
        i += 1
    return picked


def _collect(model: str, scenario: str, want_confusion: str,
             scene_dir: str) -> list[dict]:
    out: list[dict] = []
    for rpath, result in iter_results(scenario, model, MITIGATION_TRACK):
        parsed = parsed_of(result)
        if classify_confusion(result, parsed) != want_confusion:
            continue
        basename = os.path.basename(rpath).replace("_result.json", "")
        gt_path = os.path.join(scene_dir, f"{basename}_gt.json")
        png_path = os.path.join(scene_dir, f"{basename}.png")
        if not os.path.exists(gt_path):
            continue
        out.append({
            "model":            model,
            "hf_id":            MODEL_HF_ID[model],
            "track":            MITIGATION_TRACK,
            "scenario":         scenario,
            "basename":         basename,
            "room_type":        _room_of(result),
            "confusion":        want_confusion,
            "gt_is_safe":       scenario.endswith("safe"),
            "result_path":      rpath,
            "scene_gt_path":    gt_path,
            "image_path":       png_path if os.path.exists(png_path) else None,
            "first_pass_conf":  confidence_of(parsed),
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-fp", type=int, default=100,
                   help="false-alarm (safe) samples per model")
    p.add_argument("--n-tp", type=int, default=50,
                   help="true-detection (hazard) control samples per model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=MITIGATION_MODELS)
    args = p.parse_args()

    ensure_supp_dirs()

    manifest: list[dict] = []
    print("Selecting mitigation samples (track=%s)" % MITIGATION_TRACK)
    for model in args.models:
        fp_all = _collect(model, "baseline_safe", "FP", SAFE_SCENES_DIR)
        tp_all = _collect(model, "baseline",      "TP", SCENES_DIR)
        fp = _stratified(fp_all, args.n_fp, args.seed)
        tp = _stratified(tp_all, args.n_tp, args.seed)
        manifest.extend(fp)
        manifest.extend(tp)
        print(f"  {model:22s} FP {len(fp)}/{len(fp_all)} avail  "
              f"TP {len(tp)}/{len(tp_all)} avail")

    payload = {
        "track":   MITIGATION_TRACK,
        "n_fp":    args.n_fp,
        "n_tp":    args.n_tp,
        "seed":    args.seed,
        "models":  args.models,
        "samples": manifest,
        "n_total": len(manifest),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {MANIFEST_PATH}  ({len(manifest)} samples)")


if __name__ == "__main__":
    main()
