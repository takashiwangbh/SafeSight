"""Select hazard scenes for the RGB-corruption study (offline, no GPU).

We want scenes where the image-track models ALREADY succeed on the clean
render, so any drop after corruption is attributable to the corruption and
not to a pre-existing perception failure.

Default policy: take the INTERSECTION of clean-baseline true positives
(pred=dangerous on the real hazard) across all visual-noise models, on the
image_only track.  This guarantees every selected scene starts at recall=1
for every model, making the degradation curves directly comparable.

Sampling is stratified by severity then room_type, deterministic.

Output: data/supplementary/visual_noise/sample_manifest.json
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
    VISUAL_NOISE_MODELS, VISUAL_NOISE_TRACK, VISUAL_NOISE_DIR, MODEL_HF_ID,
    SCENES_DIR, ensure_supp_dirs,
    classify_confusion, parsed_of, iter_results, load_result,
)

MANIFEST_PATH = os.path.join(VISUAL_NOISE_DIR, "sample_manifest.json")


def _tp_basenames(model: str) -> dict[str, dict]:
    """basename → {room_type, severity} for clean-baseline TPs of one model."""
    out: dict[str, dict] = {}
    for rpath, result in iter_results("baseline", model, VISUAL_NOISE_TRACK):
        if classify_confusion(result, parsed_of(result)) != "TP":
            continue
        basename = os.path.basename(rpath).replace("_result.json", "")
        png = os.path.join(SCENES_DIR, f"{basename}.png")
        gt = os.path.join(SCENES_DIR, f"{basename}_gt.json")
        if not (os.path.exists(png) and os.path.exists(gt)):
            continue
        scene = load_result(gt) or {}
        out[basename] = {
            "room_type": result.get("room_type") or "unknown",
            "severity":  scene.get("ground_truth", {}).get("severity", "unknown"),
            "danger_labels": scene.get("ground_truth", {}).get("danger_labels", []),
            "gt_path":   gt,
            "png_path":  png,
        }
    return out


def _stratified(items: list[tuple[str, dict]], n: int, seed: int):
    if n <= 0 or len(items) <= n:
        return items
    buckets: dict[tuple, list] = defaultdict(list)
    for b, meta in items:
        buckets[(meta["severity"], meta["room_type"])].append((b, meta))
    rng = random.Random(seed)
    for v in buckets.values():
        rng.shuffle(v)
    keys = sorted(buckets, key=lambda k: (str(k[0]), str(k[1])))
    picked = []
    i = 0
    while len(picked) < n:
        advanced = False
        for k in keys:
            if i < len(buckets[k]):
                picked.append(buckets[k][i])
                advanced = True
                if len(picked) >= n:
                    break
        if not advanced:
            break
        i += 1
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=50, help="scenes to select")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=VISUAL_NOISE_MODELS)
    p.add_argument("--policy", choices=["intersection", "union"],
                   default="intersection")
    args = p.parse_args()

    ensure_supp_dirs()

    per_model = {m: _tp_basenames(m) for m in args.models}
    for m in args.models:
        print(f"  {m:22s} clean-baseline TP scenes: {len(per_model[m])}")

    sets = [set(d) for d in per_model.values()]
    if args.policy == "intersection":
        chosen = set.intersection(*sets) if sets else set()
    else:
        chosen = set.union(*sets) if sets else set()
    print(f"  {args.policy} pool: {len(chosen)} scenes")

    # attach metadata (prefer the first model that has it)
    meta_of: dict[str, dict] = {}
    for b in chosen:
        for m in args.models:
            if b in per_model[m]:
                meta_of[b] = per_model[m][b]
                break

    items = sorted(meta_of.items())
    picked = _stratified(items, args.n, args.seed)

    samples = []
    for b, meta in picked:
        samples.append({
            "basename":      b,
            "room_type":     meta["room_type"],
            "severity":      meta["severity"],
            "danger_labels": meta["danger_labels"],
            "gt_path":       meta["gt_path"],
            "png_path":      meta["png_path"],
        })

    payload = {
        "track":   VISUAL_NOISE_TRACK,
        "policy":  args.policy,
        "n":       args.n,
        "seed":    args.seed,
        "models":  args.models,
        "hf_ids":  {m: MODEL_HF_ID[m] for m in args.models},
        "samples": samples,
        "n_selected": len(samples),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {MANIFEST_PATH}  ({len(samples)} scenes)")


if __name__ == "__main__":
    main()
