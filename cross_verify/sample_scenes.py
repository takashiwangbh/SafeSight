"""Stratified scene sampling for the cross-verification study.

Reads every *_gt.json under data/scenes/ (hazardous) and data/safe_scenes/
(safe), then picks a balanced, deterministic subset:

  N_HAZARD scenes stratified over (severity, room_type)
  N_SAFE  scenes stratified over (room_type,)

The output is one JSON manifest at SAMPLED_SCENES_FILE so the rest of the
pipeline (jury_eval.py, consensus.py) reads from a single source of truth.

Usage:
    python -m cross_verify.sample_scenes
    python -m cross_verify.sample_scenes --n-hazard 300 --n-safe 200
    python -m cross_verify.sample_scenes --seed 7   # different sample
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from cross_verify.config import (
    HAZARD_STRATIFY_KEYS, N_HAZARD, N_SAFE,
    RANDOM_SEED, SAFE_SCENES_DIR, SAFE_STRATIFY_KEYS,
    SAMPLED_SCENES_FILE, SCENES_DIR, ensure_dirs,
)


def _load_gt_files(scenes_dir: str) -> list[dict]:
    """Return a list of minimal records describing each *_gt.json in dir."""
    records = []
    for fname in sorted(os.listdir(scenes_dir)):
        if not fname.endswith("_gt.json"):
            continue
        path = os.path.join(scenes_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [skip] {fname}: {e}")
            continue

        gt = data.get("ground_truth") or {}
        records.append({
            "basename":    fname.replace("_gt.json", ""),
            "gt_path":     path,
            "scene_name":  data.get("scene_name", ""),
            "recipe_name": data.get("recipe_name", ""),
            "room_type":   data.get("room_type", ""),
            "is_safe":     bool(gt.get("is_safe", False)),
            "severity":    gt.get("severity", "none"),
            "danger_labels": list(gt.get("danger_labels") or []),
        })
    return records


def _stratified_sample(records: list[dict], keys: tuple[str, ...],
                       n_target: int, rng: random.Random) -> list[dict]:
    """Sample approximately n_target items, balanced across the strata."""
    if not records:
        return []
    if n_target >= len(records):
        return list(records)

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = tuple(r.get(k, "") for k in keys)
        buckets[key].append(r)

    for key in buckets:
        rng.shuffle(buckets[key])

    out: list[dict] = []
    bucket_keys = sorted(buckets.keys())
    pointers = {k: 0 for k in bucket_keys}

    while len(out) < n_target:
        progressed = False
        for key in bucket_keys:
            if len(out) >= n_target:
                break
            idx = pointers[key]
            if idx < len(buckets[key]):
                out.append(buckets[key][idx])
                pointers[key] = idx + 1
                progressed = True
        if not progressed:
            break

    return out


def _summary(records: list[dict], keys: tuple[str, ...]) -> dict:
    cnt: Counter = Counter()
    for r in records:
        cnt[tuple(r.get(k, "") for k in keys)] += 1
    return {" / ".join(keys) + f"={k}": v
            for k, v in sorted(cnt.items())}


def main():
    p = argparse.ArgumentParser(description="Stratified scene sampling for cross-verification.")
    p.add_argument("--n-hazard", type=int, default=N_HAZARD)
    p.add_argument("--n-safe", type=int, default=N_SAFE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--out", default=SAMPLED_SCENES_FILE)
    args = p.parse_args()

    ensure_dirs()
    rng = random.Random(args.seed)

    print("=" * 60)
    print(f"Cross-Verify  Sampling Seed={args.seed}")
    print("=" * 60)

    print(f"Loading hazardous GT from {SCENES_DIR} ...")
    hazard_all = _load_gt_files(SCENES_DIR)
    hazard_all = [r for r in hazard_all if not r["is_safe"]]
    print(f"  -> {len(hazard_all)} hazardous scenes")

    print(f"Loading safe GT from {SAFE_SCENES_DIR} ...")
    safe_all = _load_gt_files(SAFE_SCENES_DIR)
    safe_all = [r for r in safe_all if r["is_safe"]]
    print(f"  -> {len(safe_all)} safe scenes")

    hazard_sample = _stratified_sample(
        hazard_all, HAZARD_STRATIFY_KEYS, args.n_hazard, rng,
    )
    safe_sample = _stratified_sample(
        safe_all, SAFE_STRATIFY_KEYS, args.n_safe, rng,
    )

    print(f"\nHazard sample:  requested={args.n_hazard}  obtained={len(hazard_sample)}")
    print(f"Safe   sample:  requested={args.n_safe}  obtained={len(safe_sample)}")

    print("\nHazard stratum breakdown:")
    for k, v in _summary(hazard_sample, HAZARD_STRATIFY_KEYS).items():
        print(f"  {k}: {v}")
    print("\nSafe stratum breakdown:")
    for k, v in _summary(safe_sample, SAFE_STRATIFY_KEYS).items():
        print(f"  {k}: {v}")

    manifest = {
        "seed":     args.seed,
        "n_hazard": len(hazard_sample),
        "n_safe":   len(safe_sample),
        "hazard_strat_keys": list(HAZARD_STRATIFY_KEYS),
        "safe_strat_keys":   list(SAFE_STRATIFY_KEYS),
        "hazard":   hazard_sample,
        "safe":     safe_sample,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
