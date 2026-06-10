"""
Step 2 — Noise Engine.

Injects controllable, reproducible noise into clean ground-truth JSONs.
This is the core "Method" of the paper.

Noise types:
  1. LabelSwap     — rename object types (Knife → Spoon)
  2. StateFlip     — flip boolean states (isToggled: true → false)
  3. InfoDrop      — delete objects entirely
  4. PositionJitter — add Gaussian noise to position coordinates
  5. DistanceWarp  — distort distance readings
  6. PhantomInject — add fake objects that don't exist
  7. PropertyCorrupt — change specific property values
  8. SensorBlackout — replace values with null / unknown

Usage:
    # Hazardous scenes (default):
    python -m benchmark.noise.noise_engine
    python -m benchmark.noise.noise_engine --noise-levels 0.1 0.3

    # Safe scenes (NEW — feed safe_scenes/ → safe_noisy/):
    python -m benchmark.noise.noise_engine \
        --input-dir data/safe_scenes --output-dir data/safe_noisy
"""

import argparse
import copy
import glob
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.config import (
    BENCHMARK_VERSION, LABEL_SWAP_TABLE, NOISE_LEVELS,
    NOISE_SEEDS_PER_LEVEL, NOISY_DIR, PHANTOM_OBJECTS,
    SCENES_DIR, ensure_dirs,
)


class NoiseType(str, Enum):
    LABEL_SWAP = "label_swap"
    STATE_FLIP = "state_flip"
    INFO_DROP = "info_drop"
    POSITION_JITTER = "position_jitter"
    DISTANCE_WARP = "distance_warp"
    PHANTOM_INJECT = "phantom_inject"
    PROPERTY_CORRUPT = "property_corrupt"
    SENSOR_BLACKOUT = "sensor_blackout"


@dataclass
class NoiseProfile:
    """
    Defines the probability distribution over noise types.
    Probabilities are relative weights (will be normalised internally).
    """
    weights: dict[str, float] = field(default_factory=lambda: {
        NoiseType.LABEL_SWAP: 2.0,
        NoiseType.STATE_FLIP: 3.0,
        NoiseType.INFO_DROP: 2.0,
        NoiseType.POSITION_JITTER: 1.5,
        NoiseType.DISTANCE_WARP: 1.5,
        NoiseType.PHANTOM_INJECT: 1.0,
        NoiseType.PROPERTY_CORRUPT: 1.0,
        NoiseType.SENSOR_BLACKOUT: 0.5,
    })

    def pick(self, rng: random.Random) -> NoiseType:
        types = list(self.weights.keys())
        weights = list(self.weights.values())
        return rng.choices(types, weights=weights, k=1)[0]


DEFAULT_PROFILE = NoiseProfile()

BOOL_STATE_FIELDS = [
    "isToggled", "isOpen", "isBroken", "isSliced",
    "isCooked", "isDirty", "isFilledWithLiquid", "visible",
]


def _apply_label_swap(obj: dict, rng: random.Random) -> dict | None:
    """Swap the object's type label with a plausible alternative."""
    original_type = obj["objectType"]
    candidates = LABEL_SWAP_TABLE.get(original_type)
    if not candidates:
        all_types = list(LABEL_SWAP_TABLE.keys())
        if original_type in all_types:
            candidates = LABEL_SWAP_TABLE[original_type]
        else:
            return None

    new_type = rng.choice(candidates)
    obj = copy.deepcopy(obj)
    obj["objectType"] = new_type
    obj["_noise"] = {"type": "label_swap", "original": original_type, "result": new_type}
    return obj


def _apply_state_flip(obj: dict, rng: random.Random) -> dict:
    """Flip one or more boolean state fields."""
    obj = copy.deepcopy(obj)
    flippable = [f for f in BOOL_STATE_FIELDS if f in obj and isinstance(obj[f], bool)]
    if not flippable:
        return obj

    field_to_flip = rng.choice(flippable)
    obj[field_to_flip] = not obj[field_to_flip]
    obj["_noise"] = {"type": "state_flip", "field": field_to_flip,
                     "original": not obj[field_to_flip]}
    return obj


def _apply_position_jitter(obj: dict, rng: random.Random, sigma: float = 0.3) -> dict:
    """Add Gaussian noise to position coordinates."""
    obj = copy.deepcopy(obj)
    if "position" in obj and isinstance(obj["position"], dict):
        original = copy.deepcopy(obj["position"])
        for axis in ["x", "y", "z"]:
            if axis in obj["position"]:
                obj["position"][axis] += rng.gauss(0, sigma)
                obj["position"][axis] = round(obj["position"][axis], 4)
        obj["_noise"] = {"type": "position_jitter", "sigma": sigma, "original": original}
    return obj


def _apply_distance_warp(obj: dict, rng: random.Random) -> dict:
    """Distort distance reading by a random factor."""
    obj = copy.deepcopy(obj)
    if "distance" in obj and isinstance(obj["distance"], (int, float)):
        original = obj["distance"]
        factor = rng.uniform(0.3, 3.0)
        obj["distance"] = round(obj["distance"] * factor, 4)
        obj["_noise"] = {"type": "distance_warp", "factor": round(factor, 3),
                         "original": original}
    return obj


def _apply_property_corrupt(obj: dict, rng: random.Random) -> dict:
    """Corrupt a specific non-boolean property."""
    obj = copy.deepcopy(obj)
    corruptible = []
    if obj.get("pickupable") is not None:
        corruptible.append("pickupable")
    if obj.get("receptacle") is not None:
        corruptible.append("receptacle")
    if obj.get("openable") is not None:
        corruptible.append("openable")
    if obj.get("toggleable") is not None:
        corruptible.append("toggleable")

    if corruptible:
        field_name = rng.choice(corruptible)
        obj[field_name] = not obj[field_name]
        obj["_noise"] = {"type": "property_corrupt", "field": field_name}
    return obj


def _apply_sensor_blackout(obj: dict, rng: random.Random) -> dict:
    """Replace some values with None to simulate sensor failure."""
    obj = copy.deepcopy(obj)
    blackout_fields = rng.sample(
        ["position", "distance", "visible"] + BOOL_STATE_FIELDS,
        k=rng.randint(1, 3),
    )
    for f in blackout_fields:
        if f in obj:
            obj[f] = None
    obj["_noise"] = {"type": "sensor_blackout", "blacked_out": blackout_fields}
    return obj


def _make_phantom(rng: random.Random) -> dict:
    """Generate a fake object that doesn't really exist in the scene."""
    template = rng.choice(PHANTOM_OBJECTS)
    phantom = copy.deepcopy(template)
    phantom["objectId"] = f"{phantom['objectType']}|phantom_{rng.randint(100, 999)}"
    phantom["position"] = {
        "x": round(rng.uniform(-2, 4), 3),
        "y": round(rng.uniform(0.5, 1.5), 3),
        "z": round(rng.uniform(-2, 4), 3),
    }
    phantom["distance"] = round(rng.uniform(0.5, 3.0), 3)
    phantom["pickupable"] = False
    phantom["receptacle"] = False
    phantom["_noise"] = {"type": "phantom_inject"}
    return phantom


def _apply_noise_to_list(
    objects: list[dict],
    noise_level: float,
    rng: random.Random,
    prof: NoiseProfile,
) -> tuple[list[dict], list[dict]]:
    """Apply noise to a list of objects. Returns (new_objects, noise_log)."""
    noise_log: list[dict] = []
    new_objects = []

    for obj in objects:
        if rng.random() >= noise_level:
            new_objects.append(obj)
            continue

        attack = prof.pick(rng)

        if attack == NoiseType.INFO_DROP:
            noise_log.append({
                "type": "info_drop",
                "objectId": obj["objectId"],
                "objectType": obj["objectType"],
            })
            continue

        elif attack == NoiseType.LABEL_SWAP:
            result = _apply_label_swap(obj, rng)
            if result:
                new_objects.append(result)
                noise_log.append(result.get("_noise", {}))
            else:
                new_objects.append(obj)

        elif attack == NoiseType.STATE_FLIP:
            result = _apply_state_flip(obj, rng)
            new_objects.append(result)
            noise_log.append(result.get("_noise", {}))

        elif attack == NoiseType.POSITION_JITTER:
            result = _apply_position_jitter(obj, rng)
            new_objects.append(result)
            noise_log.append(result.get("_noise", {}))

        elif attack == NoiseType.DISTANCE_WARP:
            result = _apply_distance_warp(obj, rng)
            new_objects.append(result)
            noise_log.append(result.get("_noise", {}))

        elif attack == NoiseType.PROPERTY_CORRUPT:
            result = _apply_property_corrupt(obj, rng)
            new_objects.append(result)
            noise_log.append(result.get("_noise", {}))

        elif attack == NoiseType.SENSOR_BLACKOUT:
            result = _apply_sensor_blackout(obj, rng)
            new_objects.append(result)
            noise_log.append(result.get("_noise", {}))

        elif attack == NoiseType.PHANTOM_INJECT:
            new_objects.append(obj)
            phantom = _make_phantom(rng)
            new_objects.append(phantom)
            noise_log.append(phantom.get("_noise", {}))

    n_phantoms = max(0, int(noise_level * 3) - 1)
    for _ in range(n_phantoms):
        if rng.random() < noise_level:
            phantom = _make_phantom(rng)
            new_objects.append(phantom)
            noise_log.append(phantom.get("_noise", {}))

    return new_objects, noise_log


def inject_noise(
    gt_data: dict,
    noise_level: float,
    seed: int = 42,
    profile: NoiseProfile | None = None,
) -> dict:
    """
    Inject noise into a clean ground-truth scene JSON.

    Works with both old format ("objects") and new format ("visible_objects"
    + "all_objects"). Noise is applied ONLY to visible objects, because
    only visible data reaches the LLM.

    Args:
        gt_data:     clean scene dict (from harvest step)
        noise_level: 0.0 ~ 1.0, probability that each object gets corrupted
        seed:        random seed for reproducibility
        profile:     NoiseProfile (defaults to DEFAULT_PROFILE)

    Returns:
        Noisy scene dict with "_noise_meta" tracking what was changed.
    """
    rng = random.Random(seed)
    prof = profile or DEFAULT_PROFILE
    noisy = copy.deepcopy(gt_data)

    has_visible_key = "visible_objects" in noisy
    if has_visible_key:
        vis_objects = noisy["visible_objects"]
    else:
        vis_objects = [o for o in noisy.get("objects", []) if o.get("visible", False)]

    original_count = len(vis_objects)
    new_visible, noise_log = _apply_noise_to_list(vis_objects, noise_level, rng, prof)

    if has_visible_key:
        noisy["visible_objects"] = new_visible
    else:
        non_visible = [o for o in noisy.get("objects", []) if not o.get("visible", False)]
        noisy["objects"] = new_visible + non_visible

    noisy["_noise_meta"] = {
        "noise_level": noise_level,
        "seed": seed,
        "num_corrupted": len(noise_log),
        "num_visible_original": original_count,
        "num_visible_after": len(new_visible),
        "log": noise_log,
    }

    return noisy


def run_noise_generation(
    noise_levels: list[float] | None = None,
    input_dir: str | None = None,
    output_dir: str | None = None,
    seeds_per_level: int | None = None,
):
    """Generate noisy variants for all GT files in `input_dir`.

    Args:
        noise_levels    : subset of levels (default: all in NOISE_LEVELS)
        input_dir       : directory of *_gt.json files (default: SCENES_DIR)
        output_dir      : directory to write noisy JSONs (default: NOISY_DIR)
        seeds_per_level : number of random seeds per (scene, level) pair.
                          Default: NOISE_SEEDS_PER_LEVEL.
    """
    ensure_dirs()
    levels = noise_levels or NOISE_LEVELS
    in_dir = input_dir or SCENES_DIR
    out_dir = output_dir or NOISY_DIR
    n_seeds = seeds_per_level if seeds_per_level is not None else NOISE_SEEDS_PER_LEVEL

    os.makedirs(out_dir, exist_ok=True)

    gt_files = sorted(glob.glob(os.path.join(in_dir, "*_gt.json")))
    if not gt_files:
        print(f"No ground-truth files found in {in_dir}.")
        print("Run the harvest step first.")
        return

    print(f"SafeSight Noise Engine v{BENCHMARK_VERSION}")
    print(f"Input  dir         : {in_dir}")
    print(f"Output dir         : {out_dir}")
    print(f"Ground-truth files : {len(gt_files)}")
    print(f"Noise levels       : {levels}")
    print(f"Seeds per level    : {n_seeds}")
    total = len(gt_files) * len(levels) * n_seeds
    print(f"Total outputs      : {total}")
    print("-" * 60)

    count = 0
    for gt_path in gt_files:
        scene_id = os.path.basename(gt_path).replace("_gt.json", "")
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        for level in levels:
            for s in range(n_seeds):
                seed = int(level * 1000) + s
                noisy = inject_noise(gt_data, noise_level=level, seed=seed)

                out_name = f"{scene_id}_noise{level:.1f}_seed{seed}.json"
                out_path = os.path.join(out_dir, out_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(noisy, f, indent=2, ensure_ascii=False)

                count += 1

        print(f"  [OK] {scene_id} → {len(levels) * n_seeds} variants")

    print(f"\nDone. {count} noisy files saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeSight Noise Engine")
    parser.add_argument(
        "--noise-levels", nargs="+", type=float,
        help="Specific noise levels (default: all from config)",
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Directory of *_gt.json files (default: data/scenes/)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write noisy JSONs (default: data/noisy/)",
    )
    parser.add_argument(
        "--seeds-per-level", type=int, default=None,
        help=f"Random seeds per (scene, level) pair (default: {NOISE_SEEDS_PER_LEVEL})",
    )
    args = parser.parse_args()
    run_noise_generation(
        noise_levels=args.noise_levels,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seeds_per_level=args.seeds_per_level,
    )
