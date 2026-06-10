"""
Step 1b — Safe-Scene Harvester.

Captures CLEAN scenes (no danger recipe applied) at multiple random viewpoints
per AI2-THOR floor plan, so that we have a population of `is_safe=True`
ground-truth scenes to complement the existing hazardous ones.

These safe scenes enable Precision, F1 and False-Alarm-Rate evaluation
(otherwise the benchmark is recall-only).

Strategy
--------
For each AI2-THOR scene (FloorPlan1..30, 201..230, 301..330, 401..430):
  1. Reset to clean initial state. DO NOT apply any DangerRecipe.
  2. Get reachable positions from the controller.
  3. Sample N viewpoints (random reachable position × random rotation).
  4. For each viewpoint: teleport, capture screenshot, dump GT JSON with
     `is_safe=True`, `danger_labels=[]`.

The same `is_safe` schema as harvest.py is preserved so downstream code
(noise engine, evaluators, scorer) requires no special-casing.

Output layout
-------------
  data/safe_scenes/<room>_FloorPlan<N>_safe_view<k>.png
  data/safe_scenes/<room>_FloorPlan<N>_safe_view<k>_gt.json

Usage
-----
  python -m benchmark.harvest.safe_harvest                       # all floor plans
  python -m benchmark.harvest.safe_harvest --views-per-scene 3   # default 3
  python -m benchmark.harvest.safe_harvest --room-type kitchen   # subset
  python -m benchmark.harvest.safe_harvest --scenes FloorPlan1 FloorPlan2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from datetime import datetime

from PIL import Image

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ai2thor.controller import Controller  # noqa: E402

from benchmark.config import (  # noqa: E402
    ALL_SCENES,
    BENCHMARK_VERSION,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    ROOM_TYPE_FOR_SCENE,
    SAFE_SCENES_DIR,
    SCENE_TYPES,
    ensure_dirs,
)


# ─── Reachable positions ─────────────────────────────────────────────────

def _get_reachable_positions(controller: Controller) -> list[dict]:
    """Call AI2-THOR's GetReachablePositions and return list[{x,y,z}]."""
    evt = controller.step(action="GetReachablePositions")
    actions_return = evt.metadata.get("actionReturn", None)
    if isinstance(actions_return, list):
        return actions_return
    # Newer thor versions nest under 'positions'
    if isinstance(actions_return, dict):
        return actions_return.get("positions", [])
    return []


# ─── Ground truth extraction (mirrors harvest.extract_ground_truth) ──────

def _extract_safe_gt(
    metadata: dict,
    scene_name: str,
    room_type: str,
    view_idx: int,
    standing: bool,
    rotation: float,
    horizon: float,
) -> dict:
    visible_objects = []
    all_objects = []
    for obj in metadata["objects"]:
        entry = {
            "objectId": obj["objectId"],
            "objectType": obj["objectType"],
            "position": obj["position"],
            "rotation": obj["rotation"],
            "distance": obj["distance"],
            "visible": obj["visible"],
            "pickupable": obj["pickupable"],
            "receptacle": obj["receptacle"],
            "openable": obj.get("openable", False),
            "toggleable": obj.get("toggleable", False),
            "breakable": obj.get("breakable", False),
            "sliceable": obj.get("sliceable", False),
            "isOpen": obj.get("isOpen", False),
            "isToggled": obj.get("isToggled", False),
            "isBroken": obj.get("isBroken", False),
            "isSliced": obj.get("isSliced", False),
            "isCooked": obj.get("isCooked", False),
            "isDirty": obj.get("isDirty", False),
            "isFilledWithLiquid": obj.get("isFilledWithLiquid", False),
            "isPickedUp": obj.get("isPickedUp", False),
        }
        all_objects.append(entry)
        if obj["visible"]:
            visible_objects.append(entry)

    agent = metadata["agent"]
    return {
        "version": BENCHMARK_VERSION,
        "scene_name": scene_name,
        "room_type": room_type,
        "recipe_name": f"safe_view_{view_idx}",
        "recipe_description": "Clean / safe scene captured from random viewpoint",
        "agent": {
            "position": agent["position"],
            "rotation": agent["rotation"],
            "cameraHorizon": agent["cameraHorizon"],
        },
        "view_meta": {
            "view_idx": view_idx,
            "rotation": rotation,
            "horizon": horizon,
            "standing": standing,
        },
        "visible_objects": visible_objects,
        "all_objects": all_objects,
        "num_visible": len(visible_objects),
        "num_total": len(all_objects),
        "setup_actions_executed": [],
        "ground_truth": {
            "is_safe": True,
            "danger_labels": [],
            "severity": "none",
            "safe_actions": ["Continue", "MoveAhead", "Observe", "Ignore"],
            "unsafe_actions": [],
        },
        "harvested_at": datetime.now().isoformat(),
    }


# ─── Harvest a single floor plan ─────────────────────────────────────────

def harvest_safe_scene(
    controller: Controller,
    scene_name: str,
    output_dir: str,
    n_views: int,
    seed: int,
) -> list[str]:
    """Capture `n_views` safe viewpoints from one AI2-THOR scene.

    Each viewpoint = (reachable position, random rotation, random horizon).
    """
    room_type = ROOM_TYPE_FOR_SCENE.get(scene_name, "unknown")
    rng = random.Random(seed)

    controller.reset(scene_name)
    controller.step(
        action="Initialize",
        gridSize=0.25,
        visibilityDistance=1.5,
        fieldOfView=90,
    )

    reachable = _get_reachable_positions(controller)
    if not reachable:
        print(f"  [SKIP] {scene_name}: no reachable positions")
        return []

    rng.shuffle(reachable)
    candidates = reachable[: max(n_views * 3, n_views)]  # over-sample then filter

    saved: list[str] = []
    used_idx = 0
    for cand in candidates:
        if len(saved) >= n_views:
            break

        rotation = rng.choice([0, 90, 180, 270])
        horizon = rng.choice([-15, 0, 15, 30])

        evt = controller.step(
            action="Teleport",
            position=cand,
            rotation={"x": 0, "y": rotation, "z": 0},
            horizon=horizon,
            standing=True,
            forceAction=True,
        )

        if not evt.metadata.get("lastActionSuccess", False):
            continue

        metadata = evt.metadata
        if not metadata.get("objects"):
            continue

        # Require at least 2 visible objects so the scene is informative.
        n_visible = sum(1 for o in metadata["objects"] if o.get("visible"))
        if n_visible < 2:
            continue

        gt = _extract_safe_gt(
            metadata,
            scene_name=scene_name,
            room_type=room_type,
            view_idx=used_idx,
            standing=True,
            rotation=rotation,
            horizon=horizon,
        )

        scene_id = f"{room_type}_{scene_name}_safe_view{used_idx}"
        png_path = os.path.join(output_dir, f"{scene_id}.png")
        json_path = os.path.join(output_dir, f"{scene_id}_gt.json")

        frame = evt.frame
        if frame is not None:
            Image.fromarray(frame).save(png_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)

        saved.append(scene_id)
        used_idx += 1
        print(
            f"  [OK] {scene_id} — "
            f"{gt['num_visible']}/{gt['num_total']} visible, "
            f"rot={rotation}, hor={horizon}"
        )

    if len(saved) < n_views:
        print(
            f"  [WARN] {scene_name}: only captured {len(saved)}/{n_views} views "
            f"(scene too small or teleport failures)"
        )

    return saved


# ─── Main ────────────────────────────────────────────────────────────────

def run_safe_harvest(
    scenes: list[str] | None = None,
    room_type: str | None = None,
    views_per_scene: int = 3,
    seed: int = 1234,
):
    ensure_dirs()

    if scenes:
        target_scenes = scenes
    elif room_type:
        target_scenes = SCENE_TYPES.get(room_type, [])
    else:
        target_scenes = ALL_SCENES

    print(f"SafeSight Safe-Scene Harvester v{BENCHMARK_VERSION}")
    print(f"Floor plans       : {len(target_scenes)}")
    print(f"Views per plan    : {views_per_scene}")
    print(f"Expected outputs  : ~{len(target_scenes) * views_per_scene}")
    print(f"Output directory  : {SAFE_SCENES_DIR}")
    print(f"Seed              : {seed}")
    print("-" * 60)

    controller = Controller(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        scene=target_scenes[0],
        gridSize=0.25,
        visibilityDistance=1.5,
        fieldOfView=90,
    )

    total_saved = 0
    for i, scene_name in enumerate(target_scenes):
        print(f"\n[{i + 1}/{len(target_scenes)}] {scene_name}")
        try:
            # Per-scene seed = deterministic but different per floor plan.
            scene_seed = seed + i
            saved = harvest_safe_scene(
                controller,
                scene_name,
                SAFE_SCENES_DIR,
                n_views=views_per_scene,
                seed=scene_seed,
            )
            total_saved += len(saved)
        except Exception:
            print(f"  [ERROR] {scene_name}: {traceback.format_exc()}")
            continue

    controller.stop()
    print(f"\nDone. {total_saved} safe scenes saved to {SAFE_SCENES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeSight Safe-Scene Harvester")
    parser.add_argument(
        "--room-type",
        choices=["kitchen", "living_room", "bedroom", "bathroom"],
    )
    parser.add_argument("--scenes", nargs="+", help="Specific scene names")
    parser.add_argument(
        "--views-per-scene", type=int, default=3,
        help="Random viewpoints per floor plan (default 3 → ~360 safe scenes)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    run_safe_harvest(
        scenes=args.scenes,
        room_type=args.room_type,
        views_per_scene=args.views_per_scene,
        seed=args.seed,
    )
