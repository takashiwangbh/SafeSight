"""
Step 1 — Data Harvester.

Loads ai2thor scenes, applies danger recipes, and saves:
  - scene screenshot (PNG)  — captured AFTER navigating agent to face the danger
  - ground-truth JSON       — only visible objects, with danger labels

The agent actively rotates / teleports to face the primary danger object,
so both the screenshot and the JSON faithfully reflect what a real robot
would perceive from that viewpoint.

Usage:
    python -m benchmark.harvest.harvest                       # all scenes
    python -m benchmark.harvest.harvest --room-type kitchen   # kitchen only
    python -m benchmark.harvest.harvest --scenes FloorPlan1 FloorPlan2
"""

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai2thor.controller import Controller
from benchmark.config import (
    ALL_SCENES, BENCHMARK_VERSION, RENDER_HEIGHT, RENDER_WIDTH,
    ROOM_TYPE_FOR_SCENE, SCENES_DIR, SCENE_TYPES, ensure_dirs,
)
from benchmark.harvest.danger_recipes import DangerRecipe, get_applicable_recipes


# ═══════════════════════════════════════════════════════════════════════════
# Object helpers
# ═══════════════════════════════════════════════════════════════════════════

def find_object_by_type(metadata: dict, object_type: str) -> dict | None:
    for obj in metadata["objects"]:
        if obj["objectType"] == object_type:
            return obj
    return None


def find_all_objects_by_type(metadata: dict, object_type: str) -> list[dict]:
    return [o for o in metadata["objects"] if o["objectType"] == object_type]


def _get_receptacle_bounds(metadata: dict, rtype: str):
    """Return (obj_dict, min_corner, max_corner) for the first matching receptacle."""
    obj = find_object_by_type(metadata, rtype)
    if not obj:
        return None, None, None
    corners = obj.get("axisAlignedBoundingBox", {}).get("cornerPoints", [])
    if not corners:
        return obj, None, None
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    zs = [p[2] for p in corners]
    return obj, {"x": min(xs), "y": min(ys), "z": min(zs)}, {"x": max(xs), "y": max(ys), "z": max(zs)}


# ═══════════════════════════════════════════════════════════════════════════
# Runtime feasibility
# ═══════════════════════════════════════════════════════════════════════════

_ACTION_TO_PROPERTY = {
    "ToggleObjectOn":  "toggleable",
    "ToggleObjectOff": "toggleable",
    "OpenObject":      "openable",
    "CloseObject":     "openable",
    "BreakObject":     "breakable",
    "SliceObject":     "sliceable",
    "FillObjectWithLiquid": "canFillWithLiquid",
}


def _object_supports_action(obj_meta: dict, action_name: str) -> bool:
    prop = _ACTION_TO_PROPERTY.get(action_name)
    if prop is None:
        return True
    return bool(obj_meta.get(prop, False))


def recipe_is_feasible(metadata: dict, recipe: DangerRecipe) -> bool:
    for obj_type in recipe.required_object_types:
        obj = find_object_by_type(metadata, obj_type)
        if not obj:
            return False

    for step in recipe.setup_steps:
        action_name = step["action"]
        obj_type = step.get("objectType")
        if not obj_type:
            continue
        if action_name in ("EdgePlace", "PutOn", "PutNear", "TeleportObject"):
            continue
        obj = find_object_by_type(metadata, obj_type)
        if obj and not _object_supports_action(obj, action_name):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Execute recipe — handles all action types correctly
# ═══════════════════════════════════════════════════════════════════════════

def execute_recipe(controller: Controller, metadata: dict,
                   recipe: DangerRecipe) -> list[dict]:
    """
    Execute a danger recipe's setup steps.

    Supported action types:
      - Standard AI2-THOR actions: ToggleObjectOn, OpenObject, BreakObject,
        SliceObject, FillObjectWithLiquid  → use forceAction=True
      - "EdgePlace"  → PickupObject + PutObject on receptacle, then shift
        toward the bounding-box edge via PlaceObjectAtPoint
      - "PutOn"      → PickupObject + PutObject on target receptacle
      - "PutNear"    → PickupObject + TeleportObject next to reference object
      - select="all" → apply action to every instance of that objectType
    """
    executed = []

    for step in recipe.setup_steps:
        action_name = step["action"]
        obj_type = step.get("objectType")

        # ── Resolve targets ──────────────────────────────────────────
        if step.get("select") == "all" and obj_type:
            targets = find_all_objects_by_type(metadata, obj_type)
        elif obj_type:
            t = find_object_by_type(metadata, obj_type)
            targets = [t] if t else []
        else:
            targets = [None]

        for target in targets:
            if action_name == "EdgePlace":
                result = _do_edge_place(controller, target, step)
                executed.append(result)

            elif action_name == "PutOn":
                result = _do_put_on(controller, target, step, metadata)
                executed.append(result)

            elif action_name == "PutNear":
                result = _do_put_near(controller, target, step, metadata)
                executed.append(result)

            else:
                result = _do_standard_action(controller, target, step)
                executed.append(result)

        metadata = controller.last_event.metadata

    return executed


def _do_standard_action(controller, target, step) -> dict:
    """Execute a standard AI2-THOR action (Toggle, Open, Break, Slice, Fill…)."""
    action_name = step["action"]

    # If ToggleObjectOn but already on → toggle off first, then on again
    if action_name == "ToggleObjectOn" and target and target.get("isToggled"):
        controller.step(action="ToggleObjectOff", objectId=target["objectId"],
                        forceAction=True)

    # If OpenObject but already open → close first
    if action_name == "OpenObject" and target and target.get("isOpen"):
        controller.step(action="CloseObject", objectId=target["objectId"],
                        forceAction=True)

    action_dict = {"action": action_name}
    if target:
        action_dict["objectId"] = target["objectId"]
    if step.get("forceAction"):
        action_dict["forceAction"] = True
    if step.get("openness") is not None:
        action_dict["openness"] = step["openness"]
    if step.get("fillLiquid"):
        action_dict["fillLiquid"] = step["fillLiquid"]

    event = controller.step(**action_dict)
    return {
        "action": action_name,
        "objectId": action_dict.get("objectId", ""),
        "success": event.metadata["lastActionSuccess"],
        "errorMessage": event.metadata.get("errorMessage", ""),
    }


def _do_edge_place(controller, target, step) -> dict:
    """
    Place an object near the edge of a receptacle.

    Strategy:
      1. PickupObject (forceAction)
      2. PutObject on the receptacle (forceAction)  → natural resting pos
      3. Read the natural position and the receptacle bounding box
      4. Compute a position near the bounding-box edge
      5. PlaceObjectAtPoint to shift the object there
    """
    if not target:
        return {"action": "EdgePlace", "objectId": "", "success": False,
                "errorMessage": "No target object"}

    obj_id = target["objectId"]
    rec_type = step.get("receptacleType", "CounterTop")

    # Step 1: pick up
    e = controller.step(action="PickupObject", objectId=obj_id, forceAction=True)
    if not e.metadata["lastActionSuccess"]:
        return {"action": "EdgePlace", "objectId": obj_id, "success": False,
                "errorMessage": f"Pickup failed: {e.metadata.get('errorMessage','')}"}

    # Step 2: put on receptacle
    meta = controller.last_event.metadata
    rec = find_object_by_type(meta, rec_type)
    if not rec:
        return {"action": "EdgePlace", "objectId": obj_id, "success": False,
                "errorMessage": f"No {rec_type} found"}

    e = controller.step(action="PutObject", objectId=rec["objectId"], forceAction=True)
    if not e.metadata["lastActionSuccess"]:
        return {"action": "EdgePlace", "objectId": obj_id, "success": False,
                "errorMessage": f"PutObject failed: {e.metadata.get('errorMessage','')}"}

    # Step 3: read natural position
    meta = controller.last_event.metadata
    placed = None
    for o in meta["objects"]:
        if o["objectId"] == obj_id:
            placed = o
            break
    if not placed:
        return {"action": "EdgePlace", "objectId": obj_id, "success": False,
                "errorMessage": "Cannot find placed object"}

    natural = placed["position"]

    # Step 4: find the receptacle's nearest edge
    rec_obj = find_object_by_type(meta, rec_type)
    corners = rec_obj.get("axisAlignedBoundingBox", {}).get("cornerPoints", [])
    if not corners:
        return {"action": "EdgePlace", "objectId": obj_id, "success": True,
                "errorMessage": "No bounding box, placed at center"}

    xs = [p[0] for p in corners]
    zs = [p[2] for p in corners]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)

    # Determine which edge the object is closest to
    dist_to_edges = {
        "x+": abs(natural["x"] - x_max),
        "x-": abs(natural["x"] - x_min),
        "z+": abs(natural["z"] - z_max),
        "z-": abs(natural["z"] - z_min),
    }
    nearest_edge = min(dist_to_edges, key=dist_to_edges.get)

    MARGIN = 0.03
    edge_pos = dict(natural)
    if nearest_edge == "x+":
        edge_pos["x"] = x_max - MARGIN
    elif nearest_edge == "x-":
        edge_pos["x"] = x_min + MARGIN
    elif nearest_edge == "z+":
        edge_pos["z"] = z_max - MARGIN
    else:
        edge_pos["z"] = z_min + MARGIN

    # Step 5: shift to edge
    e = controller.step(
        action="PlaceObjectAtPoint",
        objectId=obj_id,
        position=edge_pos,
        forceKinematic=True,
    )
    success = e.metadata["lastActionSuccess"]

    if not success:
        # Try smaller shift
        edge_pos = dict(natural)
        half = dist_to_edges[nearest_edge] * 0.7
        if nearest_edge == "x+":
            edge_pos["x"] = natural["x"] + half
        elif nearest_edge == "x-":
            edge_pos["x"] = natural["x"] - half
        elif nearest_edge == "z+":
            edge_pos["z"] = natural["z"] + half
        else:
            edge_pos["z"] = natural["z"] - half

        e = controller.step(
            action="PlaceObjectAtPoint",
            objectId=obj_id,
            position=edge_pos,
            forceKinematic=True,
        )
        success = e.metadata["lastActionSuccess"]

    # Verify the object didn't fall to the floor
    meta = controller.last_event.metadata
    final_obj = None
    for o in meta["objects"]:
        if o["objectId"] == obj_id:
            final_obj = o
            break
    if final_obj and final_obj["position"]["y"] < natural["y"] - 0.3:
        # Fell off — revert to natural position
        controller.step(
            action="PlaceObjectAtPoint",
            objectId=obj_id,
            position=natural,
            forceKinematic=True,
        )
        return {"action": "EdgePlace", "objectId": obj_id, "success": True,
                "errorMessage": "Object fell off edge; reverted to natural pos on counter"}

    return {"action": "EdgePlace", "objectId": obj_id, "success": success,
            "errorMessage": e.metadata.get("errorMessage", "")}


def _do_put_on(controller, target, step, metadata) -> dict:
    """PickupObject + PutObject on target receptacle."""
    if not target:
        return {"action": "PutOn", "objectId": "", "success": False,
                "errorMessage": "No target"}

    obj_id = target["objectId"]
    rec_type = step.get("receptacleType")

    e = controller.step(action="PickupObject", objectId=obj_id, forceAction=True)
    if not e.metadata["lastActionSuccess"]:
        return {"action": "PutOn", "objectId": obj_id, "success": False,
                "errorMessage": f"Pickup failed: {e.metadata.get('errorMessage','')}"}

    meta = controller.last_event.metadata
    rec = find_object_by_type(meta, rec_type)
    if not rec:
        return {"action": "PutOn", "objectId": obj_id, "success": False,
                "errorMessage": f"No {rec_type} found"}

    e = controller.step(action="PutObject", objectId=rec["objectId"], forceAction=True)
    return {
        "action": "PutOn",
        "objectId": obj_id,
        "success": e.metadata["lastActionSuccess"],
        "errorMessage": e.metadata.get("errorMessage", ""),
    }


def _do_put_near(controller, target, step, metadata) -> dict:
    """TeleportObject to a position near the reference object (no pickup needed)."""
    if not target:
        return {"action": "PutNear", "objectId": "", "success": False,
                "errorMessage": "No target"}

    obj_id = target["objectId"]
    near_type = step.get("nearType")
    offset_x = step.get("offset_x", 0.15)
    offset_z = step.get("offset_z", 0.0)

    meta = controller.last_event.metadata
    near_obj = find_object_by_type(meta, near_type)
    if not near_obj:
        return {"action": "PutNear", "objectId": obj_id, "success": False,
                "errorMessage": f"No {near_type} found"}

    target_pos = {
        "x": near_obj["position"]["x"] + offset_x,
        "y": near_obj["position"]["y"] + 0.03,
        "z": near_obj["position"]["z"] + offset_z,
    }

    e = controller.step(
        action="TeleportObject",
        objectId=obj_id,
        position=target_pos,
        rotation={"x": 0, "y": 0, "z": 0},
        forceAction=True,
    )
    return {
        "action": "PutNear",
        "objectId": obj_id,
        "success": e.metadata["lastActionSuccess"],
        "errorMessage": e.metadata.get("errorMessage", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Navigate agent to face the primary danger object
# ═══════════════════════════════════════════════════════════════════════════

_BROKEN_SUFFIXES = ["_Broken", "Broken"]
_SLICED_SUFFIX = "Sliced"


def _get_navigation_targets(metadata: dict, recipe: DangerRecipe) -> list[dict]:
    candidates = []
    seen_ids = set()

    for req_type in recipe.required_object_types:
        for obj in metadata["objects"]:
            otype = obj["objectType"]
            oid = obj["objectId"]
            if oid in seen_ids:
                continue

            if otype == req_type:
                candidates.append(obj)
                seen_ids.add(oid)
            elif otype == req_type + _SLICED_SUFFIX:
                candidates.append(obj)
                seen_ids.add(oid)
            elif any(otype == req_type + suf for suf in _BROKEN_SUFFIXES):
                candidates.append(obj)
                seen_ids.add(oid)
            elif otype.startswith(req_type) and ("Sliced" in otype or "Broken" in otype):
                candidates.append(obj)
                seen_ids.add(oid)

    return candidates


def _any_target_visible(metadata: dict, target_ids: set[str]) -> bool:
    for obj in metadata["objects"]:
        if obj["objectId"] in target_ids and obj.get("visible", False):
            return True
    return False


def _compute_centroid(objects: list[dict]) -> dict:
    xs = [o["position"]["x"] for o in objects]
    ys = [o["position"]["y"] for o in objects]
    zs = [o["position"]["z"] for o in objects]
    n = len(objects)
    return {"x": sum(xs) / n, "y": sum(ys) / n, "z": sum(zs) / n}


def navigate_to_face_targets(controller: Controller,
                             target_objects: list[dict]) -> bool:
    """
    Navigate agent to see the danger like a first-person view.

    Strategy: try many candidate positions, collect those where the
    target is visible, then pick the best one (most visible targets,
    ideal distance, natural pitch).
    """
    if not target_objects:
        return False

    metadata = controller.last_event.metadata
    centroid = _compute_centroid(target_objects)
    target_ids = {o["objectId"] for o in target_objects}

    event = controller.step(action="GetReachablePositions")
    reachable = event.metadata.get("actionReturn", [])
    if not reachable:
        return _rotate_sweep(controller, target_ids)

    agent_y = metadata["agent"]["position"]["y"]

    obj_height = centroid["y"]
    is_floor_level = obj_height < 0.3
    is_high = obj_height > 1.2

    if is_floor_level:
        pitch_candidates = [35, 45, 25, 50]
    elif is_high:
        pitch_candidates = [-10, 0, -15, 10]
    else:
        pitch_candidates = [20, 25, 15, 30, 10]

    def dist_to_centroid(p):
        dx = p["x"] - centroid["x"]
        dz = p["z"] - centroid["z"]
        return math.sqrt(dx * dx + dz * dz)

    candidates = sorted(reachable, key=dist_to_centroid)

    # Collect visible viewpoints and score them
    best_score = -1
    best_state = None  # (pos, yaw, pitch)

    for pos in candidates[:20]:
        d = dist_to_centroid(pos)
        if d < 0.5 or d > 2.5:
            continue

        dx = centroid["x"] - pos["x"]
        dz = centroid["z"] - pos["z"]
        yaw = math.degrees(math.atan2(dx, dz)) % 360

        for pitch in pitch_candidates:
            controller.step(
                action="Teleport",
                position=dict(x=pos["x"], y=agent_y, z=pos["z"]),
                rotation=dict(x=0, y=yaw, z=0),
                horizon=pitch,
                standing=True,
                forceAction=True,
            )
            meta = controller.last_event.metadata

            # Count how many target objects are visible
            vis_count = sum(
                1 for o in meta["objects"]
                if o["objectId"] in target_ids and o.get("visible")
            )
            if vis_count == 0:
                continue

            # Score: more visible targets is better; prefer distance 0.8-1.5m
            dist_score = 1.0 - abs(d - 1.1) / 2.0
            score = vis_count * 10 + dist_score * 5

            if score > best_score:
                best_score = score
                best_state = (
                    dict(x=pos["x"], y=agent_y, z=pos["z"]),
                    yaw, pitch,
                )

            # Early exit if we found a great position
            if vis_count >= len(target_ids) and 0.7 < d < 1.5:
                break
        else:
            continue
        break

    if best_state:
        pos, yaw, pitch = best_state
        controller.step(
            action="Teleport",
            position=pos,
            rotation=dict(x=0, y=yaw, z=0),
            horizon=pitch,
            standing=True,
            forceAction=True,
        )
        return True

    return _rotate_sweep(controller, target_ids)


def _rotate_sweep(controller: Controller, target_ids: set[str]) -> bool:
    """Fallback: rotate 360° at several pitch angles."""
    for horizon in [15, 0, 30, 45, -10]:
        for _ in range(12):
            controller.step(action="RotateRight", degrees=30, forceAction=True)
            controller.step(
                action="Teleport",
                position=controller.last_event.metadata["agent"]["position"],
                rotation=controller.last_event.metadata["agent"]["rotation"],
                horizon=horizon,
                standing=True,
                forceAction=True,
            )
            if _any_target_visible(controller.last_event.metadata, target_ids):
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Extract ground truth
# ═══════════════════════════════════════════════════════════════════════════

def extract_ground_truth(metadata: dict, recipe: DangerRecipe, scene_name: str,
                         room_type: str, executed_actions: list[dict]) -> dict:
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
        "recipe_name": recipe.name,
        "recipe_description": recipe.description,
        "agent": {
            "position": agent["position"],
            "rotation": agent["rotation"],
            "cameraHorizon": agent["cameraHorizon"],
        },
        "visible_objects": visible_objects,
        "all_objects": all_objects,
        "num_visible": len(visible_objects),
        "num_total": len(all_objects),
        "setup_actions_executed": executed_actions,
        "ground_truth": {
            "is_safe": False,
            "danger_labels": recipe.danger_labels,
            "severity": recipe.severity,
            "safe_actions": recipe.safe_actions,
            "unsafe_actions": recipe.unsafe_actions,
        },
        "harvested_at": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Harvest one scene
# ═══════════════════════════════════════════════════════════════════════════

def harvest_scene(controller: Controller, scene_name: str,
                  output_dir: str) -> list[str]:
    room_type = ROOM_TYPE_FOR_SCENE.get(scene_name, "unknown")
    recipes = get_applicable_recipes(room_type)
    saved = []

    for recipe in recipes:
        controller.reset(scene_name)
        controller.step(action="Initialize", gridSize=0.25,
                        visibilityDistance=1.5, fieldOfView=90)

        metadata = controller.last_event.metadata
        if not recipe_is_feasible(metadata, recipe):
            print(f"  [SKIP] {recipe.name} — not feasible in {scene_name}")
            continue

        executed = execute_recipe(controller, metadata, recipe)

        any_success = any(a["success"] for a in executed)
        if not any_success:
            msgs = "; ".join(a["errorMessage"][:60] for a in executed if a["errorMessage"])
            print(f"  [SKIP] {recipe.name} — all setup failed: {msgs[:120]}")
            continue

        post_metadata = controller.last_event.metadata
        target_objects = _get_navigation_targets(post_metadata, recipe)

        navigated = False
        if target_objects:
            navigated = navigate_to_face_targets(controller, target_objects)
            if not navigated:
                target_names = [o["objectType"] for o in target_objects[:3]]
                print(f"  [WARN] {recipe.name} — could not face {target_names}, "
                      f"using best effort position")

        post_metadata = controller.last_event.metadata
        gt = extract_ground_truth(post_metadata, recipe, scene_name,
                                  room_type, executed)
        gt["target_visible"] = navigated

        scene_id = f"{room_type}_{scene_name}_{recipe.name}"
        png_path = os.path.join(output_dir, f"{scene_id}.png")
        json_path = os.path.join(output_dir, f"{scene_id}_gt.json")

        frame = controller.last_event.frame
        if frame is not None:
            Image.fromarray(frame).save(png_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)

        status = "OK" if navigated else "OK (target not visible)"
        saved.append(scene_id)
        print(f"  [{status}] {scene_id} — "
              f"{gt['num_visible']}/{gt['num_total']} visible, "
              f"actions: {[a['action']+('✓' if a['success'] else '✗') for a in executed]}")

    return saved


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_harvest(scenes: list[str] | None = None, room_type: str | None = None):
    ensure_dirs()

    if scenes:
        target_scenes = scenes
    elif room_type:
        target_scenes = SCENE_TYPES.get(room_type, [])
    else:
        target_scenes = ALL_SCENES

    print(f"SafeSight Harvester v{BENCHMARK_VERSION}")
    print(f"Scenes to process: {len(target_scenes)}")
    print(f"Output directory : {SCENES_DIR}")
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
            saved = harvest_scene(controller, scene_name, SCENES_DIR)
            total_saved += len(saved)
        except Exception:
            print(f"  [ERROR] {scene_name}: {traceback.format_exc()}")
            continue

    controller.stop()
    print(f"\nDone. {total_saved} danger scenes saved to {SCENES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeSight Data Harvester")
    parser.add_argument("--room-type",
                        choices=["kitchen", "living_room", "bedroom", "bathroom"])
    parser.add_argument("--scenes", nargs="+", help="Specific scene names")
    args = parser.parse_args()
    run_harvest(scenes=args.scenes, room_type=args.room_type)
