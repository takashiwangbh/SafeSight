import json
from typing import Optional

from ai2thor.controller import Controller


class ThorWrapper:
    """Wraps ai2thor Controller to provide a clean interface for LLM integration."""

    ALLOWED_ACTIONS = {
        "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
        "RotateLeft", "RotateRight", "LookUp", "LookDown",
        "PickupObject", "PutObject", "DropHandObject", "ThrowObject",
        "OpenObject", "CloseObject",
        "ToggleObjectOn", "ToggleObjectOff",
        "SliceObject", "CookObject", "CleanObject", "DirtyObject",
        "BreakObject", "FillObjectWithLiquid",
        "Done",
    }

    def __init__(
        self,
        scene: str = "FloorPlan1",
        grid_size: float = 0.25,
        visibility_distance: float = 1.5,
        render_depth: bool = False,
        render_instance_seg: bool = False,
        fov: int = 90,
        width: int = 600,
        height: int = 600,
    ):
        self.controller = Controller(
            scene=scene,
            gridSize=grid_size,
            visibilityDistance=visibility_distance,
            renderDepthImage=render_depth,
            renderInstanceSegmentation=render_instance_seg,
            fieldOfView=fov,
            width=width,
            height=height,
        )
        self.last_event = self.controller.last_event

    def get_scene_state(self) -> dict:
        """Return structured scene state suitable for LLM prompt assembly."""
        meta = self.last_event.metadata

        agent = meta["agent"]
        agent_info = {
            "position": agent["position"],
            "rotation": agent["rotation"],
            "cameraHorizon": agent["cameraHorizon"],
            "isStanding": agent.get("isStanding", True),
            "heldObject": (
                meta["inventoryObjects"][0]["objectId"]
                if meta["inventoryObjects"]
                else None
            ),
        }

        objects = []
        for obj in meta["objects"]:
            objects.append({
                "objectId": obj["objectId"],
                "objectType": obj["objectType"],
                "position": obj["position"],
                "visible": obj["visible"],
                "distance": obj["distance"],
                "isPickedUp": obj.get("isPickedUp", False),
                "isOpen": obj.get("isOpen", False),
                "isToggled": obj.get("isToggled", False),
                "isSliced": obj.get("isSliced", False),
                "isBroken": obj.get("isBroken", False),
                "isCooked": obj.get("isCooked", False),
                "isDirty": obj.get("isDirty", False),
                "isFilledWithLiquid": obj.get("isFilledWithLiquid", False),
                "pickupable": obj["pickupable"],
                "openable": obj.get("openable", False),
                "toggleable": obj.get("toggleable", False),
                "receptacle": obj["receptacle"],
            })

        return {
            "sceneName": meta["sceneName"],
            "agent": agent_info,
            "objects": objects,
        }

    def get_visible_objects_summary(self) -> str:
        """Generate a human-readable environment description for the LLM prompt."""
        state = self.get_scene_state()
        lines = []
        lines.append(f"Scene: {state['sceneName']}")

        held = state["agent"]["heldObject"]
        lines.append(f"Held object: {held if held else 'None'}")

        pos = state["agent"]["position"]
        lines.append(f"Agent position: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")

        rot_y = state["agent"]["rotation"]["y"]
        horizon = state["agent"]["cameraHorizon"]
        lines.append(f"Agent rotation Y: {rot_y:.1f}, camera horizon: {horizon:.1f}")
        lines.append("")
        lines.append("Visible objects:")

        visible = [o for o in state["objects"] if o["visible"]]
        visible.sort(key=lambda o: o["distance"])

        for obj in visible:
            props = []
            if obj["pickupable"]:
                props.append("pickupable")
            if obj["openable"]:
                status = "open" if obj["isOpen"] else "closed"
                props.append(f"openable({status})")
            if obj["toggleable"]:
                status = "on" if obj["isToggled"] else "off"
                props.append(f"toggleable({status})")
            if obj["receptacle"]:
                props.append("receptacle")
            if obj["isSliced"]:
                props.append("sliced")
            if obj["isBroken"]:
                props.append("broken")
            if obj["isCooked"]:
                props.append("cooked")
            if obj["isDirty"]:
                props.append("dirty")
            if obj["isFilledWithLiquid"]:
                props.append("filled")

            prop_str = ", ".join(props) if props else ""
            lines.append(
                f"  - {obj['objectType']} "
                f"(id: {obj['objectId']}, dist: {obj['distance']:.2f}m) "
                f"[{prop_str}]"
            )

        if not visible:
            lines.append("  (No visible objects — try rotating or moving)")

        return "\n".join(lines)

    def execute_action(self, action_dict: dict) -> dict:
        """
        Execute a single action in the simulator.

        Args:
            action_dict: e.g. {"action": "MoveAhead"} or
                         {"action": "PickupObject", "objectId": "Mug|0.25|0.5|1.0"}

        Returns:
            {"success": bool, "error": str, "message": str}
        """
        action_name = action_dict.get("action", "")

        if action_name == "Done":
            return {
                "success": True,
                "error": "",
                "message": "Task marked as Done.",
            }

        if action_name not in self.ALLOWED_ACTIONS:
            return {
                "success": False,
                "error": f"Unknown action: {action_name}",
                "message": f"Action '{action_name}' is not in the allowed list.",
            }

        self.last_event = self.controller.step(**action_dict)

        success = self.last_event.metadata["lastActionSuccess"]
        error_msg = self.last_event.metadata.get("errorMessage", "")

        return {
            "success": success,
            "error": error_msg if not success else "",
            "message": (
                f"Action {action_name} {'succeeded' if success else 'failed'}."
                + (f" Reason: {error_msg}" if not success else "")
            ),
        }

    def reset(self, scene: str = "FloorPlan1"):
        """Reset to a new scene."""
        self.controller.reset(scene)
        self.last_event = self.controller.step(
            action="Initialize",
            **self.controller.initialization_parameters,
        )

    def close(self):
        """Shut down the Unity simulator."""
        self.controller.stop()
