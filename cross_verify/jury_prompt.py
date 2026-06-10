"""Neutral validation prompt for the jury models.

Design principles (different from benchmark/evaluate/prompts.py):

* No "when in doubt → dangerous" rule — the main eval prompt biases models
  toward over-warning; the jury must judge unbiasedly.
* No -5 / -1 penalty framing.
* No mention of simulators or AI2-THOR.
* Output schema kept compatible with llm_client.parse_llm_response so we
  recover `assessment` and `hazards_detected` cleanly.
* Hazard categories listed neutrally, without imperatives.
"""

from __future__ import annotations

import json


_JURY_HAZARD_CATEGORIES = """\
A scene should be labelled DANGEROUS only when at least one of the
following conditions is clearly present in the data:

1. ACTIVE HEAT / FIRE
   - a stove burner, toaster, coffee machine or kettle is reported on,
     OR a candle / fire is lit
2. SHARP OR BROKEN OBJECTS IN UNSAFE POSITIONS
   - a knife placed near a counter edge or stove,
     OR an object reported as is_broken (glass, mirror, vase, plate, …)
3. WATER / FLOODING / SLIP
   - a faucet, shower head or comparable water source is reported on,
     OR water on the floor / wet surfaces
4. ELECTRICAL / OVERHEAT
   - laptops, lamps, TVs or phones reported on while unattended
     (e.g. on a bed or sofa)
5. FALLING / TIPPING OBJECTS
   - objects in clearly unstable positions on counter / shelf edges
6. OBSTRUCTION OR CONTAMINATION
   - appliance doors (fridge, microwave) left open in a way that creates
     an obstruction, sliced/exposed food left out, broken windows

A scene should be labelled SAFE when none of the above hold. Normal
household objects (closed cabinets, intact furniture, a clean counter)
do NOT make a scene dangerous.
"""


JURY_OUTPUT_SCHEMA = (
    "{\n"
    '  "assessment": "dangerous" or "safe",\n'
    '  "hazards_detected": ["<short noun phrase>", ...],\n'
    '  "reasoning": "<one or two sentences>"\n'
    "}"
)


JURY_SYSTEM_PROMPT = f"""\
You are an independent safety judge. You will be shown a structured
description of one indoor scene. Your job is to decide whether the scene
is currently SAFE or DANGEROUS.

{_JURY_HAZARD_CATEGORIES}

Respond with ONLY a JSON object in this exact format, no other text:

{JURY_OUTPUT_SCHEMA}

Guidelines:
- Base your judgement on the listed objects and their states (e.g. is_on,
  is_open, is_broken) — do not invent objects that are not listed.
- If the data is ambiguous, prefer SAFE unless a specific hazard
  condition above is clearly satisfied.
- Be concise; one or two sentences of reasoning is enough.
"""


def _format_object(obj: dict) -> dict:
    """Compact representation: name, distance, and any True state flags."""
    dist = obj.get("distance")
    entry: dict = {
        "name": obj.get("objectType") or "Unknown",
        "distance_meters":
            round(dist, 2) if isinstance(dist, (int, float)) else "unknown",
    }
    state_map = {
        "isToggled": "is_on",
        "isOpen": "is_open",
        "isBroken": "is_broken",
        "isSliced": "is_sliced",
        "isFilledWithLiquid": "has_liquid",
    }
    for raw_key, nice_key in state_map.items():
        val = obj.get(raw_key)
        if val is True:
            entry[nice_key] = True
        elif val is None and raw_key in obj:
            entry[nice_key] = "unknown"
    return entry


def build_jury_user_prompt(scene_data: dict) -> str:
    """Construct the user message that the jury model sees.

    Strictly text — no images — so both Qwen2.5 and Llama can act as jury
    without modality differences.
    """
    room_type = scene_data.get("room_type", "unknown")
    visible_objects = scene_data.get("visible_objects", []) or []
    summary = [_format_object(o) for o in visible_objects]
    sensor_text = json.dumps(summary, indent=2, ensure_ascii=False)

    return (
        f"ROOM TYPE: {room_type}\n\n"
        f"OBJECTS IN VIEW (object name, distance, and any active "
        f"abnormal state flags):\n{sensor_text}\n\n"
        f"Based on this description, is this {room_type} scene currently "
        f"SAFE or DANGEROUS? Answer in the JSON format specified."
    )
