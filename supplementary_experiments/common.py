"""Shared helpers for the supplementary experiments.

Centralises:
  * path constants (re-using the main benchmark config)
  * the *data-driven* model choices for both studies
  * result-JSON loading + confusion classification that is byte-for-byte
    consistent with ``benchmark/score/scorer_v2.py`` (we import its helpers
    rather than re-implementing, so Table 4 / Appendix C numbers line up
    with the main results tables).

Nothing here needs a GPU or torch; the offline ``select_*`` / ``score_*``
scripts depend only on this module + the standard library.
"""

from __future__ import annotations

import glob
import json
import os
import sys

# ─── Make `benchmark.*` and `llm_client` importable ──────────────────────
_MY_EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MY_EXP_ROOT not in sys.path:
    sys.path.insert(0, _MY_EXP_ROOT)

from benchmark.config import (  # noqa: E402
    DATA_DIR, RESULTS_BASELINE_DIR, RESULTS_BASELINE_SAFE_DIR,
    RESULTS_NOISY_DIR, RESULTS_NOISY_SAFE_DIR, SCENES_DIR, SAFE_SCENES_DIR,
)
from benchmark.score.scorer_v2 import (  # noqa: E402
    _assessment_polarity, hazard_alignment,
)
from llm_client import parse_llm_response  # noqa: E402


# ─── Output locations (everything new lives here) ────────────────────────
SUPP_DIR = os.path.join(DATA_DIR, "supplementary")
MITIGATION_DIR = os.path.join(SUPP_DIR, "mitigation")
VISUAL_NOISE_DIR = os.path.join(SUPP_DIR, "visual_noise")


def ensure_supp_dirs() -> None:
    for d in (SUPP_DIR, MITIGATION_DIR, VISUAL_NOISE_DIR):
        os.makedirs(d, exist_ok=True)


# ─── Data-driven model selection ─────────────────────────────────────────
#
# Chosen from data/scores/v2/by_model_track_scenario.csv:
#
#   Mitigation needs HIGH-FAR "yes-man" models (room to improve):
#     qwen2vl-7b        text_and_image  baseline_safe FAR = 0.882, R = 0.987
#     llama3.2-vision   text_and_image  baseline_safe FAR = 0.761, R = 0.888
#   (Qwen2.5-VL-7B is deliberately NOT used here: its text+image FAR is only
#    0.079 — already a strong discriminator with nothing to mitigate.)
#
#   Visual noise needs the BEST clean image-track recall (so degradation is
#   meaningful):
#     llama3.2-vision   image_only  baseline recall = 0.944  (highest)
#     qwen2vl-7b        image_only  baseline recall = 0.934
#   (Qwen2.5-VL-7B image_only recall is only 0.273 — fails even on clean
#    images, so corrupting them would prove nothing.)

# short_name → HuggingFace id (matches benchmark.evaluate.baseline_eval registry)
MODEL_HF_ID = {
    "qwen2vl-7b":          "Qwen/Qwen2-VL-7B-Instruct",
    "llama3.2-vision-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "qwen2.5-vl-7b":       "Qwen/Qwen2.5-VL-7B-Instruct",
}

MITIGATION_MODELS = ["qwen2vl-7b", "llama3.2-vision-11b"]
MITIGATION_TRACK = "text_and_image"

VISUAL_NOISE_MODELS = ["qwen2vl-7b", "llama3.2-vision-11b"]
VISUAL_NOISE_TRACK = "image_only"


# ─── Result-JSON helpers ─────────────────────────────────────────────────

SCENARIO_DIRS = {
    "baseline":      RESULTS_BASELINE_DIR,
    "baseline_safe": RESULTS_BASELINE_SAFE_DIR,
    "noisy":         RESULTS_NOISY_DIR,
    "noisy_safe":    RESULTS_NOISY_SAFE_DIR,
}


def load_result(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def parsed_of(result: dict) -> dict:
    """Return the parsed-output dict from a result JSON (re-parsing the raw
    string when the stored ``parsed`` block is missing/empty)."""
    lr = result.get("llm_result", {})
    parsed = lr.get("parsed")
    if isinstance(parsed, dict) and parsed:
        return parsed
    raw = lr.get("raw_response", "")
    return parse_llm_response(raw) if raw else {}


def gt_is_safe(result: dict) -> bool:
    return bool(result.get("ground_truth", {}).get("is_safe", False))


def classify_confusion(result: dict, parsed: dict | None = None) -> str:
    """TP / FP / TN / FN — identical logic to scorer_v2."""
    if parsed is None:
        parsed = parsed_of(result)
    pred_dangerous = _assessment_polarity(parsed) == "dangerous"
    safe = gt_is_safe(result)
    if safe:
        return "FP" if pred_dangerous else "TN"
    return "TP" if pred_dangerous else "FN"


def confidence_of(parsed: dict) -> float | None:
    c = parsed.get("confidence")
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


def iter_results(scenario: str, model_short: str, track: str):
    """Yield (path, result_dict) for every result file of one config."""
    base = SCENARIO_DIRS[scenario]
    d = os.path.join(base, f"{model_short}_{track}")
    if not os.path.isdir(d):
        return
    for path in sorted(glob.glob(os.path.join(d, "*_result.json"))):
        r = load_result(path)
        if r is not None:
            yield path, r


# ─── Scene / image lookup ────────────────────────────────────────────────

def hazard_scene_paths(basename: str) -> tuple[str, str]:
    """Return (gt_json_path, png_path) for a hazard scene basename."""
    gt = os.path.join(SCENES_DIR, f"{basename}_gt.json")
    png = os.path.join(SCENES_DIR, f"{basename}.png")
    return gt, png


def safe_scene_gt(basename: str) -> str:
    return os.path.join(SAFE_SCENES_DIR, f"{basename}_gt.json")


def load_scene(gt_path: str) -> dict | None:
    return load_result(gt_path)  # same JSON-load logic


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"
