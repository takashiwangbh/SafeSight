"""Configuration for the cross-verification pipeline.

Pipeline goal: audit simulator-derived hazard labels with an independent
large-model jury, producing agreement statistics that can be cited in the
paper without inflating claims.

Layout under data/cross_verify/:
    sampled_scenes.json            stratified sample manifest (deterministic)
    jury_results/<short>/<scene_basename>.json
                                  one file per (jury_model, scene)
    consensus/per_scene.csv        one row per sampled scene
    consensus/summary.json         top-line agreement numbers
    consensus/by_segment.csv       breakdown by severity / room / recipe
"""

from __future__ import annotations

import os

from benchmark.config import DATA_DIR, SCENES_DIR, SAFE_SCENES_DIR


# ─── Paths ───────────────────────────────────────────────────────────────

CROSS_VERIFY_DIR = os.path.join(DATA_DIR, "cross_verify")
SAMPLED_SCENES_FILE = os.path.join(CROSS_VERIFY_DIR, "sampled_scenes.json")
JURY_RESULTS_DIR = os.path.join(CROSS_VERIFY_DIR, "jury_results")
CONSENSUS_DIR = os.path.join(CROSS_VERIFY_DIR, "consensus")


# ─── Stratified sampling parameters ──────────────────────────────────────

N_HAZARD = 250        # hazardous scenes sampled from data/scenes/
N_SAFE = 250          # safe scenes sampled from data/safe_scenes/
RANDOM_SEED = 42

# Stratification keys for the hazardous half (severity + room_type).
# safe scenes only need room_type stratification since they all share
# severity == "none".
HAZARD_STRATIFY_KEYS = ("severity", "room_type")
SAFE_STRATIFY_KEYS = ("room_type",)


# ─── Jury model registry ─────────────────────────────────────────────────
#
# Each entry has a ``provider`` field that ``jury_eval.py`` dispatches on:
#
#   provider="huggingface"  → loaded locally via vlm_client.load_model().
#                             ``hf_id`` is the HuggingFace model id.
#                             vlm_client._is_quantized() auto-detects the
#                             "bnb-4bit" tag, so no extra kwargs are needed.
#                             The model is loaded, used on all scenes, then
#                             unloaded — so a single A6000 (48 GiB) works.
#
#   provider="openai"       → called over the network via the official
#                             OpenAI Responses API (cross_verify.openai_client).
#                             Requires OPENAI_API_KEY env var.  Optional
#                             ``reasoning_effort`` ∈ {minimal, low, medium, high}.
#
# If a HF id is gated / missing on your environment, swap with an AWQ
# alternative such as ``Qwen/Qwen2.5-72B-Instruct-AWQ`` or
# ``casperhansen/llama-3-70b-instruct-awq`` (requires ``pip install autoawq``).

JURY_MODELS: list[dict] = [
    {
        "short_name":  "qwen2.5-72b-bnb4",
        "hf_id":       "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "label":       "Qwen2.5-72B-Instruct",
        "provider":    "huggingface",
    },
    {
        "short_name":  "llama3.1-70b-bnb4",
        "hf_id":       "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "label":       "Llama-3.1-70B-Instruct",
        "provider":    "huggingface",
    },
    {
        "short_name":      "gpt-5.5",
        "hf_id":           "gpt-5.5",   # OpenAI API model name
        "label":           "GPT-5.5",
        "provider":        "openai",
        "reasoning_effort": "medium",
    },
]


# ─── Generation settings (kept identical across jury models) ─────────────
#
# Pure greedy decoding so the jury verdict is deterministic regardless of
# torch / CUDA state.  `do_sample=False` makes `temperature` / `top_p` /
# `top_k` no-ops on the Transformers side, so we omit them to avoid the
# "temperature is ignored" warning in newer transformers releases.

JURY_GENERATE_KWARGS = dict(
    max_new_tokens=512,
    do_sample=False,
)
JURY_SEED = 42


# ─── Cache dir default ──────────────────────────────────────────────────

DEFAULT_CACHE_DIR = os.environ.get(
    "HF_HOME", "/data/huggingface_cache"
)


def ensure_dirs() -> None:
    os.makedirs(CROSS_VERIFY_DIR, exist_ok=True)
    os.makedirs(JURY_RESULTS_DIR, exist_ok=True)
    os.makedirs(CONSENSUS_DIR, exist_ok=True)


__all__ = [
    "CROSS_VERIFY_DIR", "SAMPLED_SCENES_FILE",
    "JURY_RESULTS_DIR", "CONSENSUS_DIR",
    "N_HAZARD", "N_SAFE", "RANDOM_SEED",
    "HAZARD_STRATIFY_KEYS", "SAFE_STRATIFY_KEYS",
    "JURY_MODELS",
    "JURY_GENERATE_KWARGS", "JURY_SEED",
    "DEFAULT_CACHE_DIR",
    "SCENES_DIR", "SAFE_SCENES_DIR",
    "ensure_dirs",
]
