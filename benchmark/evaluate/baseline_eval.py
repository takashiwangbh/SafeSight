#!/usr/bin/env python3
"""
SafeSight — Baseline Evaluation on Clean (no-noise) Data.

Reads ground-truth scene JSONs from data/scenes/, runs LLM inference
in three evaluation tracks, and saves structured results.

Tracks:
  text_only      — sensor text only  (any model)
  image_only     — screenshot only   (VLM only)
  text_and_image — sensor + image    (VLM only)

Usage examples:

  # Run ALL configured models × applicable tracks (full auto):
  python -m benchmark.evaluate.baseline_eval

  # Run a single model (use short name or full HF id):
  python -m benchmark.evaluate.baseline_eval --models qwen2vl-7b

  # Test with 5 scenes:
  python -m benchmark.evaluate.baseline_eval --max-scenes 5

  # Specify tracks manually:
  python -m benchmark.evaluate.baseline_eval --models qwen2vl-7b --tracks image_only text_and_image

  # GPU diagnostic first (dry run):
  python -m benchmark.evaluate.baseline_eval --check-gpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from benchmark.config import BENCHMARK_VERSION, DATA_DIR, SCENES_DIR
from benchmark.evaluate.gpu_utils import (
    preflight_check,
    print_gpu_status,
)
from benchmark.evaluate.prompts import (
    IMAGE_ONLY_SYSTEM_PROMPT,
    IMAGE_ONLY_SYSTEM_PROMPT_SHORT,
    SAFETY_SYSTEM_PROMPT,
    SAFETY_SYSTEM_PROMPT_SHORT,
    build_image_only_prompt,
    build_multimodal_prompt,
    build_multimodal_prompt_short,
    build_text_only_prompt,
    build_text_only_prompt_short,
)
from benchmark.evaluate.vlm_client import EVAL_GENERATE_KWARGS, EVAL_SEED, load_model
from llm_client import parse_llm_response


# ─── Model Registry (short name → full config) ──────────────────────────

@dataclass
class ModelSpec:
    short_name: str        # folder / CLI name, e.g. "qwen2.5-7b"
    hf_id: str             # HuggingFace model id
    tracks: list[str]      # tracks this model should run
    is_vlm: bool = False   # whether it supports vision


MODEL_REGISTRY: list[ModelSpec] = [
    # ── Text-only: Qwen2.5 family (parameter scaling) ────────────────
    ModelSpec("qwen2.5-7b",   "Qwen/Qwen2.5-7B-Instruct",            ["text_only"]),
    ModelSpec("qwen2.5-14b",  "Qwen/Qwen2.5-14B-Instruct",           ["text_only"]),
    # ── Text-only: Qwen3 family (next-gen comparison) ────────────────
    ModelSpec("qwen3-14b",    "Qwen/Qwen3-14B",                       ["text_only"]),
    # ── Text-only: other families ────────────────────────────────────
    ModelSpec("mistral-7b",       "mistralai/Mistral-7B-Instruct-v0.2",        ["text_only"]),
    ModelSpec("mistral-nemo-12b", "mistralai/Mistral-Nemo-Instruct-2407",      ["text_only"]),
    ModelSpec("falcon-7b",        "tiiuae/falcon-7b-instruct",                 ["text_only"]),
    ModelSpec("llama3-chatqa-8b", "nvidia/Llama3-ChatQA-1.5-8B",              ["text_only"]),
    # ── VLM: Qwen2-VL ───────────────────────────────────────────────
    ModelSpec("qwen2vl-7b",   "Qwen/Qwen2-VL-7B-Instruct",
             ["text_only", "image_only", "text_and_image"], is_vlm=True),
    # ── VLM: Qwen2.5-VL (newer Qwen multimodal) ─────────────────────
    ModelSpec("qwen2.5-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct",
             ["text_only", "image_only", "text_and_image"], is_vlm=True),
    # ── VLM: LLaVA-1.6 family (parameter scaling) ───────────────────
    ModelSpec("llava1.6-7b",  "llava-hf/llava-v1.6-mistral-7b-hf",
             ["text_only", "image_only", "text_and_image"], is_vlm=True),
    ModelSpec("llava1.6-13b", "llava-hf/llava-v1.6-vicuna-13b-hf",
             ["text_only", "image_only", "text_and_image"], is_vlm=True),
    # ── VLM: Llama-3.2-Vision (Mllama, gated) ───────────────────────
    ModelSpec("llama3.2-vision-11b", "meta-llama/Llama-3.2-11B-Vision-Instruct",
             ["text_only", "image_only", "text_and_image"], is_vlm=True),
    # ── VLM: InternVL2 (custom code, model.chat API) ────────────────
    ModelSpec("internvl2-8b", "OpenGVLab/InternVL2-8B",
             ["text_only", "image_only", "text_and_image"], is_vlm=True),
    # ── VLM: Phi-3.5-Vision (small efficient VLM) ───────────────────
    ModelSpec("phi3.5-vision", "microsoft/Phi-3.5-vision-instruct",
             ["text_only", "image_only", "text_and_image"], is_vlm=True),
]

_REGISTRY_BY_SHORT = {m.short_name: m for m in MODEL_REGISTRY}
_REGISTRY_BY_HF = {m.hf_id: m for m in MODEL_REGISTRY}

BASELINE_RESULTS_DIR = os.path.join(DATA_DIR, "results", "baseline")


def resolve_model(name: str) -> ModelSpec:
    """Resolve a CLI model name (short or full HF id) to ModelSpec."""
    if name in _REGISTRY_BY_SHORT:
        return _REGISTRY_BY_SHORT[name]
    if name in _REGISTRY_BY_HF:
        return _REGISTRY_BY_HF[name]
    # Unknown model — treat as text-only by default
    short = name.split("/")[-1].lower().replace("-instruct", "").replace("_", "-")
    return ModelSpec(short, name, ["text_only"])


# ─── Scene loading ────────────────────────────────────────────────────────

def _collect_scenes(scenes_dir: str) -> list[dict]:
    """Load all *_gt.json files and pair with their PNG screenshots."""
    scenes = []
    for fname in sorted(os.listdir(scenes_dir)):
        if not fname.endswith("_gt.json"):
            continue
        json_path = os.path.join(scenes_dir, fname)
        png_name = fname.replace("_gt.json", ".png")
        png_path = os.path.join(scenes_dir, png_name)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenes.append({
            "json_path": json_path,
            "image_path": png_path if os.path.exists(png_path) else None,
            "data": data,
            "basename": fname.replace("_gt.json", ""),
        })
    return scenes


def _build_prompt_for_track(
    track: str,
    scene_data: dict,
    model_short_name: str = "",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given track.

    For LLaVA-1.6 models (4096-token context) we use a compact variant
    so that image tokens (~2.3k for AnyRes) + text still fit within ctx.
    All other models keep the full-length prompts to stay consistent
    with previously collected results.
    """
    room_type = scene_data.get("room_type", "unknown")
    agent_info = scene_data.get("agent", {"position": {"x": 0, "y": 0, "z": 0}})
    visible_objects = scene_data.get("visible_objects", [])

    use_short = model_short_name.startswith("llava")

    if track == "text_only":
        if use_short:
            return SAFETY_SYSTEM_PROMPT_SHORT, build_text_only_prompt_short(
                visible_objects, agent_info, room_type,
            )
        return SAFETY_SYSTEM_PROMPT, build_text_only_prompt(
            visible_objects, agent_info, room_type,
        )
    elif track == "image_only":
        if use_short:
            return IMAGE_ONLY_SYSTEM_PROMPT_SHORT, build_image_only_prompt(room_type)
        return IMAGE_ONLY_SYSTEM_PROMPT, build_image_only_prompt(room_type)
    elif track == "text_and_image":
        if use_short:
            return SAFETY_SYSTEM_PROMPT_SHORT, build_multimodal_prompt_short(
                visible_objects, agent_info, room_type,
            )
        return SAFETY_SYSTEM_PROMPT, build_multimodal_prompt(
            visible_objects, agent_info, room_type,
        )
    else:
        raise ValueError(f"Unknown track: {track}")


# ─── Main evaluation loop ────────────────────────────────────────────────

def run_baseline(
    model_specs: list[ModelSpec] | None = None,
    cache_dir: str = "/data/huggingface_cache",
    max_scenes: int | None = None,
    scenes_dir: str = SCENES_DIR,
    output_dir: str = BASELINE_RESULTS_DIR,
    override_tracks: list[str] | None = None,
):
    """Main entry: iterate models x tracks x scenes with VRAM safety."""
    if model_specs is None:
        model_specs = MODEL_REGISTRY

    scenes = _collect_scenes(scenes_dir)
    if max_scenes:
        scenes = scenes[:max_scenes]

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 64)
    print(f"SafeSight Baseline Evaluation v{BENCHMARK_VERSION}")
    print(f"Scenes         : {len(scenes)}")
    print(f"Models         : {', '.join(m.short_name for m in model_specs)}")
    print(f"Output dir     : {output_dir}")
    print("=" * 64)
    print_gpu_status("Initial GPU")

    for spec in model_specs:
        tracks = override_tracks if override_tracks else spec.tracks

        print(f"\n{'━' * 64}")
        print(f"  Model : {spec.short_name}  ({spec.hf_id})")
        print(f"  Tracks: {tracks}")
        print(f"{'━' * 64}")

        # --- VRAM pre-flight ---
        check = preflight_check(spec.hf_id)
        if not check["ok"]:
            print(f"  ✗ SKIPPING: {check['message']}")
            continue

        try:
            client = load_model(spec.hf_id, cache_dir)
        except Exception:
            print(f"  ✗ Failed to load {spec.hf_id}:")
            traceback.print_exc()
            continue

        for track in tracks:
            needs_image = track in ("image_only", "text_and_image")
            if needs_image and not client.supports_vision:
                print(f"  [SKIP] {track}: {spec.short_name} has no vision")
                continue

            # Folder uses short_name → e.g. "qwen2vl-7b_image_only"
            track_dir = os.path.join(output_dir, f"{spec.short_name}_{track}")
            os.makedirs(track_dir, exist_ok=True)

            ok, err, skip = 0, 0, 0
            print(f"\n  >> {track}  ({len(scenes)} scenes → {track_dir})")

            for i, scene in enumerate(scenes):
                basename = scene["basename"]
                out_path = os.path.join(track_dir, f"{basename}_result.json")

                if os.path.exists(out_path):
                    skip += 1
                    continue

                image_path = scene["image_path"] if needs_image else None
                if needs_image and image_path is None:
                    skip += 1
                    continue

                try:
                    sys_prompt, user_prompt = _build_prompt_for_track(
                        track, scene["data"],
                        model_short_name=spec.short_name,
                    )
                    raw_response, latency = client.generate(
                        system_prompt=sys_prompt,
                        user_text=user_prompt,
                        image_path=image_path,
                    )
                    parsed = parse_llm_response(raw_response)

                    result = {
                        "version": BENCHMARK_VERSION,
                        "scene_name": scene["data"].get("scene_name", ""),
                        "recipe_name": scene["data"].get("recipe_name", ""),
                        "room_type": scene["data"].get("room_type", ""),
                        "model": spec.hf_id,
                        "model_short": spec.short_name,
                        "track": track,
                        "noise_level": 0.0,
                        "ground_truth": scene["data"].get("ground_truth", {}),
                        "generation_config": {
                            "seed": EVAL_SEED,
                            **EVAL_GENERATE_KWARGS,
                        },
                        "llm_result": {
                            "raw_response": raw_response,
                            "parsed": parsed,
                            "track": track,
                            "num_objects_sent": len(
                                scene["data"].get("visible_objects", [])
                            ) if track != "image_only" else 0,
                            "latency_sec": round(latency, 2),
                        },
                    }

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    ok += 1
                    tag = parsed.get("assessment", "?")
                    if (i + 1) % 50 == 0 or (i + 1) == len(scenes):
                        print(
                            f"    [{i+1}/{len(scenes)}] "
                            f"ok={ok} err={err} skip={skip}  "
                            f"last: {tag} ({latency:.1f}s)"
                        )

                except torch.cuda.OutOfMemoryError:
                    err += 1
                    print(f"    [{i+1}] {basename} — OOM!")
                    print_gpu_status("OOM snapshot")
                    # Try to recover: clear cache and continue
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception:
                    err += 1
                    print(f"    [{i+1}] {basename} — ERROR")
                    traceback.print_exc()

            print(
                f"  -> {track}: {ok} ok / {err} err / {skip} skip "
                f"(total {ok + err + skip})"
            )

        client.unload()

    print(f"\n{'=' * 64}")
    print("All done.")
    print(f"Results: {output_dir}")
    print_gpu_status("Final GPU")
    print("=" * 64)


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SafeSight Baseline Evaluation (clean data, no noise)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Model short names or HF ids. "
            f"Available: {', '.join(m.short_name for m in MODEL_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--tracks", nargs="+",
        choices=["text_only", "image_only", "text_and_image"],
        default=None,
        help="Override tracks (default: auto per model)",
    )
    parser.add_argument(
        "--cache-dir", default="/data/huggingface_cache",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Limit scenes for quick testing",
    )
    parser.add_argument(
        "--scenes-dir", default=SCENES_DIR,
    )
    parser.add_argument(
        "--output-dir", default=BASELINE_RESULTS_DIR,
    )
    parser.add_argument(
        "--check-gpu", action="store_true",
        help="Only print GPU diagnostic, don't run evaluation",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all registered models and exit",
    )

    args = parser.parse_args()

    # --- GPU check mode ---
    if args.check_gpu:
        from benchmark.evaluate.gpu_utils import _cli as gpu_cli
        gpu_cli()
        return

    # --- List models mode ---
    if args.list_models:
        print("Registered models:")
        for m in MODEL_REGISTRY:
            vlm_tag = " [VLM]" if m.is_vlm else ""
            print(f"  {m.short_name:16s} → {m.hf_id}{vlm_tag}")
            print(f"  {'':16s}   tracks: {m.tracks}")
        return

    # --- Resolve models ---
    specs = None
    if args.models:
        specs = [resolve_model(m) for m in args.models]

    run_baseline(
        model_specs=specs,
        cache_dir=args.cache_dir,
        max_scenes=args.max_scenes,
        scenes_dir=args.scenes_dir,
        output_dir=args.output_dir,
        override_tracks=args.tracks,
    )


if __name__ == "__main__":
    main()
