#!/usr/bin/env python3
"""
SafeSight — Noisy Evaluation (perturbed sensor data).

Reads noisy scene JSONs from data/noisy/ (produced by
benchmark.noise.noise_engine), runs LLM/VLM inference in three tracks,
and saves results to a SEPARATE folder so they are never mixed with
the clean baseline runs.

I/O layout
----------
Input  noisy JSONs : <DATA_DIR>/noisy/<scene_id>_noise<L>_seed<S>.json
Input  PNGs        : <DATA_DIR>/scenes/<scene_id>.png   (shared with baseline)
Output results     : <DATA_DIR>/results/noisy/<model>_<track>/
                     └── <scene_id>_noise<L>_seed<S>_result.json

Tracks
------
  text_only      — perturbed sensor text only       (any model)
  image_only     — clean screenshot only            (VLM only, noise = 0 by definition)
  text_and_image — perturbed sensor + clean image   (VLM only)

NOTE: image_only carries the same input as baseline (image is never noised).
We still run it under noisy_eval so that downstream scoring keeps the
"per-noise-level" structure tidy and consistent across tracks; if you
do not need it, pass `--tracks text_only text_and_image`.

Usage
-----
  # Smoke test — 1 model, 5 noisy files
  python -m benchmark.evaluate.noisy_eval --models qwen2vl-7b --max-scenes 5

  # Single noise level (e.g. only 0.3) for all 14 models
  python -m benchmark.evaluate.noisy_eval --noise-levels 0.3

  # Seed subset, single model
  python -m benchmark.evaluate.noisy_eval --models llava1.6-7b \
      --noise-levels 0.1 0.3 0.5 --seeds 100 300 500

  # Use noise-aware system prompt (warn the model that data may be wrong)
  python -m benchmark.evaluate.noisy_eval --noise-aware \
      --models qwen2.5-7b --noise-levels 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from typing import Iterable

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from benchmark.config import (
    BENCHMARK_VERSION,
    DATA_DIR,
    NOISE_LEVELS,
    NOISY_DIR,
    SCENES_DIR,
)
from benchmark.evaluate.baseline_eval import (
    MODEL_REGISTRY,
    ModelSpec,
    resolve_model,
)
from benchmark.evaluate.gpu_utils import preflight_check, print_gpu_status
from benchmark.evaluate.prompts import (
    IMAGE_ONLY_SYSTEM_PROMPT,
    IMAGE_ONLY_SYSTEM_PROMPT_SHORT,
    NOISY_MULTIMODAL_SYSTEM_PROMPT,
    NOISY_SYSTEM_PROMPT,
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


# ─── Output dir (kept separate from baseline) ─────────────────────────────

NOISY_RESULTS_DIR = os.path.join(DATA_DIR, "results", "noisy")
NOISY_AWARE_RESULTS_DIR = os.path.join(DATA_DIR, "results", "noisy_aware")


# ─── Noisy file discovery ────────────────────────────────────────────────

# Matches "<scene_id>_noise<level>_seed<seed>.json"
_NOISY_NAME_RE = re.compile(
    r"^(?P<scene_id>.+?)_noise(?P<level>\d+(?:\.\d+)?)_seed(?P<seed>\d+)\.json$"
)


def _collect_noisy_scenes(
    noisy_dir: str,
    scenes_dir: str,
    noise_levels: list[float] | None = None,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Walk the noisy directory and pair each noisy JSON with its PNG.

    Returns a list of dicts with keys:
        json_path, image_path, data, basename, scene_id, noise_level, noise_seed
    """
    if not os.path.isdir(noisy_dir):
        raise FileNotFoundError(
            f"Noisy directory not found: {noisy_dir}\n"
            f"Run `python -m benchmark.noise.noise_engine` first."
        )

    level_filter = {round(float(x), 1) for x in noise_levels} if noise_levels else None
    seed_filter = set(seeds) if seeds else None

    out: list[dict] = []
    for fname in sorted(os.listdir(noisy_dir)):
        if not fname.endswith(".json"):
            continue
        m = _NOISY_NAME_RE.match(fname)
        if not m:
            continue

        level = round(float(m.group("level")), 1)
        seed = int(m.group("seed"))
        scene_id = m.group("scene_id")

        if level_filter is not None and level not in level_filter:
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue

        json_path = os.path.join(noisy_dir, fname)
        png_path = os.path.join(scenes_dir, f"{scene_id}.png")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out.append({
            "json_path": json_path,
            "image_path": png_path if os.path.exists(png_path) else None,
            "data": data,
            "basename": fname[:-len(".json")],   # "<scene_id>_noise<L>_seed<S>"
            "scene_id": scene_id,
            "noise_level": level,
            "noise_seed": seed,
        })
    return out


# ─── Prompt builder (noise-aware aware) ───────────────────────────────────

def _build_prompt_for_track_noisy(
    track: str,
    scene_data: dict,
    model_short_name: str,
    noise_aware: bool,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt).

    If `noise_aware` and the model is NOT a tight-context LLaVA, swap in the
    NOISY_*_SYSTEM_PROMPT variants which warn the model that sensor data may
    be wrong / missing / corrupted.

    LLaVA-1.6 (4096 ctx) keeps the SHORT prompts even under --noise-aware
    because the noise-awareness block adds ~500 tokens that would push the
    image-token + prompt total over the context window.
    """
    room_type = scene_data.get("room_type", "unknown")
    agent_info = scene_data.get("agent", {"position": {"x": 0, "y": 0, "z": 0}})
    visible_objects = scene_data.get("visible_objects", [])

    use_short = model_short_name.startswith("llava")  # tight 4k ctx

    # ── text_only ───────────────────────────────────────────────────────
    if track == "text_only":
        if use_short:
            sys_prompt = SAFETY_SYSTEM_PROMPT_SHORT
            user_prompt = build_text_only_prompt_short(
                visible_objects, agent_info, room_type,
            )
        else:
            sys_prompt = NOISY_SYSTEM_PROMPT if noise_aware else SAFETY_SYSTEM_PROMPT
            user_prompt = build_text_only_prompt(
                visible_objects, agent_info, room_type,
            )
        return sys_prompt, user_prompt

    # ── image_only ──────────────────────────────────────────────────────
    if track == "image_only":
        # image is unaltered, so we never use the noise-aware prompt here.
        sys_prompt = (
            IMAGE_ONLY_SYSTEM_PROMPT_SHORT if use_short else IMAGE_ONLY_SYSTEM_PROMPT
        )
        return sys_prompt, build_image_only_prompt(room_type)

    # ── text_and_image ──────────────────────────────────────────────────
    if track == "text_and_image":
        if use_short:
            sys_prompt = SAFETY_SYSTEM_PROMPT_SHORT
            user_prompt = build_multimodal_prompt_short(
                visible_objects, agent_info, room_type,
            )
        else:
            sys_prompt = (
                NOISY_MULTIMODAL_SYSTEM_PROMPT
                if noise_aware
                else SAFETY_SYSTEM_PROMPT
            )
            user_prompt = build_multimodal_prompt(
                visible_objects, agent_info, room_type,
            )
        return sys_prompt, user_prompt

    raise ValueError(f"Unknown track: {track}")


# ─── Main loop ────────────────────────────────────────────────────────────

def run_noisy(
    model_specs: list[ModelSpec] | None = None,
    cache_dir: str = "/data/huggingface_cache",
    noisy_dir: str = NOISY_DIR,
    scenes_dir: str = SCENES_DIR,
    output_dir: str | None = None,
    noise_levels: list[float] | None = None,
    seeds: list[int] | None = None,
    max_scenes: int | None = None,
    override_tracks: list[str] | None = None,
    noise_aware: bool = False,
):
    """Iterate models × tracks × noisy files."""
    if model_specs is None:
        model_specs = MODEL_REGISTRY

    if output_dir is None:
        output_dir = NOISY_AWARE_RESULTS_DIR if noise_aware else NOISY_RESULTS_DIR

    scenes = _collect_noisy_scenes(
        noisy_dir=noisy_dir,
        scenes_dir=scenes_dir,
        noise_levels=noise_levels,
        seeds=seeds,
    )

    if not scenes:
        print(f"No noisy scenes found in {noisy_dir}")
        print("Generate them first:")
        print("  python -m benchmark.noise.noise_engine")
        return

    if max_scenes:
        scenes = scenes[:max_scenes]

    os.makedirs(output_dir, exist_ok=True)

    levels_seen = sorted({s["noise_level"] for s in scenes})
    seeds_seen = sorted({s["noise_seed"] for s in scenes})

    print("=" * 64)
    print(f"SafeSight Noisy Evaluation v{BENCHMARK_VERSION}")
    print(f"Noisy dir      : {noisy_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Scenes         : {len(scenes)}")
    print(f"Noise levels   : {levels_seen}")
    print(f"Seeds          : {seeds_seen}")
    print(f"Models         : {', '.join(m.short_name for m in model_specs)}")
    print(f"Noise-aware    : {noise_aware}")
    print("=" * 64)
    print_gpu_status("Initial GPU")

    for spec in model_specs:
        tracks = override_tracks if override_tracks else spec.tracks

        print(f"\n{'━' * 64}")
        print(f"  Model : {spec.short_name}  ({spec.hf_id})")
        print(f"  Tracks: {tracks}")
        print(f"{'━' * 64}")

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

            track_dir = os.path.join(output_dir, f"{spec.short_name}_{track}")
            os.makedirs(track_dir, exist_ok=True)

            ok, err, skip = 0, 0, 0
            print(f"\n  >> {track}  ({len(scenes)} files → {track_dir})")

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
                    sys_prompt, user_prompt = _build_prompt_for_track_noisy(
                        track,
                        scene["data"],
                        model_short_name=spec.short_name,
                        noise_aware=noise_aware,
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
                        "noise_level": scene["noise_level"],
                        "noise_seed": scene["noise_seed"],
                        "noise_aware_prompt": noise_aware,
                        "noise_meta": scene["data"].get("_noise_meta", {}),
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
                    if (i + 1) % 100 == 0 or (i + 1) == len(scenes):
                        print(
                            f"    [{i+1}/{len(scenes)}] "
                            f"ok={ok} err={err} skip={skip}  "
                            f"last(L={scene['noise_level']}): {tag} ({latency:.1f}s)"
                        )

                except torch.cuda.OutOfMemoryError:
                    err += 1
                    print(f"    [{i+1}] {basename} — OOM!")
                    print_gpu_status("OOM snapshot")
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

def _parse_levels(raw: Iterable[str] | None) -> list[float] | None:
    if not raw:
        return None
    return [round(float(x), 1) for x in raw]


def main():
    parser = argparse.ArgumentParser(
        description="SafeSight Noisy Evaluation (perturbed data)",
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
        "--noise-levels", nargs="+", type=float, default=None,
        help=f"Subset of noise levels. Available: {NOISE_LEVELS}",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Subset of seeds (e.g. 100 101 102 for level=0.1).",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Limit total noisy files for quick testing.",
    )
    parser.add_argument(
        "--noisy-dir", default=NOISY_DIR,
        help="Directory of noisy JSONs (default: data/noisy/).",
    )
    parser.add_argument(
        "--scenes-dir", default=SCENES_DIR,
        help="Directory of clean PNG screenshots (default: data/scenes/).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Output dir. Default depends on --noise-aware: "
            f"{NOISY_RESULTS_DIR} (standard) or {NOISY_AWARE_RESULTS_DIR} "
            "(noise-aware)."
        ),
    )
    parser.add_argument(
        "--noise-aware", action="store_true",
        help=(
            "Use NOISY_SYSTEM_PROMPT (tells the model that data may be wrong). "
            "Saved to a separate output dir to allow apples-to-apples comparison."
        ),
    )
    parser.add_argument(
        "--cache-dir", default="/data/huggingface_cache",
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--check-gpu", action="store_true",
        help="Only print GPU diagnostic, don't run evaluation.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all registered models and exit.",
    )

    args = parser.parse_args()

    if args.check_gpu:
        from benchmark.evaluate.gpu_utils import _cli as gpu_cli
        gpu_cli()
        return

    if args.list_models:
        print("Registered models:")
        for m in MODEL_REGISTRY:
            vlm_tag = " [VLM]" if m.is_vlm else ""
            print(f"  {m.short_name:20s} → {m.hf_id}{vlm_tag}")
            print(f"  {'':20s}   tracks: {m.tracks}")
        return

    specs = None
    if args.models:
        specs = [resolve_model(m) for m in args.models]

    run_noisy(
        model_specs=specs,
        cache_dir=args.cache_dir,
        noisy_dir=args.noisy_dir,
        scenes_dir=args.scenes_dir,
        output_dir=args.output_dir,
        noise_levels=_parse_levels(args.noise_levels),
        seeds=args.seeds,
        max_scenes=args.max_scenes,
        override_tracks=args.tracks,
        noise_aware=args.noise_aware,
    )


if __name__ == "__main__":
    main()
