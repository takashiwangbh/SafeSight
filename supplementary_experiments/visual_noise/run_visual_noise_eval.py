"""Corrupt selected renders + re-run image-track inference (needs GPU).

For every selected hazard scene we evaluate the image_only track under:
    · a CLEAN control (original render, re-run in this same session), and
    · each (noise_type, intensity) RGB corruption.

The clean control is re-run here (rather than reused from the main results)
so that the clean-vs-corrupted comparison is fully matched: same model load,
same generation config, same session — isolating the effect of the
corruption itself.

Corrupted frames are cached under
    data/supplementary/visual_noise/corrupted/<type>_<intensity>/<basename>.png
and per-condition predictions under
    data/supplementary/visual_noise/results/<model>/<basename>__<cond>.json

Incremental & restart-friendly.

Usage:
    python -m supplementary_experiments.visual_noise.run_visual_noise_eval
    python -m supplementary_experiments.visual_noise.run_visual_noise_eval \
        --types motion_blur low_illumination --intensities 0.3 0.7
    python -m supplementary_experiments.visual_noise.run_visual_noise_eval --max-scenes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from supplementary_experiments.common import (  # noqa: E402
    VISUAL_NOISE_DIR, VISUAL_NOISE_TRACK, load_scene,
)
from supplementary_experiments.visual_noise.visual_noise_injector import (  # noqa: E402
    NOISE_TYPES, inject_physical_sensor_noise,
)

MANIFEST_PATH = os.path.join(VISUAL_NOISE_DIR, "sample_manifest.json")
CORRUPT_DIR = os.path.join(VISUAL_NOISE_DIR, "corrupted")
RESULTS_DIR = os.path.join(VISUAL_NOISE_DIR, "results")

DEFAULT_TYPES = ["motion_blur", "low_illumination", "gaussian_sensor"]
DEFAULT_INTENSITIES = [0.3, 0.7]
GAUSS_SEED = 0  # fixed so corrupted frames are reproducible


def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run: python -m supplementary_experiments.visual_noise.select_visual_samples"
        )
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _conditions(types: list[str], intensities: list[float]):
    """Yield (cond_name, noise_type|None, intensity) including clean control."""
    yield ("clean", None, 0.0)
    for t in types:
        for inten in intensities:
            yield (f"{t}_{inten}", t, inten)


def _corrupted_path(noise_type: str, intensity: float, basename: str) -> str:
    return os.path.join(CORRUPT_DIR, f"{noise_type}_{intensity}", f"{basename}.png")


def _ensure_corrupted(sample: dict, noise_type: str, intensity: float) -> str:
    """Create (cache) and return the corrupted frame path."""
    out = _corrupted_path(noise_type, intensity, sample["basename"])
    if not os.path.exists(out):
        inject_physical_sensor_noise(
            sample["png_path"], noise_type=noise_type,
            intensity=intensity, seed=GAUSS_SEED, out_path=out,
        )
    return out


def run_for_model(model: str, hf_id: str, samples: list[dict],
                  conditions: list, cache_dir: str):
    import torch  # noqa: F401
    from benchmark.evaluate.vlm_client import load_model
    from benchmark.evaluate.baseline_eval import _build_prompt_for_track
    from benchmark.evaluate.gpu_utils import preflight_check
    from llm_client import parse_llm_response

    out_dir = os.path.join(RESULTS_DIR, model)
    os.makedirs(out_dir, exist_ok=True)

    print("─" * 64)
    print(f"  Model : {model}  ({hf_id})  track={VISUAL_NOISE_TRACK}")
    print(f"  Scenes: {len(samples)}  Conditions: {[c[0] for c in conditions]}")
    print("─" * 64)

    check = preflight_check(hf_id)
    print(f"  [preflight] {check['message']}")
    if not check["ok"]:
        print(f"  ✗ SKIP {model}: insufficient VRAM")
        return

    try:
        client = load_model(hf_id, cache_dir)
    except Exception:
        print(f"  ✗ Failed to load {hf_id}:")
        traceback.print_exc()
        return
    if not client.supports_vision:
        print(f"  ✗ {model} has no vision support; skipping")
        client.unload()
        return

    ok = err = skip = 0
    t0 = time.time()
    total = len(samples) * len(conditions)
    done = 0
    for s in samples:
        scene = load_scene(s["gt_path"])
        if scene is None:
            err += len(conditions)
            continue
        sys_p, user_p = _build_prompt_for_track(
            VISUAL_NOISE_TRACK, scene, model_short_name=model,
        )
        for cond_name, noise_type, intensity in conditions:
            done += 1
            out_path = os.path.join(out_dir, f"{s['basename']}__{cond_name}.json")
            if os.path.exists(out_path):
                skip += 1
                continue
            try:
                if noise_type is None:
                    image_path = s["png_path"]            # clean control
                else:
                    image_path = _ensure_corrupted(s, noise_type, intensity)

                raw, latency = client.generate(
                    system_prompt=sys_p, user_text=user_p, image_path=image_path,
                )
                parsed = parse_llm_response(raw)
                record = {
                    "model":         model,
                    "hf_id":         hf_id,
                    "track":         VISUAL_NOISE_TRACK,
                    "basename":      s["basename"],
                    "room_type":     s["room_type"],
                    "severity":      s["severity"],
                    "danger_labels": s["danger_labels"],
                    "condition":     cond_name,
                    "noise_type":    noise_type or "clean",
                    "intensity":     intensity,
                    "image_path":    image_path,
                    "raw_response":  raw,
                    "parsed":        parsed,
                    "latency_sec":   round(latency, 2),
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                ok += 1
                if done % 50 == 0 or done == total:
                    a = parsed.get("assessment", "?")
                    print(f"    [{done}/{total}] ok={ok} err={err} skip={skip} "
                          f"last({s['basename'][:24]}|{cond_name}→{a}) {latency:.1f}s")
            except Exception:
                err += 1
                print(f"    {s['basename']}|{cond_name} — ERROR")
                traceback.print_exc()

    dt = time.time() - t0
    print(f"  -> {model}: {ok} ok / {err} err / {skip} skip ({dt:.0f}s)")
    client.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--types", nargs="+", default=DEFAULT_TYPES, choices=NOISE_TYPES)
    p.add_argument("--intensities", nargs="+", type=float, default=DEFAULT_INTENSITIES)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--cache-dir", default="/data/huggingface_cache")
    p.add_argument("--max-scenes", type=int, default=None, help="smoke test cap")
    args = p.parse_args()

    payload = _load_manifest()
    samples = payload["samples"]
    if args.max_scenes:
        samples = samples[:args.max_scenes]
    models = args.models or payload["models"]
    hf_ids = payload["hf_ids"]
    conditions = list(_conditions(args.types, args.intensities))

    os.makedirs(CORRUPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Visual-noise eval: {len(samples)} scenes × {len(conditions)} conditions "
          f"× {len(models)} models")
    for model in models:
        run_for_model(model, hf_ids[model], samples, conditions, args.cache_dir)

    print("\nDone. Next: python -m supplementary_experiments.visual_noise.score_visual_noise")


if __name__ == "__main__":
    main()
