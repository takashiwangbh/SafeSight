"""Strategy 2 — run the cognitive self-correction second pass (needs GPU).

Reads the offline manifest (select_fp_samples.py), loads each model ONCE,
and for every sampled scene:
    1. rebuilds the ORIGINAL first-pass user prompt from the scene JSON
       (identical to benchmark.evaluate.baseline_eval), so the auditor sees
       the same evidence the first pass saw;
    2. builds the self-correction prompt that feeds the first verdict back;
    3. re-runs inference (re-attaching the SAME image for text+image track);
    4. parses + stores the second-pass verdict next to the first one.

Incremental & restart-friendly: existing per-sample outputs are skipped.

Usage (run from the repository root; GPU required):
    python -m supplementary_experiments.mitigation.run_self_correction
    python -m supplementary_experiments.mitigation.run_self_correction --models qwen2vl-7b
    python -m supplementary_experiments.mitigation.run_self_correction --max-per-model 10   # smoke
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
    MITIGATION_DIR, load_scene,
)
from supplementary_experiments.mitigation.mitigation_baseline import (  # noqa: E402
    SELF_CORRECTION_SYSTEM_PROMPT, generate_self_correction_prompt,
)

MANIFEST_PATH = os.path.join(MITIGATION_DIR, "sample_manifest.json")
SC_OUT_DIR = os.path.join(MITIGATION_DIR, "self_correction")


def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run: python -m supplementary_experiments.mitigation.select_fp_samples"
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_pass_parsed(sample: dict) -> dict:
    """Pull the first-pass parsed verdict from the original result JSON."""
    from supplementary_experiments.common import load_result, parsed_of
    res = load_result(sample["result_path"])
    return parsed_of(res) if res else {}


def run_for_model(model: str, samples: list[dict], cache_dir: str,
                  max_per_model: int | None):
    # Lazy imports so the offline scripts never need torch.
    import torch  # noqa: F401
    from benchmark.evaluate.vlm_client import load_model
    from benchmark.evaluate.baseline_eval import _build_prompt_for_track
    from benchmark.evaluate.gpu_utils import preflight_check
    from llm_client import parse_llm_response

    my = [s for s in samples if s["model"] == model]
    if max_per_model:
        my = my[:max_per_model]
    if not my:
        print(f"  (no samples for {model})")
        return

    track = my[0]["track"]
    hf_id = my[0]["hf_id"]
    out_dir = os.path.join(SC_OUT_DIR, f"{model}_{track}")
    os.makedirs(out_dir, exist_ok=True)

    print("─" * 64)
    print(f"  Model : {model}  ({hf_id})  track={track}")
    print(f"  Samples: {len(my)}   Output: {out_dir}")
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

    ok = err = skip = 0
    t0 = time.time()
    for i, s in enumerate(my):
        tag = f"{s['scenario']}__{s['basename']}"
        out_path = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(out_path):
            skip += 1
            continue

        try:
            scene = load_scene(s["scene_gt_path"])
            if scene is None:
                err += 1
                continue

            # 1. rebuild the original first-pass user prompt (same evidence)
            _sys_unused, original_user = _build_prompt_for_track(
                track, scene, model_short_name=model,
            )
            # 2. first-pass verdict + self-correction prompt
            first = _first_pass_parsed(s)
            sc_user = generate_self_correction_prompt(original_user, first)

            # 3. re-run (re-attach image for multimodal track)
            image_path = s.get("image_path") if track in (
                "image_only", "text_and_image") else None
            raw, latency = client.generate(
                system_prompt=SELF_CORRECTION_SYSTEM_PROMPT,
                user_text=sc_user,
                image_path=image_path,
            )
            second = parse_llm_response(raw)

            record = {
                "model":              model,
                "hf_id":              hf_id,
                "track":              track,
                "scenario":           s["scenario"],
                "basename":           s["basename"],
                "room_type":          s["room_type"],
                "gt_is_safe":         s["gt_is_safe"],
                "original_confusion": s["confusion"],
                "first_pass_parsed":  first,
                "second_pass_raw":    raw,
                "second_pass_parsed": second,
                "latency_sec":        round(latency, 2),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            ok += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(my):
                a = second.get("assessment", "?")
                print(f"    [{i+1}/{len(my)}] ok={ok} err={err} skip={skip} "
                      f"last({s['confusion']}→{a}) {latency:.1f}s")
        except Exception:
            err += 1
            print(f"    [{i+1}] {tag} — ERROR")
            traceback.print_exc()

    dt = time.time() - t0
    print(f"  -> {model}: {ok} ok / {err} err / {skip} skip ({dt:.0f}s)")
    client.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of manifest models (default: all)")
    p.add_argument("--cache-dir", default="/data/huggingface_cache")
    p.add_argument("--max-per-model", type=int, default=None,
                   help="cap samples per model (smoke test)")
    args = p.parse_args()

    payload = _load_manifest()
    samples = payload["samples"]
    models = args.models or payload["models"]

    os.makedirs(SC_OUT_DIR, exist_ok=True)
    print(f"Self-correction over {len(samples)} manifest samples; "
          f"models={models}")
    for model in models:
        run_for_model(model, samples, args.cache_dir, args.max_per_model)

    print("\nDone. Next: python -m supplementary_experiments.mitigation.score_mitigation")


if __name__ == "__main__":
    main()
