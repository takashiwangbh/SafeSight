"""
Step 3 — Two-Track Evaluator.

Reads noisy scene JSONs, assembles prompts, feeds them to LLMs,
and saves structured results.

Track A (text-only):  Llama / Qwen / Gemma  — text sensor data only
Track B (multimodal): LLaVA / Qwen-VL      — image + text sensor data

Usage:
    python -m benchmark.evaluate.evaluator --llm ollama --model llama3.2:3b
    python -m benchmark.evaluate.evaluator --llm huggingface --model google/gemma-4-31B-it \\
                                           --cache-dir /data/huggingface_cache
    python -m benchmark.evaluate.evaluator --track text_only --noise-levels 0.0 0.3
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.config import (
    BENCHMARK_VERSION, NOISY_DIR, RESULTS_DIR, ensure_dirs,
)
from benchmark.evaluate.prompts import (
    NOISY_MULTIMODAL_SYSTEM_PROMPT,
    NOISY_SYSTEM_PROMPT,
    SAFETY_SYSTEM_PROMPT,
    build_multimodal_prompt,
    build_text_only_prompt,
)
from llm_client import (
    BaseLLMClient,
    HuggingFaceClient,
    OllamaClient,
    OpenAICompatibleClient,
    parse_llm_response,
)


def _pick_system_prompt(track: str, noise_aware: bool) -> str:
    """Select the right system prompt based on track and noise-awareness."""
    if not noise_aware:
        return SAFETY_SYSTEM_PROMPT
    if track == "multimodal":
        return NOISY_MULTIMODAL_SYSTEM_PROMPT
    return NOISY_SYSTEM_PROMPT


def build_llm_client(args, system_prompt: str) -> BaseLLMClient:
    """Construct the appropriate LLM client from CLI args."""
    if args.llm == "ollama":
        client = OllamaClient(
            model_name=args.model,
            host=args.llm_host,
            system_prompt=system_prompt,
        )
    elif args.llm == "huggingface":
        client = HuggingFaceClient(
            model_name=args.model,
            cache_dir=args.cache_dir,
            system_prompt=system_prompt,
        )
    else:
        client = OpenAICompatibleClient(
            model_name=args.model,
            base_url=args.llm_host,
            api_key=args.api_key,
            system_prompt=system_prompt,
        )
    return client


def evaluate_single(llm: BaseLLMClient, noisy_data: dict,
                    track: str = "text_only") -> dict:
    """
    Run one evaluation: build prompt → call LLM → parse response.

    Only visible objects are sent to the LLM (matching real-robot perception).
    The key is "visible_objects" (new harvest format) with "objects" as fallback.
    """
    room_type = noisy_data.get("room_type", "unknown")
    agent_info = noisy_data.get("agent", {"position": {"x": 0, "y": 0, "z": 0}})

    visible_objects = noisy_data.get("visible_objects")
    if visible_objects is None:
        all_objects = noisy_data.get("objects", [])
        visible_objects = [o for o in all_objects if o.get("visible", False)]

    if track == "text_only":
        user_prompt = build_text_only_prompt(visible_objects, agent_info, room_type)
    else:
        user_prompt = build_multimodal_prompt(visible_objects, agent_info, room_type)

    llm.reset_history()

    t0 = time.time()
    parsed, raw_response = llm.chat(user_prompt, "")
    elapsed = time.time() - t0

    return {
        "raw_response": raw_response,
        "parsed": parsed,
        "track": track,
        "num_objects_sent": len(visible_objects),
        "latency_sec": round(elapsed, 2),
    }


def run_evaluation(
    llm_type: str = "ollama",
    model: str = "llama3.2:3b",
    track: str = "text_only",
    noise_levels: list[float] | None = None,
    llm_host: str = "http://localhost:11434",
    cache_dir: str | None = None,
    api_key: str = "not-needed",
    max_scenes: int | None = None,
    noise_aware: bool = False,
):
    """Main entry point for the evaluator.

    Args:
        noise_aware: If True, use noise-aware system prompt that warns the
            LLM about potential sensor errors. This is an experimental
            condition — compare results with noise_aware=False to measure
            whether awareness of noise helps LLM performance.
    """
    ensure_dirs()

    noisy_files = sorted(glob.glob(os.path.join(NOISY_DIR, "*.json")))
    if not noisy_files:
        print(f"No noisy files found in {NOISY_DIR}.")
        print("Run the noise step first: python -m benchmark.noise.noise_engine")
        return

    if noise_levels:
        level_strs = [f"noise{nl:.1f}" for nl in noise_levels]
        noisy_files = [f for f in noisy_files
                       if any(ls in os.path.basename(f) for ls in level_strs)]

    if max_scenes:
        noisy_files = noisy_files[:max_scenes]

    model_safe = model.replace("/", "_").replace(":", "_")
    suffix = f"_{track}_noiseaware" if noise_aware else f"_{track}"
    model_results_dir = os.path.join(RESULTS_DIR, f"{model_safe}{suffix}")
    os.makedirs(model_results_dir, exist_ok=True)

    prompt_mode = "noise-aware" if noise_aware else "standard"
    system_prompt = _pick_system_prompt(track, noise_aware)

    print(f"SafeSight Evaluator v{BENCHMARK_VERSION}")
    print(f"Model       : {model}")
    print(f"Backend     : {llm_type}")
    print(f"Track       : {track}")
    print(f"Prompt mode : {prompt_mode}")
    print(f"Scenes      : {len(noisy_files)}")
    print(f"Output dir  : {model_results_dir}")
    print("-" * 60)

    args_ns = argparse.Namespace(
        llm=llm_type, model=model, llm_host=llm_host,
        cache_dir=cache_dir, api_key=api_key,
    )
    llm = build_llm_client(args_ns, system_prompt)

    success_count = 0
    error_count = 0

    for i, noisy_path in enumerate(noisy_files):
        basename = os.path.basename(noisy_path).replace(".json", "")
        out_path = os.path.join(model_results_dir, f"{basename}_result.json")

        if os.path.exists(out_path):
            print(f"  [{i + 1}/{len(noisy_files)}] {basename} — skipped (exists)")
            success_count += 1
            continue

        with open(noisy_path, "r", encoding="utf-8") as f:
            noisy_data = json.load(f)

        try:
            result = evaluate_single(llm, noisy_data, track=track)

            output = {
                "version": BENCHMARK_VERSION,
                "scene_name": noisy_data.get("scene_name", ""),
                "recipe_name": noisy_data.get("recipe_name", ""),
                "room_type": noisy_data.get("room_type", ""),
                "noise_level": noisy_data.get("_noise_meta", {}).get("noise_level", 0),
                "noise_seed": noisy_data.get("_noise_meta", {}).get("seed", 0),
                "model": model,
                "track": track,
                "ground_truth": noisy_data.get("ground_truth", {}),
                "llm_result": result,
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            status = "OK"
            success_count += 1
        except Exception:
            status = "ERROR"
            error_count += 1
            print(f"    Error: {traceback.format_exc()}")

        print(f"  [{i + 1}/{len(noisy_files)}] {basename} — {status}"
              f" ({result.get('latency_sec', '?')}s)" if status == "OK" else "")

    print(f"\nDone. {success_count} succeeded, {error_count} errors.")
    print(f"Results in: {model_results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeSight LLM Evaluator")
    parser.add_argument("--llm", choices=["ollama", "openai", "huggingface"],
                        default="ollama")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--track", choices=["text_only", "multimodal"],
                        default="text_only")
    parser.add_argument("--noise-levels", nargs="+", type=float)
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes (for testing)")
    parser.add_argument("--noise-aware", action="store_true",
                        help="Use noise-aware system prompt (warns LLM about sensor errors)")

    args = parser.parse_args()
    run_evaluation(
        llm_type=args.llm, model=args.model, track=args.track,
        noise_levels=args.noise_levels, llm_host=args.llm_host,
        cache_dir=args.cache_dir, api_key=args.api_key,
        max_scenes=args.max_scenes,
        noise_aware=args.noise_aware,
    )
