"""
SafeSight — One-click full benchmark pipeline.

Runs all four steps in sequence:
  1. Harvest  — collect scene data from ai2thor (only if data/scenes/ is empty)
  2. Noise    — inject noise at multiple levels
  3. Evaluate — run LLM on noisy data
  4. Score    — compute scores and generate tables

Usage:
    # Full pipeline with Ollama (local testing)
    python run_benchmark.py --llm ollama --model llama3.2:3b

    # Full pipeline on GPU server
    python run_benchmark.py --llm huggingface --model google/gemma-4-31B-it \\
                            --cache-dir /data/huggingface_cache

    # Skip harvest (reuse existing data), only re-evaluate + score
    python run_benchmark.py --skip-harvest --skip-noise \\
                            --llm huggingface --model meta-llama/Llama-3.1-8B-Instruct

    # Harvest only (on a machine with display)
    python run_benchmark.py --only-harvest

    # Evaluate multiple models (run this script once per model)
    python run_benchmark.py --skip-harvest --skip-noise \\
                            --llm huggingface --model Qwen/Qwen2.5-7B-Instruct

Scene selection:
    --room-type kitchen          # only kitchen scenes
    --scenes FloorPlan1 FloorPlan2   # specific scenes
    (default: all 120 scenes across 4 room types)
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.config import (
    BENCHMARK_VERSION, NOISY_DIR, SCENES_DIR, ensure_dirs,
)


def step_harvest(args):
    """Step 1: Data collection from ai2thor."""
    from benchmark.harvest.harvest import run_harvest

    existing = glob.glob(os.path.join(SCENES_DIR, "*_gt.json"))
    if existing and not args.force_harvest:
        print(f"\n[Step 1] Harvest: SKIPPED ({len(existing)} scenes already exist)")
        print(f"  Use --force-harvest to re-collect")
        return

    print(f"\n{'=' * 60}")
    print(f"[Step 1] Harvesting scene data from ai2thor")
    print(f"{'=' * 60}")
    run_harvest(scenes=args.scenes, room_type=args.room_type)


def step_noise(args):
    """Step 2: Noise injection."""
    from benchmark.noise.noise_engine import run_noise_generation

    print(f"\n{'=' * 60}")
    print(f"[Step 2] Generating noisy variants")
    print(f"{'=' * 60}")
    levels = args.noise_levels if args.noise_levels else None
    run_noise_generation(noise_levels=levels)


def step_evaluate(args):
    """Step 3: LLM evaluation."""
    from benchmark.evaluate.evaluator import run_evaluation

    print(f"\n{'=' * 60}")
    print(f"[Step 3] Evaluating with LLM: {args.model}")
    print(f"{'=' * 60}")
    run_evaluation(
        llm_type=args.llm,
        model=args.model,
        track=args.track,
        noise_levels=args.noise_levels,
        llm_host=args.llm_host,
        cache_dir=args.cache_dir,
        api_key=args.api_key,
        max_scenes=args.max_scenes,
    )


def step_score(args):
    """Step 4: Auto-scoring."""
    from benchmark.score.scorer import run_scoring

    print(f"\n{'=' * 60}")
    print(f"[Step 4] Scoring results")
    print(f"{'=' * 60}")
    run_scoring()


def main():
    parser = argparse.ArgumentParser(
        description=f"SafeSight v{BENCHMARK_VERSION} — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    step_group = parser.add_argument_group("Step control")
    step_group.add_argument("--only-harvest", action="store_true",
                            help="Only run step 1 (harvest)")
    step_group.add_argument("--only-noise", action="store_true",
                            help="Only run step 2 (noise)")
    step_group.add_argument("--only-evaluate", action="store_true",
                            help="Only run step 3 (evaluate)")
    step_group.add_argument("--only-score", action="store_true",
                            help="Only run step 4 (score)")
    step_group.add_argument("--skip-harvest", action="store_true",
                            help="Skip step 1")
    step_group.add_argument("--skip-noise", action="store_true",
                            help="Skip step 2")
    step_group.add_argument("--skip-evaluate", action="store_true",
                            help="Skip step 3")
    step_group.add_argument("--skip-score", action="store_true",
                            help="Skip step 4")
    step_group.add_argument("--force-harvest", action="store_true",
                            help="Re-harvest even if data exists")

    scene_group = parser.add_argument_group("Scene selection")
    scene_group.add_argument("--room-type",
                             choices=["kitchen", "living_room", "bedroom", "bathroom"])
    scene_group.add_argument("--scenes", nargs="+")

    llm_group = parser.add_argument_group("LLM configuration")
    llm_group.add_argument("--llm", choices=["ollama", "openai", "huggingface"],
                           default="ollama")
    llm_group.add_argument("--model", default="llama3.2:3b")
    llm_group.add_argument("--track", choices=["text_only", "multimodal"],
                           default="text_only")
    llm_group.add_argument("--llm-host", default="http://localhost:11434")
    llm_group.add_argument("--cache-dir", default=None)
    llm_group.add_argument("--api-key", default="not-needed")

    noise_group = parser.add_argument_group("Noise configuration")
    noise_group.add_argument("--noise-levels", nargs="+", type=float)

    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument("--max-scenes", type=int, default=None,
                            help="Limit scenes for quick testing")

    args = parser.parse_args()
    ensure_dirs()

    print(f"{'=' * 60}")
    print(f"  SafeSight v{BENCHMARK_VERSION}")
    print(f"  Embodied AI Safety Benchmark")
    print(f"{'=' * 60}")

    only_flags = [args.only_harvest, args.only_noise,
                  args.only_evaluate, args.only_score]
    run_specific = any(only_flags)

    if run_specific:
        if args.only_harvest:
            step_harvest(args)
        if args.only_noise:
            step_noise(args)
        if args.only_evaluate:
            step_evaluate(args)
        if args.only_score:
            step_score(args)
    else:
        if not args.skip_harvest:
            step_harvest(args)
        if not args.skip_noise:
            step_noise(args)
        if not args.skip_evaluate:
            step_evaluate(args)
        if not args.skip_score:
            step_score(args)

    print(f"\n{'=' * 60}")
    print("Pipeline complete.")
    print(f"  Scenes  : {SCENES_DIR}")
    print(f"  Noisy   : {NOISY_DIR}")
    print(f"  Results : {os.path.join('data', 'results')}")
    print(f"  Scores  : {os.path.join('data', 'scores')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
