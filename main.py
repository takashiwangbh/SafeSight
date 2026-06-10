"""
ai2thor + LLM interactive control system.

Usage:
    # Ollama (local, lightweight)
    python main.py --llm ollama --model llama3

    # HuggingFace Transformers (lab GPU server)
    python main.py --llm huggingface --model google/gemma-4-31B-it --cache-dir /data/huggingface_cache

    # OpenAI-compatible API (vLLM, LM Studio, etc.)
    python main.py --llm openai --model my-model --llm-host http://localhost:8000/v1

    # Auto mode — LLM executes a task autonomously
    python main.py --mode auto --task "Put the mug in the fridge" --llm huggingface --model google/gemma-4-31B-it
"""

import argparse
import json
import sys

from thor_wrapper import ThorWrapper
from llm_client import OllamaClient, OpenAICompatibleClient, HuggingFaceClient


def build_llm(args):
    if args.llm == "ollama":
        return OllamaClient(
            model_name=args.model,
            host=args.llm_host,
        )
    elif args.llm == "huggingface":
        return HuggingFaceClient(
            model_name=args.model,
            cache_dir=args.cache_dir,
        )
    else:
        return OpenAICompatibleClient(
            model_name=args.model,
            base_url=args.llm_host,
            api_key=args.api_key,
        )


def run_interactive(args):
    """Interactive mode: you type -> LLM thinks -> Agent acts -> result shown."""
    print(f"Starting ai2thor scene {args.scene} ...")
    env = ThorWrapper(scene=args.scene, width=args.width, height=args.height)
    print(f"Scene {args.scene} loaded.\n")

    llm = build_llm(args)
    step_count = 0

    print("=" * 60)
    print("  ai2thor + LLM Interactive System")
    print("  Type natural language instructions to control the agent.")
    print("  Commands:  quit | reset | state | help")
    print("=" * 60)
    print()

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            if user_input.lower() == "reset":
                env.reset(args.scene)
                llm.reset_history()
                step_count = 0
                print("[System] Scene reset.\n")
                continue

            if user_input.lower() == "state":
                print(env.get_visible_objects_summary())
                print()
                continue

            if user_input.lower() == "help":
                print("  quit   — exit the program")
                print("  reset  — reset the scene and conversation history")
                print("  state  — print current visible objects")
                print("  help   — show this message")
                print("  (anything else) — send as instruction to LLM\n")
                continue

            scene_state = env.get_visible_objects_summary()

            print(f"[Calling LLM ({llm.model_name})...]")
            parsed, raw = llm.chat(user_input, scene_state)

            thought = parsed.get("thought", "(none)")
            action = parsed.get("action", {"action": "Done"})

            print(f"  Thought : {thought}")
            print(f"  Action  : {json.dumps(action, ensure_ascii=False)}")

            result = env.execute_action(action)
            step_count += 1

            status = "OK" if result["success"] else "FAIL"
            print(f"  Result  : [{status}] {result['message']}")

            if action.get("action") == "Done":
                print(f"\n[System] Task complete. Total steps: {step_count}")
                step_count = 0
                llm.reset_history()

            if step_count >= args.max_steps:
                print(f"\n[System] Reached max steps ({args.max_steps}). Resetting.")
                step_count = 0
                llm.reset_history()

            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()
        print("ai2thor shut down.")


def run_auto(args):
    """
    Auto mode: give a task, LLM executes multiple steps until Done.
    Useful for batch evaluation.
    """
    if not args.task:
        print("Error: --task is required in auto mode.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting ai2thor scene {args.scene} (auto mode)...")
    env = ThorWrapper(scene=args.scene, width=args.width, height=args.height)
    llm = build_llm(args)

    print(f"Task: {args.task}")
    print("-" * 50)

    results = []
    for step in range(args.max_steps):
        scene_state = env.get_visible_objects_summary()

        if step == 0:
            prompt = args.task
        else:
            prev = results[-1]
            status = "succeeded" if prev["success"] else f"failed ({prev['error']})"
            prompt = f"Previous action {prev['action_name']} {status}. Continue the task."

        parsed, raw = llm.chat(prompt, scene_state)
        action = parsed.get("action", {"action": "Done"})
        thought = parsed.get("thought", "")

        exec_result = env.execute_action(action)
        action_name = action.get("action", "Unknown")

        results.append({
            "step": step,
            "thought": thought,
            "action": action,
            "action_name": action_name,
            "success": exec_result["success"],
            "error": exec_result["error"],
        })

        status_str = "OK" if exec_result["success"] else "FAIL"
        print(
            f"  Step {step:2d}: {action_name:20s} [{status_str}]"
            f"  — {thought}"
        )

        if action_name == "Done":
            print(f"\nTask completed at step {step}.")
            break
    else:
        print(f"\nReached max steps ({args.max_steps}) without completing.")

    env.close()

    log_path = "auto_result.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {"task": args.task, "scene": args.scene, "steps": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="ai2thor + LLM control system")

    parser.add_argument(
        "--mode", choices=["interactive", "auto"], default="interactive",
        help="Run mode (default: interactive)",
    )
    parser.add_argument(
        "--scene", default="FloorPlan1",
        help="ai2thor scene name (default: FloorPlan1). "
             "Kitchen: 1-30, LivingRoom: 201-230, Bedroom: 301-330, Bathroom: 401-430",
    )
    parser.add_argument(
        "--llm", choices=["ollama", "openai", "huggingface"], default="ollama",
        help="LLM backend: ollama, openai (API-compatible), huggingface (local GPU)",
    )
    parser.add_argument(
        "--model", default="llama3",
        help="Model name. For huggingface use HF model ID e.g. google/gemma-4-31B-it",
    )
    parser.add_argument(
        "--llm-host", default="http://localhost:11434",
        help="LLM server URL (for ollama/openai backends)",
    )
    parser.add_argument(
        "--api-key", default="not-needed",
        help="API key for OpenAI-compatible backends",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace cache directory (e.g. /data/huggingface_cache). "
             "Only used with --llm huggingface",
    )
    parser.add_argument(
        "--task", default=None,
        help="Task description for auto mode",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Maximum steps per task (default: 50)",
    )
    parser.add_argument("--width", type=int, default=600, help="Render width")
    parser.add_argument("--height", type=int, default=600, help="Render height")

    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive(args)
    else:
        run_auto(args)


if __name__ == "__main__":
    main()
