"""Run jury models over the sampled scenes.

Loads each jury model one at a time (so a single A6000 worth of VRAM is
enough for a 4-bit 70/72B), runs the neutral validation prompt over every
sampled scene, parses each response with the project's standard
``parse_llm_response`` helper, and stores one result JSON per scene.

Result layout::

    data/cross_verify/jury_results/<short_name>/<basename>.json

Each file contains the raw model output, the parsed dict, and the scene's
ground-truth fields (copied for convenient downstream joins).

Usage
-----
    # Run all jury models on the manifest produced by sample_scenes
    python -m cross_verify.jury_eval

    # Run only one jury model
    python -m cross_verify.jury_eval --jury qwen2.5-72b-bnb4

    # Quick smoke test on the first 5 scenes
    python -m cross_verify.jury_eval --max-scenes 5

    # Override cache dir
    python -m cross_verify.jury_eval --cache-dir /data/huggingface_cache

Notes
-----
* Per-scene output files are written atomically; rerunning skips scenes
  that already have a result file (incremental restart-friendly).
* HuggingFace juries use greedy decoding (``do_sample=False``) so the
  agreement numbers downstream are deterministic.  We bypass the
  client's stock ``generate()`` (which hard-codes ``EVAL_GENERATE_KWARGS``
  with mild sampling) and drive the underlying ``model.generate()``
  ourselves with ``JURY_GENERATE_KWARGS`` from cross_verify/config.py.
* OpenAI juries (``provider="openai"`` in the registry) are reached over
  the network through ``cross_verify/openai_client.py``.  They require
  ``OPENAI_API_KEY`` in the environment and can be launched in parallel
  with a HuggingFace juror because they consume no VRAM.

  Example::

      export OPENAI_API_KEY=sk-...
      python -m cross_verify.jury_eval --jury gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import torch  # noqa: E402

from benchmark.evaluate.gpu_utils import (        # noqa: E402
    preflight_check, print_gpu_status,
)
from benchmark.evaluate.vlm_client import (       # noqa: E402
    _apply_chat_template, _decode_response, _log_input_len,
    load_model,
)
from llm_client import parse_llm_response  # noqa: E402

from cross_verify.config import (  # noqa: E402
    DEFAULT_CACHE_DIR, JURY_GENERATE_KWARGS, JURY_MODELS,
    JURY_RESULTS_DIR, JURY_SEED, SAMPLED_SCENES_FILE, ensure_dirs,
)
from cross_verify.jury_prompt import (  # noqa: E402
    JURY_SYSTEM_PROMPT, build_jury_user_prompt,
)


def _provider(jury: dict) -> str:
    """Default to huggingface for backward compatibility."""
    return jury.get("provider", "huggingface")


def _set_seed(seed: int = JURY_SEED):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jury_generate(
    client,
    system_prompt: str,
    user_text: str,
    gen_kwargs: dict,
) -> tuple[str, float]:
    """Run one prompt through ``client.model`` with our own gen kwargs.

    Reimplements the body of ``TextModelClient.generate`` so we can pass
    deterministic greedy kwargs (``JURY_GENERATE_KWARGS``) instead of the
    sampling kwargs that ``client.generate()`` hard-codes.  Reuses the
    same chat-template / decode helpers from vlm_client so behaviour is
    otherwise identical (Gemma's ``parse_response`` fallback included).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]
    text = _apply_chat_template(
        client.processor, messages,
        tokenize=False, add_generation_prompt=True,
    )
    inputs = client.processor(
        text=text, return_tensors="pt",
    ).to(client.model.device)
    input_len = inputs["input_ids"].shape[-1]
    _log_input_len("jury", client.model, input_len)

    _set_seed()
    t0 = time.time()
    with torch.inference_mode():
        outputs = client.model.generate(**inputs, **gen_kwargs)
    latency = time.time() - t0

    return _decode_response(client.processor, outputs[0], input_len), latency


def _load_scene(gt_path: str) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_manifest() -> list[dict]:
    """Return one flat list of {basename, gt_path, gt_*} items."""
    if not os.path.exists(SAMPLED_SCENES_FILE):
        raise FileNotFoundError(
            f"{SAMPLED_SCENES_FILE} not found. "
            f"Run `python -m cross_verify.sample_scenes` first."
        )
    with open(SAMPLED_SCENES_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    items: list[dict] = []
    for split in ("hazard", "safe"):
        for rec in manifest.get(split, []):
            rec = dict(rec)
            rec["split"] = split
            items.append(rec)
    return items


def _resolve_jury(name: str) -> dict:
    for j in JURY_MODELS:
        if j["short_name"] == name or j["hf_id"] == name:
            return j
    raise KeyError(
        f"Unknown jury: {name!r}. "
        f"Available: {[j['short_name'] for j in JURY_MODELS]}"
    )


def _build_record(
    jury: dict,
    scene: dict,
    raw: str,
    parsed: dict,
    latency: float,
    gen_kwargs: dict,
    extra: dict | None = None,
) -> dict:
    record = {
        "jury_short":    jury["short_name"],
        "jury_hf_id":    jury["hf_id"],
        "jury_provider": _provider(jury),
        "basename":      scene["basename"],
        "split":         scene["split"],
        "gt": {
            "scene_name":    scene.get("scene_name"),
            "room_type":     scene.get("room_type"),
            "recipe_name":   scene.get("recipe_name"),
            "is_safe":       scene.get("is_safe"),
            "severity":      scene.get("severity"),
            "danger_labels": scene.get("danger_labels", []),
        },
        "raw_response":  raw,
        "parsed":        parsed,
        "gen_kwargs":    dict(gen_kwargs),
        "jury_seed":     JURY_SEED,
        "latency_sec":   round(latency, 2),
    }
    if extra:
        record.update(extra)
    return record


def _write_record(out_path: str, record: dict) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def _print_progress(i: int, total: int, ok: int, err: int, skip: int,
                    last_tag: str, last_latency: float):
    if (i + 1) % 25 == 0 or (i + 1) == total:
        print(
            f"    [{i+1}/{total}] ok={ok} err={err} skip={skip}  "
            f"last: {last_tag} ({last_latency:.1f}s)"
        )


def run_huggingface_jury(
    jury: dict,
    scenes: list[dict],
    cache_dir: str,
    gen_kwargs: dict,
):
    short = jury["short_name"]
    out_dir = os.path.join(JURY_RESULTS_DIR, short)
    os.makedirs(out_dir, exist_ok=True)

    print("─" * 64)
    print(f"  Jury     : {short}  ({jury['hf_id']})  [provider=huggingface]")
    print(f"  Scenes   : {len(scenes)}")
    print(f"  Output   : {out_dir}")
    print(f"  Gen cfg  : {gen_kwargs}")
    print("─" * 64)

    check = preflight_check(jury["hf_id"])
    print(f"  [preflight] {check['message']}")
    if not check["ok"]:
        print(f"  ✗ SKIPPING {short}: insufficient VRAM")
        return

    try:
        client = load_model(jury["hf_id"], cache_dir)
    except Exception:
        print(f"  ✗ Failed to load {jury['hf_id']}:")
        traceback.print_exc()
        return

    ok = err = skip = 0
    t_start = time.time()
    for i, scene in enumerate(scenes):
        out_path = os.path.join(out_dir, f"{scene['basename']}.json")
        if os.path.exists(out_path):
            skip += 1
            continue

        try:
            scene_data = _load_scene(scene["gt_path"])
            user_prompt = build_jury_user_prompt(scene_data)
            raw, latency = _jury_generate(
                client,
                system_prompt=JURY_SYSTEM_PROMPT,
                user_text=user_prompt,
                gen_kwargs=gen_kwargs,
            )
            parsed = parse_llm_response(raw)
            record = _build_record(jury, scene, raw, parsed, latency, gen_kwargs)
            _write_record(out_path, record)
            ok += 1
            _print_progress(i, len(scenes), ok, err, skip,
                            parsed.get("assessment", "?"), latency)

        except torch.cuda.OutOfMemoryError:
            err += 1
            print(f"    [{i+1}] {scene['basename']} — OOM!")
            print_gpu_status("OOM snapshot")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            err += 1
            print(f"    [{i+1}] {scene['basename']} — ERROR")
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(
        f"  -> {short}: {ok} ok / {err} err / {skip} skip  "
        f"(elapsed {elapsed:.0f}s, mean {elapsed / max(ok, 1):.1f}s / scene)"
    )
    client.unload()


def run_openai_jury(
    jury: dict,
    scenes: list[dict],
):
    """Run an OpenAI Responses-API juror over the sampled scenes.

    Does not use VRAM and does not interact with the local model loader,
    so it can be run in parallel with a HuggingFace juror on the same
    machine.  Cost (input + output + reasoning tokens) is logged per
    scene under ``usage`` for retroactive auditing.
    """
    from cross_verify.openai_client import OpenAIJury

    short = jury["short_name"]
    out_dir = os.path.join(JURY_RESULTS_DIR, short)
    os.makedirs(out_dir, exist_ok=True)

    effort = jury.get("reasoning_effort", "medium")
    api_kwargs = {
        "reasoning_effort":  effort,
        "max_output_tokens": jury.get("max_output_tokens", 512),
    }

    print("─" * 64)
    print(f"  Jury     : {short}  ({jury['hf_id']})  [provider=openai]")
    print(f"  Scenes   : {len(scenes)}")
    print(f"  Output   : {out_dir}")
    print(f"  API cfg  : {api_kwargs}")
    print("─" * 64)

    try:
        client = OpenAIJury(
            model=jury["hf_id"],
            **api_kwargs,
        )
    except Exception:
        print(f"  ✗ Failed to initialise OpenAI client for {short}:")
        traceback.print_exc()
        return

    ok = err = skip = 0
    cum_input = cum_output = cum_reasoning = 0
    t_start = time.time()
    for i, scene in enumerate(scenes):
        out_path = os.path.join(out_dir, f"{scene['basename']}.json")
        if os.path.exists(out_path):
            skip += 1
            continue

        try:
            scene_data = _load_scene(scene["gt_path"])
            user_prompt = build_jury_user_prompt(scene_data)
            raw, latency, usage = client.generate(
                system_prompt=JURY_SYSTEM_PROMPT,
                user_text=user_prompt,
            )
            parsed = parse_llm_response(raw)
            record = _build_record(
                jury, scene, raw, parsed, latency,
                gen_kwargs=api_kwargs,
                extra={"usage": usage},
            )
            _write_record(out_path, record)
            ok += 1
            cum_input += usage.get("input_tokens", 0)
            cum_output += usage.get("output_tokens", 0)
            cum_reasoning += usage.get("reasoning_tokens", 0)
            _print_progress(i, len(scenes), ok, err, skip,
                            parsed.get("assessment", "?"), latency)
        except Exception:
            err += 1
            print(f"    [{i+1}] {scene['basename']} — ERROR")
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(
        f"  -> {short}: {ok} ok / {err} err / {skip} skip  "
        f"(elapsed {elapsed:.0f}s, mean {elapsed / max(ok, 1):.1f}s / scene)"
    )
    print(
        f"     tokens: input={cum_input:,}  output={cum_output:,}  "
        f"reasoning={cum_reasoning:,}"
    )


def run_one_jury(
    jury: dict,
    scenes: list[dict],
    cache_dir: str,
    gen_kwargs: dict,
):
    """Dispatch to the provider-specific runner."""
    provider = _provider(jury)
    if provider == "openai":
        return run_openai_jury(jury, scenes)
    if provider == "huggingface":
        return run_huggingface_jury(jury, scenes, cache_dir, gen_kwargs)
    raise ValueError(f"Unknown jury provider: {provider!r}")


def main():
    p = argparse.ArgumentParser(description="Run jury models on cross-verify samples.")
    p.add_argument("--jury", nargs="+", default=None,
                   help=f"Subset of jury short names. "
                        f"Default: all ({[j['short_name'] for j in JURY_MODELS]}).")
    p.add_argument("--max-scenes", type=int, default=None,
                   help="Limit number of scenes (smoke test).")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override max_new_tokens from JURY_GENERATE_KWARGS.")
    args = p.parse_args()

    ensure_dirs()
    scenes = _load_manifest()
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    juries = (
        [_resolve_jury(n) for n in args.jury]
        if args.jury else list(JURY_MODELS)
    )

    gen_kwargs = dict(JURY_GENERATE_KWARGS)
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    print("=" * 64)
    print(f"SafeSight Cross-Verify  ({len(scenes)} scenes × {len(juries)} juries)")
    print(f"Generation kwargs: {gen_kwargs}")
    print("=" * 64)
    print_gpu_status("Initial GPU")

    for jury in juries:
        run_one_jury(
            jury=jury,
            scenes=scenes,
            cache_dir=args.cache_dir,
            gen_kwargs=gen_kwargs,
        )

    print(f"\nAll juries done. Results: {JURY_RESULTS_DIR}")
    print_gpu_status("Final GPU")


if __name__ == "__main__":
    main()
