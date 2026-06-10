#!/usr/bin/env python3
"""
GPU detection, VRAM monitoring, and memory-aware model loading helpers.

Designed for dual-A6000 (2 × 48 GB) but works on any CUDA setup.

Usage as standalone diagnostic:
    python -m benchmark.evaluate.gpu_utils

Usage in code:
    from benchmark.evaluate.gpu_utils import (
        print_gpu_status, estimate_model_vram, preflight_check, build_device_map_kwargs,
    )
"""

from __future__ import annotations

import os
import re

import torch

# ─── Constants ────────────────────────────────────────────────────────────

GiB = 1 << 30
VRAM_SAFETY_MARGIN_GIB = 6.0  # reserve per-GPU for KV cache / activations


# ─── Core helpers ─────────────────────────────────────────────────────────

def gpu_count() -> int:
    return torch.cuda.device_count()


def gpu_info() -> list[dict]:
    """Return per-GPU info: name, total, used, free (all in GiB)."""
    infos = []
    for i in range(gpu_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / GiB
        reserved = torch.cuda.memory_reserved(i) / GiB
        allocated = torch.cuda.memory_allocated(i) / GiB
        # nvidia-smi free ≈ total - reserved  (reserved includes allocated + cached)
        free = total - reserved
        infos.append({
            "id": i,
            "name": props.name,
            "total_gib": round(total, 2),
            "reserved_gib": round(reserved, 2),
            "allocated_gib": round(allocated, 2),
            "free_gib": round(free, 2),
        })
    return infos


def total_free_vram_gib() -> float:
    """Sum of free VRAM across all GPUs."""
    return sum(g["free_gib"] for g in gpu_info())


def print_gpu_status(header: str = "GPU Status"):
    """Pretty-print current GPU VRAM to stdout."""
    infos = gpu_info()
    if not infos:
        print(f"[{header}] No CUDA GPUs detected.")
        return
    print(f"\n┌─ {header} {'─' * (56 - len(header))}")
    for g in infos:
        bar_len = 30
        used_ratio = 1.0 - (g["free_gib"] / g["total_gib"]) if g["total_gib"] > 0 else 0
        filled = int(bar_len * used_ratio)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"│ GPU {g['id']} ({g['name']}): "
            f"[{bar}] "
            f"{g['total_gib'] - g['free_gib']:.1f} / {g['total_gib']:.1f} GiB  "
            f"(free {g['free_gib']:.1f} GiB)"
        )
    print(f"│ Total free: {total_free_vram_gib():.1f} GiB")
    print(f"└{'─' * 60}")


# ─── Model memory estimation ────────────────────────────────────────────

def _param_billions_from_name(model_id: str) -> float | None:
    """Heuristically extract param count from model id string."""
    m = re.search(r"(\d+\.?\d*)\s*[bB]", model_id)
    if m:
        return float(m.group(1))
    return None


def _detect_quant_bytes(model_id: str) -> float:
    """Detect quantization level from model name. Returns bytes per param."""
    mid = model_id.lower()
    if "bnb-4bit" in mid or "gptq-4bit" in mid or "int4" in mid or "4bit" in mid:
        return 0.5  # 4-bit = 0.5 bytes per param
    if "bnb-8bit" in mid or "gptq-8bit" in mid or "int8" in mid or "8bit" in mid:
        return 1.0  # 8-bit = 1 byte per param
    return 2.0  # fp16/bf16 default


def estimate_model_vram(model_id: str, dtype_bytes: int | None = None) -> float:
    """
    Rough estimate of static model weight VRAM in GiB.

    Auto-detects quantization from model name (4-bit, 8-bit) if dtype_bytes
    is not explicitly provided.
    Does NOT include KV cache or activation memory (covered by safety margin).
    """
    params_b = _param_billions_from_name(model_id)
    if params_b is None:
        params_b = 7.0  # fallback assumption
    if dtype_bytes is None:
        dtype_bytes = _detect_quant_bytes(model_id)
    vram_gib = params_b * dtype_bytes  # 1B params × bytes_per_param = GiB
    return round(vram_gib, 1)


def preflight_check(model_id: str) -> dict:
    """
    Pre-flight VRAM check before loading a model.

    Returns dict:
        ok          : bool — safe to load
        model_est   : float — estimated weight VRAM (GiB)
        total_free  : float — total free VRAM (GiB)
        strategy    : str — "single_gpu" | "multi_gpu" | "too_large"
        message     : str — human readable summary
    """
    est = estimate_model_vram(model_id)
    infos = gpu_info()
    total_free = sum(g["free_gib"] for g in infos)
    needed = est + VRAM_SAFETY_MARGIN_GIB  # static weights + inference headroom

    # Check if it fits on a single GPU (preferred for speed)
    single_fit = any(g["free_gib"] >= needed for g in infos)
    # Check if it fits across all GPUs
    multi_fit = total_free >= needed

    if single_fit:
        strategy = "single_gpu"
        ok = True
        msg = (
            f"Model ~{est:.0f} GiB + {VRAM_SAFETY_MARGIN_GIB:.0f} GiB margin "
            f"fits on 1 GPU (free {max(g['free_gib'] for g in infos):.1f} GiB)"
        )
    elif multi_fit:
        strategy = "multi_gpu"
        ok = True
        msg = (
            f"Model ~{est:.0f} GiB needs multi-GPU "
            f"(total free {total_free:.1f} GiB)"
        )
    else:
        strategy = "too_large"
        ok = False
        msg = (
            f"NOT ENOUGH VRAM: model ~{est:.0f} GiB + margin, "
            f"but only {total_free:.1f} GiB free"
        )

    return {
        "ok": ok,
        "model_est_gib": est,
        "total_free_gib": round(total_free, 1),
        "strategy": strategy,
        "message": msg,
    }


# ─── Device map builder ─────────────────────────────────────────────────

def build_device_map_kwargs(model_id: str) -> dict:
    """
    Build kwargs for model.from_pretrained() with memory-safe device_map.

    For small models that fit on 1 GPU → constrain via max_memory so the
    second GPU stays available for the next model or concurrent work.
    For large models → let accelerate spread across GPUs.
    """
    check = preflight_check(model_id)
    infos = gpu_info()

    if not check["ok"]:
        print(f"  ⚠ WARNING: {check['message']}")
        print(f"  ⚠ Will attempt to load anyway with device_map='auto'")
        return {"device_map": "auto"}

    if check["strategy"] == "single_gpu":
        # Pick the GPU with the most free VRAM
        best = max(infos, key=lambda g: g["free_gib"])
        usable = best["free_gib"] - 1.0  # 1 GiB wiggle room for torch overhead
        max_memory = {best["id"]: f"{usable:.0f}GiB"}
        # Allow CPU offload as last resort but with generous GPU budget
        max_memory["cpu"] = "8GiB"
        return {"device_map": "auto", "max_memory": max_memory}

    # multi_gpu: let accelerate spread, but cap each GPU to avoid OOM
    max_memory = {}
    for g in infos:
        usable = g["free_gib"] - 2.0
        if usable > 0:
            max_memory[g["id"]] = f"{usable:.0f}GiB"
    max_memory["cpu"] = "16GiB"
    return {"device_map": "auto", "max_memory": max_memory}


def log_vram_after_load(model_id: str):
    """Log VRAM consumption right after model loading."""
    infos = gpu_info()
    parts = []
    for g in infos:
        used = g["total_gib"] - g["free_gib"]
        if used > 0.5:
            parts.append(f"GPU{g['id']}={used:.1f}/{g['total_gib']:.1f}GiB")
    if parts:
        print(f"  [vram] after loading {model_id}: {', '.join(parts)}")


# ─── CLI: run as diagnostic ──────────────────────────────────────────────

def _cli():
    print("=" * 64)
    print("SafeSight GPU Diagnostic")
    print("=" * 64)

    n = gpu_count()
    if n == 0:
        print("No CUDA GPUs found. Make sure CUDA drivers are installed.")
        return

    print(f"\nDetected {n} GPU(s):")
    print_gpu_status("Current VRAM")

    test_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "tiiuae/falcon-7b-instruct",
        "nvidia/Llama3-ChatQA-1.5-8B",
        "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-vicuna-13b-hf",
        "llava-hf/llava-v1.6-34b-hf",
        "google/gemma-4-31B-it",
    ]
    print("\nPre-flight checks:")
    print("-" * 64)
    for mid in test_models:
        r = preflight_check(mid)
        status = "✓" if r["ok"] else "✗"
        print(f"  {status} {mid}")
        print(f"    est={r['model_est_gib']:.0f}GiB  free={r['total_free_gib']:.0f}GiB  strategy={r['strategy']}")
        dm = build_device_map_kwargs(mid)
        if "max_memory" in dm:
            print(f"    max_memory={dm['max_memory']}")
        print()


if __name__ == "__main__":
    _cli()
