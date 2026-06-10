"""
Unified model clients for SafeSight baseline evaluation.

Supports:
  - TextModelClient   : text-only LLMs (Qwen2.5, Llama-3, etc.)
  - GemmaVLMClient    : Gemma 3/4 multimodal
  - Qwen2VLClient     : Qwen2-VL series
  - LLaVAClient       : LLaVA-Next series

Loading pattern follows the user's verified HuggingFace usage:
  - AutoProcessor (not AutoTokenizer)
  - dtype="auto" for AutoModelForCausalLM
  - torch_dtype="auto" for specific model classes (Qwen2VL, LLaVA)
  - enable_thinking=False in apply_chat_template
  - processor.parse_response() for post-processing when available

Factory function `load_model()` auto-detects model family from model_id.
"""

from __future__ import annotations

import gc
import os
import time
from abc import ABC, abstractmethod

os.environ.setdefault("HF_HOME", "/data/huggingface_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/huggingface_cache/hub")

import torch
from PIL import Image

from benchmark.evaluate.gpu_utils import (
    build_device_map_kwargs,
    log_vram_after_load,
    preflight_check,
    print_gpu_status,
)

# ─── Evaluation generation config ────────────────────────────────────────
# For benchmark: low temperature → stable, high-quality, reproducible outputs.
# temperature=0 is greedy; 0.3 adds slight diversity while staying focused.
EVAL_GENERATE_KWARGS = dict(
    max_new_tokens=2048,
    temperature=0.3,
    do_sample=True,
    top_p=0.95,
    top_k=64,
)

# Reproducibility: fixed seed before every generate call
EVAL_SEED = 42


def _set_seed(seed: int = EVAL_SEED):
    """Set all random seeds for reproducible generation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_chat_template(processor, messages, **kwargs):
    """apply_chat_template with enable_thinking=False when supported."""
    try:
        return processor.apply_chat_template(
            messages, enable_thinking=False, **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def _decode_response(processor, output_ids, input_len: int) -> str:
    """
    Decode model output, using parse_response() when available (Gemma style).
    Falls back to skip_special_tokens=True for other models.
    """
    if hasattr(processor, "parse_response"):
        raw = processor.decode(output_ids[input_len:], skip_special_tokens=False)
        try:
            return processor.parse_response(raw)["content"]
        except Exception:
            pass
    return processor.decode(output_ids[input_len:], skip_special_tokens=True)


def _log_input_len(tag: str, model, input_len: int) -> None:
    """Print tokenized input length vs model context for diagnostic logs.

    Prints once per generate() call so we can verify that no silent
    truncation or context overflow is happening.  The scorer / plots
    can ignore these lines; they are informational only.
    """
    cfg = getattr(model.config, "text_config", model.config)
    ctx = getattr(cfg, "max_position_embeddings", None)
    gen_budget = EVAL_GENERATE_KWARGS.get("max_new_tokens")
    over = None
    if ctx is not None and gen_budget is not None:
        over = input_len + gen_budget - ctx
    print(
        f"  [len] {tag} input_ids={input_len}  ctx={ctx}  "
        f"max_new={gen_budget}  over={over}"
    )


class BaseModelClient(ABC):
    """Base class for all model clients."""

    supports_vision: bool = False

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        """Generate a response. Returns (text, latency_seconds)."""
        ...

    def unload(self):
        """Free GPU memory between model switches."""
        name = self.model_id
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [unload] {name} freed from GPU.")
        print_gpu_status("After Unload")


def _is_quantized(model_id: str) -> bool:
    """Check if model id indicates a pre-quantized model (4-bit / 8-bit)."""
    mid = model_id.lower()
    return any(tag in mid for tag in ("bnb-4bit", "bnb-8bit", "gptq", "int4", "int8"))


class TextModelClient(BaseModelClient):
    """Client for text-only models via AutoProcessor + AutoModelForCausalLM."""

    supports_vision = False

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        super().__init__(model_id, cache_dir)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        check = preflight_check(model_id)
        print(f"  [preflight] {check['message']}")
        dm_kwargs = build_device_map_kwargs(model_id)

        print(f"  [load] TextModel: {model_id} ...")
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id)
        except (ValueError, KeyError):
            self.processor = AutoTokenizer.from_pretrained(model_id)

        load_kwargs: dict = {**dm_kwargs}
        if _is_quantized(model_id):
            print(f"  [quant] Detected pre-quantized model, skipping dtype override")
        else:
            load_kwargs["dtype"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, **load_kwargs,
        )
        log_vram_after_load(model_id)

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        text = _apply_chat_template(
            self.processor, messages,
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        _log_input_len("text", self.model, input_len)

        _set_seed()
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **EVAL_GENERATE_KWARGS)
        latency = time.time() - t0

        result = _decode_response(self.processor, outputs[0], input_len)
        return result, latency


class GemmaVLMClient(BaseModelClient):
    """Client for Gemma 3/4 multimodal (AutoProcessor + AutoModelForCausalLM)."""

    supports_vision = True

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        super().__init__(model_id, cache_dir)
        from transformers import AutoModelForCausalLM, AutoProcessor

        check = preflight_check(model_id)
        print(f"  [preflight] {check['message']}")
        dm_kwargs = build_device_map_kwargs(model_id)

        print(f"  [load] GemmaVLM: {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype="auto", **dm_kwargs,
        )
        log_vram_after_load(model_id)

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        user_content: list[dict] = []
        if image_path:
            pil_image = Image.open(image_path).convert("RGB")
            user_content.append({"type": "image", "image": pil_image})
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        if image_path:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            text = _apply_chat_template(
                self.processor, messages,
                tokenize=False, add_generation_prompt=True,
            )
            inputs = self.processor(
                text=text, return_tensors="pt",
            ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        _log_input_len("gemma-vlm", self.model, input_len)

        _set_seed()
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **EVAL_GENERATE_KWARGS)
        latency = time.time() - t0

        result = _decode_response(self.processor, outputs[0], input_len)
        return result, latency


class Qwen2VLClient(BaseModelClient):
    """Client for Qwen2-VL series (Qwen2VLForConditionalGeneration)."""

    supports_vision = True

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        super().__init__(model_id, cache_dir)
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        check = preflight_check(model_id)
        print(f"  [preflight] {check['message']}")
        dm_kwargs = build_device_map_kwargs(model_id)

        print(f"  [load] Qwen2-VL: {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", **dm_kwargs,
        )
        log_vram_after_load(model_id)

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        user_content: list[dict] = []
        images = []
        if image_path:
            pil_image = Image.open(image_path).convert("RGB")
            images.append(pil_image)
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        text = _apply_chat_template(
            self.processor, messages,
            tokenize=False, add_generation_prompt=True,
        )

        proc_kwargs: dict = {"text": [text], "padding": True, "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        _log_input_len("qwen2vl", self.model, input_len)

        _set_seed()
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **EVAL_GENERATE_KWARGS)
        latency = time.time() - t0

        result = _decode_response(self.processor, outputs[0], input_len)
        return result, latency


class LLaVAClient(BaseModelClient):
    """Client for LLaVA-Next (llava-v1.6) series."""

    supports_vision = True

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        super().__init__(model_id, cache_dir)
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        check = preflight_check(model_id)
        print(f"  [preflight] {check['message']}")
        dm_kwargs = build_device_map_kwargs(model_id)

        # 【修复点 1】：强制确保开启多卡自动分配
        if "device_map" not in dm_kwargs:
            dm_kwargs["device_map"] = "auto"
            
        # 【修复点 2】：限制 GPU 0 的权重占用，强制溢出到 GPU 1
        # 预留出足够的显存（约 30GB）给 KV Cache 和计算过程
        if "max_memory" not in dm_kwargs:
            dm_kwargs["max_memory"] = {0: "18GiB", 1: "46GiB"}

        print(f"  [load] LLaVA: {model_id} ...")
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, **dm_kwargs,
        )
        log_vram_after_load(model_id)

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        full_prompt = f"{system_prompt}\n\n{user_text}"

        if image_path:
            pil_image = Image.open(image_path).convert("RGB")
            conversation = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt},
                ]},
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True,
            )
            inputs = self.processor(
                images=pil_image, text=prompt, return_tensors="pt",
            ).to(self.model.device)
        else:
            conversation = [
                {"role": "user", "content": [
                    {"type": "text", "text": full_prompt},
                ]},
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True,
            )
            inputs = self.processor(
                text=prompt, return_tensors="pt",
            ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        _log_input_len("llava", self.model, input_len)

        _set_seed()
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **EVAL_GENERATE_KWARGS)
        latency = time.time() - t0

        result = _decode_response(self.processor, outputs[0], input_len)
        return result, latency


class Llama32VisionClient(BaseModelClient):
    """Client for Llama-3.2-Vision (Mllama family, gated on HF)."""

    supports_vision = True

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        super().__init__(model_id, cache_dir)
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        check = preflight_check(model_id)
        print(f"  [preflight] {check['message']}")
        dm_kwargs = build_device_map_kwargs(model_id)

        print(f"  [load] Llama-3.2-Vision: {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", **dm_kwargs,
        )
        log_vram_after_load(model_id)

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        # Mllama prefers system+user style; image goes inside user content.
        full_user = f"{system_prompt}\n\n{user_text}"
        user_content: list[dict] = []
        images = None
        if image_path:
            images = [Image.open(image_path).convert("RGB")]
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": full_user})
        messages = [{"role": "user", "content": user_content}]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        proc_kwargs: dict = {"text": text, "return_tensors": "pt"}
        if images is not None:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        _log_input_len("llama3.2-vis", self.model, input_len)

        _set_seed()
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **EVAL_GENERATE_KWARGS)
        latency = time.time() - t0

        result = _decode_response(self.processor, outputs[0], input_len)
        return result, latency


class Qwen25VLClient(BaseModelClient):
    """Client for Qwen2.5-VL series (Qwen2_5_VLForConditionalGeneration).

    Note: this is a *different* class from Qwen2-VL.  Requires
    transformers >= 4.49.
    """

    supports_vision = True

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        super().__init__(model_id, cache_dir)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        check = preflight_check(model_id)
        print(f"  [preflight] {check['message']}")
        dm_kwargs = build_device_map_kwargs(model_id)

        print(f"  [load] Qwen2.5-VL: {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", **dm_kwargs,
        )
        log_vram_after_load(model_id)

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        from qwen_vl_utils import process_vision_info

        user_content: list[dict] = []
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        text = _apply_chat_template(
            self.processor, messages,
            tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        proc_kwargs: dict = {
            "text": [text], "padding": True, "return_tensors": "pt",
        }
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
        inputs = self.processor(**proc_kwargs).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        _log_input_len("qwen2.5-vl", self.model, input_len)

        _set_seed()
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **EVAL_GENERATE_KWARGS)
        latency = time.time() - t0

        result = _decode_response(self.processor, outputs[0], input_len)
        return result, latency


class InternVL2Client(BaseModelClient):
    """Client for InternVL2 series (custom code, uses model.chat API).

    InternVL2 ships its own chat template via model.chat(); we follow the
    official OpenGVLab usage pattern.
    """

    supports_vision = True

    # Image preprocessing constants (from OpenGVLab official example)
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        super().__init__(model_id, cache_dir)
        from transformers import AutoModel, AutoTokenizer

        check = preflight_check(model_id)
        print(f"  [preflight] {check['message']}")
        dm_kwargs = build_device_map_kwargs(model_id)

        print(f"  [load] InternVL2: {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **dm_kwargs,
        ).eval()
        # Expose tokenizer via .processor so unload() can free it.
        self.processor = self.tokenizer
        log_vram_after_load(model_id)

    def _load_pixels(self, image_path: str):
        """Convert image to tensor matching InternVL2's vision encoder input."""
        from torchvision import transforms

        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(
                (448, 448),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        return pixel_values

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        question = f"{system_prompt}\n\n{user_text}"
        gen_cfg = dict(
            max_new_tokens=EVAL_GENERATE_KWARGS["max_new_tokens"],
            do_sample=EVAL_GENERATE_KWARGS["do_sample"],
            temperature=EVAL_GENERATE_KWARGS["temperature"],
            top_p=EVAL_GENERATE_KWARGS["top_p"],
            top_k=EVAL_GENERATE_KWARGS["top_k"],
        )

        _set_seed()
        t0 = time.time()
        if image_path:
            pixel_values = self._load_pixels(image_path)
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                "<image>\n" + question,
                gen_cfg,
            )
        else:
            response = self.model.chat(
                self.tokenizer, None, question, gen_cfg,
            )
        latency = time.time() - t0
        # InternVL2 .chat returns a plain string already (no special tokens).
        # Skip _log_input_len since InternVL2 handles tokenization internally.
        return response, latency


class Phi35VisionClient(BaseModelClient):
    """Client for Phi-3.5-Vision (microsoft, custom code).

    Uses ``<|image_1|>`` placeholder + AutoProcessor; falls back to eager
    attention because flash-attn is not always available.
    """

    supports_vision = True

    def __init__(self, model_id: str, cache_dir: str = "/data/huggingface_cache"):
        super().__init__(model_id, cache_dir)
        from transformers import AutoModelForCausalLM, AutoProcessor

        check = preflight_check(model_id)
        print(f"  [preflight] {check['message']}")
        dm_kwargs = build_device_map_kwargs(model_id)

        print(f"  [load] Phi-3.5-Vision: {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, num_crops=4,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="eager",
            **dm_kwargs,
        )
        log_vram_after_load(model_id)

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, float]:
        images = None
        if image_path:
            images = [Image.open(image_path).convert("RGB")]
            user_block = "<|image_1|>\n" + user_text
        else:
            user_block = user_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        proc_kwargs: dict = {"text": prompt, "return_tensors": "pt"}
        if images is not None:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        _log_input_len("phi3.5-v", self.model, input_len)

        _set_seed()
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **EVAL_GENERATE_KWARGS,
            )
        latency = time.time() - t0

        result = _decode_response(self.processor, outputs[0], input_len)
        return result, latency


# ─── Factory ──────────────────────────────────────────────────────────────

def load_model(model_id: str, cache_dir: str = "/data/huggingface_cache") -> BaseModelClient:
    """Auto-detect model family and return the right client.

    Order matters: more specific patterns must be checked before more
    general ones (e.g. "qwen2.5-vl" before "qwen2-vl").
    """
    mid = model_id.lower()
    # Qwen2.5-VL must be matched BEFORE qwen2-vl to avoid false positive.
    if "qwen2.5-vl" in mid or "qwen2_5_vl" in mid or "qwen2.5vl" in mid:
        return Qwen25VLClient(model_id, cache_dir)
    if "qwen2-vl" in mid or "qwen2_vl" in mid:
        return Qwen2VLClient(model_id, cache_dir)
    if "llama-3.2" in mid and "vision" in mid:
        return Llama32VisionClient(model_id, cache_dir)
    if "internvl" in mid:
        return InternVL2Client(model_id, cache_dir)
    if "phi-3.5-vision" in mid or "phi-3.5-v" in mid or "phi3.5-vision" in mid:
        return Phi35VisionClient(model_id, cache_dir)
    if "llava" in mid:
        return LLaVAClient(model_id, cache_dir)
    if "gemma" in mid:
        return GemmaVLMClient(model_id, cache_dir)
    return TextModelClient(model_id, cache_dir)
