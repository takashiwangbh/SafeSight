"""OpenAI Responses-API jury client.

Used by ``jury_eval.py`` when a ``JURY_MODELS`` entry has
``provider == "openai"``.  Mirrors the call signature of the local
HuggingFace clients (``generate(system_prompt, user_text) -> (text, latency)``)
so that the jury_eval main loop can stay provider-agnostic.

Design points
-------------
* **Authentication**: ``OPENAI_API_KEY`` env var only — never read from a
  config file or hard-coded constant.  Raises a clear error if missing.
* **API surface**: GPT-5 / GPT-5.5 reasoning models are called through the
  Responses API (``client.responses.create``), which supports the
  ``reasoning.effort`` parameter natively.  Chat-completions-style
  ``temperature`` is NOT passed because reasoning models do not accept it.
* **Determinism**: reasoning models default to deterministic decoding; we do
  not pass ``temperature`` or ``seed`` (the latter is only honoured on
  non-reasoning chat models).
* **Output schema**: relies on the same JSON-only prompt as the local
  jurors, parsed downstream by ``llm_client.parse_llm_response``.  We do
  not enable ``json_object`` mode because the prompt instruction is
  sufficient and we want the safety-net fallback in
  ``parse_llm_response`` to kick in if the model deviates.
* **Robustness**: 5-attempt exponential backoff (jittered) on any
  exception.  Rate-limit / 5xx / transient network errors are retried
  transparently.
* **Cost telemetry**: ``input_tokens``, ``output_tokens`` and (when
  available) ``reasoning_tokens`` are returned alongside the text so
  ``jury_eval.py`` can persist them into the per-scene JSON for
  retroactive cost auditing.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any


class OpenAIJury:
    """Drop-in replacement for the local TextModelClient when provider='openai'."""

    supports_vision = False
    is_api_client = True

    def __init__(
        self,
        model: str,
        reasoning_effort: str = "medium",
        max_output_tokens: int = 512,
        api_key: str | None = None,
        max_retries: int = 5,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK not installed. Run: "
                "pip install --upgrade openai"
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Run: export OPENAI_API_KEY=sk-... before launching jury_eval."
            )

        self.client = OpenAI(api_key=key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

        # `processor` / `model` are referenced by the local
        # `_jury_generate` helper but unused for API jurors.  Provide
        # safe placeholders so any incidental access does not crash.
        self.processor = None

    # ─── Compatibility helpers ──────────────────────────────────────

    def unload(self):
        """API clients have no GPU memory to free; no-op."""
        return

    # ─── Core call ──────────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | None = None,
        max_new_tokens: int | None = None,
    ) -> tuple[str, float, dict[str, Any]]:
        """Returns (text, latency_seconds, usage_dict).

        ``usage_dict`` contains ``input_tokens``, ``output_tokens`` and,
        when reported by the API, ``reasoning_tokens``.  All values are
        ints; missing fields are zero.
        """
        if image_path is not None:
            raise ValueError("OpenAIJury does not support image inputs.")

        budget = max_new_tokens or self.max_output_tokens

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                resp = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_text},
                    ],
                    reasoning={"effort": self.reasoning_effort},
                    max_output_tokens=budget,
                )
                latency = time.time() - t0

                text = (getattr(resp, "output_text", None) or "").strip()
                usage = self._extract_usage(resp)
                return text, latency, usage

            except Exception as e:  # noqa: BLE001 — retry-everything by design
                last_err = e
                wait = (2 ** (attempt + 1)) + random.random()
                err_name = type(e).__name__
                print(
                    f"    [openai] attempt {attempt + 1}/{self.max_retries} "
                    f"failed ({err_name}: {str(e)[:120]}); "
                    f"retrying in {wait:.1f}s ..."
                )
                time.sleep(wait)

        raise RuntimeError(
            f"OpenAI Responses API failed after {self.max_retries} attempts: "
            f"{last_err!r}"
        )

    # ─── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _extract_usage(resp: Any) -> dict[str, int]:
        usage_obj = getattr(resp, "usage", None)
        if usage_obj is None:
            return {"input_tokens": 0, "output_tokens": 0,
                    "reasoning_tokens": 0}

        # The OpenAI SDK exposes `usage` as either a pydantic-like model
        # or a plain dict, depending on version.  Normalise both.
        def _g(obj: Any, key: str, default: int = 0) -> int:
            if obj is None:
                return default
            if isinstance(obj, dict):
                v = obj.get(key, default)
            else:
                v = getattr(obj, key, default)
            return int(v) if v is not None else default

        details = (getattr(usage_obj, "output_tokens_details", None)
                   or (usage_obj.get("output_tokens_details")
                       if isinstance(usage_obj, dict) else None))

        return {
            "input_tokens":     _g(usage_obj, "input_tokens"),
            "output_tokens":    _g(usage_obj, "output_tokens"),
            "reasoning_tokens": _g(details, "reasoning_tokens", 0),
        }
