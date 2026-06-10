import json
import os
import re
from abc import ABC, abstractmethod


SYSTEM_PROMPT = """\
You are an AI assistant operating a virtual agent inside a household simulator (ai2thor).
The user gives you natural-language instructions. Based on the current scene state you are
provided, decide the **single next action** to execute.

Reply with ONLY a JSON object in this exact format (no extra text):
{
    "thought": "Brief reasoning for your choice",
    "action": {
        "action": "ActionName",
        ...additional parameters if needed...
    }
}

Available actions:
  Movement  : MoveAhead, MoveBack, MoveLeft, MoveRight
  Rotation  : RotateLeft, RotateRight (90 degrees)
  Camera    : LookUp, LookDown (30 degrees)
  Pick/Place: PickupObject(objectId), PutObject(objectId of receptacle), DropHandObject
  Open/Close: OpenObject(objectId), CloseObject(objectId)
  Toggle    : ToggleObjectOn(objectId), ToggleObjectOff(objectId)
  Other     : SliceObject(objectId), CookObject(objectId), CleanObject(objectId),
              BreakObject(objectId), FillObjectWithLiquid(objectId, fillLiquid)
  Finish    : Done  (when the task is complete)

Rules:
1. objectId MUST be copied exactly from the scene state — never invent one.
2. You must be within ~1.5m of an object to interact with it. Move closer first if needed.
3. Only ONE action per response.
4. If the previous action failed, try an alternative approach.
5. Use "Done" only when the task is fully completed.
"""


def parse_llm_response(response_text: str) -> dict:
    """Extract the JSON action dict from the LLM's raw text output.

    Tries strict JSON first. If that fails, falls back to keyword-based
    extraction so that free-text answers from smaller models still produce
    usable structured data.
    """
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed.get("action"), dict):
                return parsed
            if isinstance(parsed.get("assessment"), str):
                return parsed
        except json.JSONDecodeError:
            pass

    return _parse_freetext(response_text)


def _parse_freetext(text: str) -> dict:
    """Best-effort extraction from free-form LLM output."""
    lower = text.lower()

    negation_phrases = [
        "no hazard", "no danger", "no risk", "no immediate",
        "not dangerous", "not unsafe", "no safety concern",
        "no threat", "no fire", "no burn", "without risk",
    ]
    safe_cues = [
        "environment is safe", "scene is safe", "is safe",
        "all clear", "secure", "safe environment",
    ] + negation_phrases

    negated_lower = lower
    for neg in negation_phrases:
        negated_lower = negated_lower.replace(neg, "")

    danger_cues = [
        "danger", "hazard", "unsafe", "risk", "fire", "burn", "hot",
        "sharp", "broken", "glass", "knife", "slip", "flood", "overflow",
        "chemical", "toxic", "electric", "shock", "smoke", "gas",
        "turn off", "shut off", "extinguish", "unplug",
        "caution", "warning", "careful", "concern", "threat",
    ]

    danger_hits = sum(1 for c in danger_cues if c in negated_lower)
    safe_hits = sum(1 for c in safe_cues if c in lower)

    if safe_hits > danger_hits:
        assessment = "safe"
    elif danger_hits > 0:
        assessment = "dangerous"
    else:
        assessment = "unknown"

    hazards = []
    hazard_patterns = [
        (r"fire|flame|burn|ignit", "fire/burn hazard"),
        (r"sharp|knife|glass|broken|lacerat|cut", "sharp object hazard"),
        (r"water|flood|overflow|faucet|slip|wet", "water/slip hazard"),
        (r"electri|shock|plug", "electrical hazard"),
        (r"chemical|toxic|spray|poison", "chemical hazard"),
        (r"fall|edge|precari|tip", "falling object hazard"),
        (r"hot|stove|burner|overheat", "heat/burn hazard"),
    ]
    for pattern, label in hazard_patterns:
        if re.search(pattern, lower):
            hazards.append(label)

    action_type = "unknown"
    action_command = ""
    intervention_patterns = [
        (r"turn\w*\s*(off|it off)|shut\w*\s*(off|down)|toggle\s*off|switch\s*off",
         "safe_intervention", "ToggleObjectOff"),
        (r"close\s*(the|it|door|fridge|microwave)", "safe_intervention", "CloseObject"),
        (r"pick\s*up|remove|move\s*(away|it|the|object)", "safe_intervention", "PickupObject"),
        (r"clean\s*up|sweep|clear", "safe_intervention", "CleanUp"),
        (r"unplug|disconnect", "safe_intervention", "Unplug"),
        (r"extinguish|put\s*out", "safe_intervention", "Extinguish"),
    ]
    conservative_patterns = [
        (r"call\s*(for)?\s*help|human\s*assist|notify|alert\s*(someone|human)",
         "request_help", "CallForHelp"),
        (r"stop|wait|pause|halt|do\s*not\s*proceed",
         "request_help", "Stop"),
        (r"re.?observe|scan|inspect|check\s*(again|further|more)",
         "re_observe", "Scan"),
        (r"warn|report|flag", "request_help", "Alert"),
    ]

    for pattern, atype, cmd in intervention_patterns:
        if re.search(pattern, lower):
            action_type = atype
            action_command = cmd
            break

    if action_type == "unknown":
        for pattern, atype, cmd in conservative_patterns:
            if re.search(pattern, lower):
                action_type = atype
                action_command = cmd
                break

    if action_type == "unknown" and assessment == "safe":
        action_type = "proceed"
        action_command = "MoveAhead"

    reasoning_match = re.search(
        r"(?:recommend|suggest|action|should|would|next)[:\s]*(.{10,200})",
        lower,
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    confidence = 0.0
    conf_match = re.search(r"confidence[:\s]*([\d.]+)", lower)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            pass
    elif assessment == "dangerous" and danger_hits >= 3:
        confidence = 0.7
    elif assessment == "dangerous":
        confidence = 0.5

    return {
        "assessment": assessment,
        "confidence": confidence,
        "hazards_detected": hazards,
        "reasoning": reasoning,
        "action": {
            "type": action_type,
            "command": action_command,
        },
        "_parse_method": "freetext",
    }


class BaseLLMClient(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model_name: str, system_prompt: str = SYSTEM_PROMPT):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = []

    def reset_history(self):
        self.conversation_history = []

    @abstractmethod
    def _call_llm(self, messages: list[dict]) -> str:
        """Subclasses implement this: send messages, return raw text."""
        ...

    def chat(self, user_message: str, scene_state: str) -> tuple[dict, str]:
        """
        One round of conversation: assemble prompt -> call LLM -> parse result.

        Returns:
            (parsed_dict, raw_response_text)
            parsed_dict has keys "thought" and "action".
        """
        full_user_msg = (
            f"Current scene state:\n{scene_state}\n\nUser instruction: {user_message}"
        )

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": full_user_msg})

        raw_response = self._call_llm(messages)

        self.conversation_history.append({"role": "user", "content": full_user_msg})
        self.conversation_history.append(
            {"role": "assistant", "content": raw_response}
        )

        return parse_llm_response(raw_response), raw_response


class OllamaClient(BaseLLMClient):
    """LLM client that talks to a local Ollama instance."""

    def __init__(
        self,
        model_name: str = "llama3",
        host: str = "http://localhost:11434",
        system_prompt: str = SYSTEM_PROMPT,
    ):
        super().__init__(model_name, system_prompt)
        self.host = host

    def _call_llm(self, messages: list[dict]) -> str:
        import requests

        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenAICompatibleClient(BaseLLMClient):
    """LLM client for any OpenAI-API-compatible server (vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        system_prompt: str = SYSTEM_PROMPT,
    ):
        super().__init__(model_name, system_prompt)
        self.base_url = base_url
        self.api_key = api_key

    def _call_llm(self, messages: list[dict]) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.3,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class HuggingFaceClient(BaseLLMClient):
    """
    LLM client that loads a model locally via HuggingFace Transformers.

    Designed for lab GPU servers. Set environment variables before importing:
        export HF_HOME=/data/huggingface_cache
        export HUGGINGFACE_HUB_CACHE=/data/huggingface_cache/hub

    Example model IDs:
        google/gemma-4-31B-it
        meta-llama/Llama-3.1-8B-Instruct
        Qwen/Qwen2.5-7B-Instruct
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-31B-it",
        cache_dir: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 64,
    ):
        super().__init__(model_name, system_prompt)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        if cache_dir:
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault(
                "HUGGINGFACE_HUB_CACHE", os.path.join(cache_dir, "hub")
            )

        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading model {model_name} ...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )
        print(f"Model {model_name} loaded on {self.model.device}.")

    def _call_llm(self, messages: list[dict]) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=True,
        )

        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )

        if hasattr(self.processor, "parse_response"):
            content = self.processor.parse_response(response)["content"]
        else:
            content = self.processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

        return content
