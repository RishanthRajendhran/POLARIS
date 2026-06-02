# providers/vertex_engine.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import re
import copy

from google import genai
from google.genai.types import HttpOptions  # GenerateContentConfig is optional; dict also works.

from core.engine import Engine
from core.task_base import Task
from core.sampling import get_engine_cfg
from core.structured_output import supports_native_schema

import copy
from typing import Any, Dict, List, Optional, Tuple

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _all_ints(xs: List[Any]) -> bool:
    return all(_is_num(x) and float(x).is_integer() for x in xs)

def _sanitize_schema_for_vertex(
    schema: Dict[str, Any],
    *,
    numeric_enum_mode: str = "range",   # "range" | "drop" | "string"
    numeric_const_mode: str = "range",  # "range" | "drop"
) -> Dict[str, Any]:
    """
    Make an OpenAI-style JSON Schema safer for Vertex/Gemini structured output.

    Vertex (google.genai.types.Schema) expects enum values to be strings.
    Many of your schemas use numeric enums (e.g., severity enum [1..10]).

    We transform:
      - enum=[numeric...] -> minimum=min(enum), maximum=max(enum)   (default)
        (or drop, or cast to strings)
      - const=numeric -> minimum=maximum=const (default) or drop

    We preserve:
      - string enums unchanged
      - the rest of the schema structure
    """
    s = copy.deepcopy(schema)

    def _handle_enum(node: Dict[str, Any]) -> None:
        enum = node.get("enum", None)
        if not isinstance(enum, list) or not enum:
            return

        # Already all strings: OK
        if all(isinstance(v, str) for v in enum):
            return

        # All numeric: convert
        if all(_is_num(v) for v in enum):
            if numeric_enum_mode == "drop":
                node.pop("enum", None)
                return
            if numeric_enum_mode == "string":
                node["type"] = "string"
                node["enum"] = [str(v) for v in enum]
                return

            # numeric_enum_mode == "range" (default)
            lo = min(float(v) for v in enum)
            hi = max(float(v) for v in enum)
            node.pop("enum", None)

            # Keep/repair type
            if node.get("type") not in ("integer", "number"):
                node["type"] = "integer" if _all_ints(enum) else "number"

            node["minimum"] = lo
            node["maximum"] = hi
            return

        # Mixed enum types -> force string enum as last resort
        node["type"] = "string"
        node["enum"] = [str(v) for v in enum]

    def _handle_const(node: Dict[str, Any]) -> None:
        if "const" not in node:
            return
        c = node.get("const", None)
        if not _is_num(c):
            return

        if numeric_const_mode == "drop":
            node.pop("const", None)
            return

        # numeric_const_mode == "range"
        node.pop("const", None)
        if node.get("type") not in ("integer", "number"):
            node["type"] = "integer" if float(c).is_integer() else "number"
        node["minimum"] = float(c)
        node["maximum"] = float(c)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            _handle_enum(node)
            _handle_const(node)
            for v in list(node.values()):
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(s)
    return s

def _build_http_options_with_consumption(engine_cfg: Dict[str, Any]) -> HttpOptions:
    """
    Build HttpOptions for google.genai.Client, setting headers that select
    Vertex AI consumption options (Standard / Flex / Priority PayGo).

    Config:
      engine.provider_params.vertex_consumption:
        - "standard" (default) -> no special headers (Standard PayGo)
        - "flex"               -> Flex PayGo, shared-only
        - "priority"           -> Priority PayGo, shared-only

    Notes:
      - These headers are honored when GOOGLE_GENAI_USE_VERTEXAI=true
        and the client is using Vertex AI backend.
      - If you use an API key against the Gemini Developer API only,
        these headers are harmless no-ops.
    """
    provider_params = engine_cfg.get("provider_params") or {}
    consumption = (provider_params.get("vertex_consumption") or "standard").strip().lower()

    headers: Dict[str, str] = {}

    if consumption in ("flex", "priority"):
        # Shared-only: don't spill into Provisioned Throughput.
        # See:
        #   https://cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo
        #   https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo
        headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
        headers["X-Vertex-AI-LLM-Shared-Request-Type"] = consumption

    # api_version "v1" is what you're already using for Gemini 2.5 / 3.x.
    if headers:
        return HttpOptions(api_version="v1", headers=headers)
    else:
        return HttpOptions(api_version="v1")


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


class VertexAIEngine(Engine):
    """
    Gemini / Claude API engine (synchronous) via Vertex AI.

    For Gemini models: uses google.genai SDK.
    For Claude models (model name contains 'claude'): uses anthropic.AnthropicVertex SDK.

    Expected run_config["engine"] fields:

      {
        "provider": "vertex",
        "model": "gemini-3.0-flash",        // or "claude-opus-4-6"

        // How to authenticate:
        "api_key": "sk-...",                // optional; else GEMINI_API_KEY or GOOGLE_API_KEY env

        "sampling": {
          "temperature": 0.2,
          "top_p": 0.9,
          "top_k": 40,
          "max_output_tokens": 8192,
          "stop_sequences": ["###"],
          "candidate_count": 1,
          "presence_penalty": 0.0,
          "frequency_penalty": 0.0,
          "seed": 42,
          "response_logprobs": false,
          "logprobs": 0,
          // optional extras:
          // "thinking_config": {...},
          // "response_modalities": [...],
          // "image_config": {...},
          // "speech_config": {...},
          // "media_resolution": "...",
          // "audio_timestamp": false,
          // "enable_enhanced_civic_answers": false
        },

        "provider_overrides": {
          "vertex": {
            // sampling-style overrides; these win over engine.sampling
            // e.g., "temperature": 0.0, "top_p": 0.8, ...

            // For Claude: "region": "us-east5" (default)

            // structured output (if you really want to force it in config):
            // "response_mime_type": "application/json",
            // "response_schema": {...},          // Gemini Schema
            // "response_json_schema": {...},     // JSON Schema

            // safety settings, etc.
            // "safety_settings": [...]
          }
        },

        "provider_params": {
          // Optional hint for cost accounting (online vs batch pricing)
          "vertex_mode": "online"
        }
      }

    This engine will ALSO consult task.get_response_format("vertex", run_config)
    for structured-output (JSON schema) and will only use that if
    supports_native_schema("vertex", run_config) is True.

    Authentication:
      - Preferred: set run_config["engine"]["api_key"].
      - Or set one of:
          GEMINI_API_KEY
          GOOGLE_API_KEY

    """

    name = "vertex"

    def __init__(self, run_config: Dict[str, Any]):
        engine_cfg = get_engine_cfg(run_config)
        self.engine_cfg = engine_cfg

        self.model = engine_cfg.get("model")
        if not self.model:
            raise ValueError(
                "engine.model must be set for VertexAIEngine (Gemini/Vertex), "
                "e.g. 'gemini-3-flash-preview'."
            )

        self.is_claude = "claude" in self.model.lower()

        if self.is_claude:
            # Claude models via AnthropicVertex SDK
            from anthropic import AnthropicVertex
            overrides = (engine_cfg.get("provider_overrides", {}) or {}).get("vertex", {}) or {}
            region = overrides.get("region", "us-east5")
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            self.claude_client = AnthropicVertex(
                project_id=project_id,
                region=region,
            )
        else:
            # Gemini models via google.genai
            # Build HttpOptions with any Vertex consumption headers (standard/flex/priority).
            http_options = _build_http_options_with_consumption(engine_cfg)

            # Auth / backend selection:
            api_key = (
                engine_cfg.get("api_key")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
            )

            if api_key:
                self.client = genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                )
            else:
                self.client = genai.Client(
                    http_options=http_options,
                )

    # ---------- helpers: messages -> (system_instruction, contents) ----------

    def _messages_to_gemini(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, Any]:
        """
        Convert OpenAI-style messages into:
          - system_instruction (string or "")
          - contents (string or simple list) for Gemini API.

        For your current story tasks, this is effectively:
          - system_instruction: concatenation of all system messages.
          - contents: the *last* user message content as a single string.

        This matches your old script's usage, and is sufficient for story judges.
        """
        sys_buf: List[str] = []
        last_user: str = ""

        for m in messages:
            role = (m.get("role") or "user").strip()
            text = str(m.get("content", "") or "")

            if role == "system":
                sys_buf.append(text)
            elif role == "user":
                last_user = text
            else:
                # For now, ignore assistant/tool messages for Gemini;
                # your judge tasks don't use them.
                pass

        system_instruction = _normalize_whitespace("\n\n".join(sys_buf)) if sys_buf else ""
        contents = last_user or ""

        return system_instruction, contents

    # ---------- helpers: build GenerateContentConfig dict ----------

    def _build_generate_config(self) -> Dict[str, Any]:
        """
        Build a dict suitable for types.GenerateContentConfig(**cfg).

        We merge:
          - engine.sampling
          - engine.provider_overrides["vertex"]
        with provider_overrides taking precedence.

        Supported keys (snake_case on your side):

         vertexai_user: label to track expense in billing; default: polaris

          # Core sampling
          temperature
          top_p              -> top_p
          top_k              -> top_k
          max_output_tokens  (or max_tokens / max_new_tokens)
          stop_sequences     -> stop_sequences
          candidate_count / n -> candidate_count

          # Repetition / style
          presence_penalty   -> presence_penalty
          frequency_penalty  -> frequency_penalty

          # Determinism / diagnostics
          seed               -> seed
          response_logprobs  -> response_logprobs
          logprobs           -> logprobs

          # Thinking / multimodal options
          thinking_config        -> thinking_config
          response_modalities    -> response_modalities
          image_config           -> image_config
          speech_config          -> speech_config
          media_resolution       -> media_resolution
          audio_timestamp        -> audio_timestamp
          enable_enhanced_civic_answers -> enable_enhanced_civic_answers

          # If you set response_mime_type / response_schema / response_json_schema
          # here, they will be passed through, but task-provided schema (if any)
          # will override only when those keys are missing.
        """
        sampling = self.engine_cfg.get("sampling", {}) or {}
        overrides = (self.engine_cfg.get("provider_overrides", {}) or {}).get("vertex", {}) or {}
        merged: Dict[str, Any] = {**sampling, **overrides}
        thinking_cfg = (merged.get("thinking_config"))

        vertexai_user="polaris"
        if "vertexai_user" in merged:
            vertexai_user = str(merged["vertexai_user"])

        cfg: Dict[str, Any] = {
            "labels":{"user": vertexai_user},
        }

        # --- Randomness ---
        if "temperature" in merged:
            cfg["temperature"] = float(merged["temperature"])

        if "top_p" in merged:
            cfg["top_p"] = float(merged["top_p"])
        if "topP" in merged:
            cfg["top_p"] = float(merged["topP"])

        if "top_k" in merged:
            cfg["top_k"] = int(merged["top_k"])
        if "topK" in merged:
            cfg["top_k"] = int(merged["topK"])

        # --- Length ---
        if "max_output_tokens" in merged:
            cfg["max_output_tokens"] = int(merged["max_output_tokens"])
        elif "max_tokens" in merged:
            cfg["max_output_tokens"] = int(merged["max_tokens"])
        elif "max_new_tokens" in merged:
            cfg["max_output_tokens"] = int(merged["max_new_tokens"])

        # --- Stop sequences ---
        if "stop_sequences" in merged:
            cfg["stop_sequences"] = list(merged["stop_sequences"])

        # --- Candidates ---
        if "candidate_count" in merged:
            cfg["candidate_count"] = int(merged["candidate_count"])
        elif "candidateCount" in merged:
            cfg["candidate_count"] = int(merged["candidateCount"])
        elif "n" in merged:
            cfg["candidate_count"] = int(merged["n"])

        # --- Repetition / style ---
        if "presence_penalty" in merged:
            cfg["presence_penalty"] = float(merged["presence_penalty"])
        if "frequency_penalty" in merged:
            cfg["frequency_penalty"] = float(merged["frequency_penalty"])

        # --- Determinism / diagnostics ---
        if "seed" in merged:
            cfg["seed"] = int(merged["seed"])
        if "response_logprobs" in merged:
            cfg["response_logprobs"] = bool(merged["response_logprobs"])
        if "logprobs" in merged:
            cfg["logprobs"] = int(merged["logprobs"])

        # --- Thinking / multimodal extras ---
        if "thinking_config" in merged:
            cfg["thinking_config"] = merged["thinking_config"]
        if "thinkingConfig" in merged:
            cfg["thinking_config"] = merged["thinkingConfig"]

        if "response_modalities" in merged:
            cfg["response_modalities"] = list(merged["response_modalities"])
        if "image_config" in merged:
            cfg["image_config"] = merged["image_config"]
        if "speech_config" in merged:
            cfg["speech_config"] = merged["speech_config"]
        if "media_resolution" in merged:
            cfg["media_resolution"] = merged["media_resolution"]
        if "audio_timestamp" in merged:
            cfg["audio_timestamp"] = bool(merged["audio_timestamp"])
        if "enable_enhanced_civic_answers" in merged:
            cfg["enable_enhanced_civic_answers"] = bool(merged["enable_enhanced_civic_answers"])

        # --- Structured-output fields (optional; task may also set them) ---
        if "response_mime_type" in merged:
            cfg["response_mime_type"] = merged["response_mime_type"]
        if "response_schema" in merged:
            cfg["response_schema"] = merged["response_schema"]
        if "response_json_schema" in merged:
            cfg["response_json_schema"] = merged["response_json_schema"]

        if thinking_cfg:
            cfg["thinking_config"] = thinking_cfg

        return cfg

    # ---------- Claude via AnthropicVertex ----------
    def _generate_one_claude(
        self,
        messages: List[Dict[str, str]],
        task: Task,
        run_config: Dict[str, Any],
    ) -> Any:
        """Generate using Claude via AnthropicVertex SDK."""
        sampling = self.engine_cfg.get("sampling", {}) or {}
        overrides = (self.engine_cfg.get("provider_overrides", {}) or {}).get("vertex", {}) or {}
        merged = {**sampling, **overrides}

        # Separate system messages from user/assistant messages
        system_parts = []
        claude_messages = []
        for m in messages:
            role = (m.get("role") or "user").strip()
            text = str(m.get("content", "") or "")
            if role == "system":
                system_parts.append(text)
            else:
                claude_messages.append({"role": role, "content": text})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": int(merged.get("max_output_tokens", merged.get("max_tokens", 8192))),
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if "temperature" in merged:
            kwargs["temperature"] = float(merged["temperature"])
        if "top_p" in merged:
            kwargs["top_p"] = float(merged["top_p"])
        if "top_k" in merged:
            kwargs["top_k"] = int(merged["top_k"])
        if "stop_sequences" in merged:
            kwargs["stop_sequences"] = list(merged["stop_sequences"])

        resp = self.claude_client.messages.create(**kwargs)

        content_text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content_text += block.text

        usage = {
            "prompt_tokens": getattr(resp.usage, "input_tokens", None),
            "completion_tokens": getattr(resp.usage, "output_tokens", None),
            "total_tokens": (
                (getattr(resp.usage, "input_tokens", 0) or 0)
                + (getattr(resp.usage, "output_tokens", 0) or 0)
            ),
        }

        meta = {
            "model": resp.model,
            "usage": usage,
            "thoughts": None,
            "thought_signatures": None,
        }

        return content_text, meta

    # ---------- main generate_one ----------
    def generate_one(
        self,
        messages: List[Dict[str, str]],
        task: Task,
        run_config: Dict[str, Any],
    ) -> Any:
        if self.is_claude:
            return self._generate_one_claude(messages, task, run_config)

        # Convert OpenAI-style messages to Gemini-style inputs
        system_instruction, contents = self._messages_to_gemini(messages)

        # Build base config from sampling + provider_overrides
        cfg = self._build_generate_config()

        # Structured output from task (if supported for this model)
        rf = task.get_response_format("vertex", run_config)
        if isinstance(rf, dict):
            # Only fill in if not already set explicitly in cfg.
            mime = rf.get("response_mime_type") or rf.get("mime_type")
            if mime and "response_mime_type" not in cfg:
                cfg["response_mime_type"] = mime

            resp_schema = rf.get("response_schema") or rf.get("schema")
            if (
                resp_schema is not None
                and "response_schema" not in cfg
                and "response_json_schema" not in cfg
            ):
                # Sanitize JSON schema for Vertex expectations (e.g. no non-string enums).
                cfg["response_schema"] = _sanitize_schema_for_vertex(resp_schema)

        # System instruction from messages (if any) wins unless user override already set it.
        if system_instruction and "system_instruction" not in cfg:
            cfg["system_instruction"] = system_instruction

        # /models/{model}:generateContent via google.genai
        # The Python SDK accepts dicts for config.
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=cfg,
        )

        # ----------------------------------------------------
        # Visible primary text output
        # ----------------------------------------------------
        try:
            content_text = resp.text  # convenience property
        except Exception:
            # Fallback: join text parts if .text is missing
            content_text = ""
            try:
                parts = getattr(resp, "parts", []) or []
                chunks: List[str] = []
                for p in parts:
                    if getattr(p, "text", None) is not None:
                        chunks.append(p.text)
                if chunks:
                    content_text = "".join(chunks)
            except Exception:
                content_text = ""

        # ----------------------------------------------------
        # Reasoning / thoughts extraction (if include_thoughts=True and model supports it)
        # ----------------------------------------------------
        thoughts_text: Optional[str] = None
        thought_signatures: List[str] = []

        try:
            # In python-genai, resp.parts is a flattened view of the first candidate’s content.parts.
            parts = getattr(resp, "parts", []) or []
            thought_chunks: List[str] = []

            for p in parts:
                # Part.thought is a bool that marks this part as "thought" (reasoning). ([firebase.google.com](https://firebase.google.com/docs/reference/ai-logic/rest/v1beta/GenerateContentResponse?utm_source=openai))
                if getattr(p, "thought", False):
                    txt = getattr(p, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        thought_chunks.append(txt)
                    sig = getattr(p, "thought_signature", None)
                    if isinstance(sig, str):
                        thought_signatures.append(sig)

            if thought_chunks:
                # Join with newlines so you can inspect the full reasoning trace.
                thoughts_text = "\n".join(thought_chunks)
        except Exception:
            # If anything goes wrong, we just omit thoughts rather than failing the call.
            thoughts_text = None
            thought_signatures = []

        # ----------------------------------------------------
        # Usage metadata (prompt_token_count, candidates_token_count, thoughts_token_count, total_token_count)
        # ----------------------------------------------------
        usage = None
        try:
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                prompt_toks = getattr(um, "prompt_token_count", None)
                completion_toks = getattr(um, "candidates_token_count", None)
                total_toks = getattr(um, "total_token_count", None)
                thoughts_toks = getattr(um, "thoughts_token_count", None)

                usage = {
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": completion_toks,
                    "total_tokens": total_toks,
                    # Reasoning token usage, if thinking is enabled. ([googleapis.github.io](https://googleapis.github.io/python-genai/index.html))
                    "thoughts_tokens": thoughts_toks,
                    # Optional raw view if you want it:
                    "raw_usage_metadata": {
                        "prompt_token_count": prompt_toks,
                        "candidates_token_count": completion_toks,
                        "total_token_count": total_toks,
                        "thoughts_token_count": thoughts_toks,
                    },
                }
        except Exception:
            usage = None

        meta = {
            "model": self.model,
            "usage": usage,
            # Unified way for higher-level tasks to inspect reasoning:
            # - StoryIterativeRefineTask can read this as author_v2_thoughts.
            "thoughts": thoughts_text,
            # Thought signatures if you ever want to pass them back for continued reasoning.
            "thought_signatures": thought_signatures or None,
        }

        return content_text, meta