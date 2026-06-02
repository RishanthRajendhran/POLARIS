# providers/openai_sync.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

from openai import OpenAI  # pip install openai

from core.engine import Engine
from core.task_base import Task
from core.sampling import get_engine_cfg
from core.openai_params import build_openai_chat_body_params


class OpenAIChatEngine(Engine):
    """
    OpenAI engine backed by the Responses API.

    - Accepts chat-style `messages` from tasks.
    - Uses client.responses.create(...) under the hood.
    - Supports:
        * sampling params from build_openai_chat_body_params(...)
        * structured outputs (task.get_response_format("openai", ...) with json_schema)
        * reasoning summaries for reasoning models (via `reasoning.summary` items)
        * usage metadata (prompt / completion / total tokens)
    """

    name = "openai"

    def __init__(self, run_config: Dict[str, Any]):
        engine_cfg = get_engine_cfg(run_config)
        api_key = engine_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not set. Provide engine.api_key or OPENAI_API_KEY env var.")
        self.client = OpenAI(api_key=api_key)
        self.engine_cfg = engine_cfg
        model = engine_cfg.get("model")
        if not model:
            raise ValueError("engine.model must be set for OpenAIChatEngine.")

    # ---------------- helpers ----------------

    @staticmethod
    def _extract_output_text(resp: Any) -> str:
        """
        Best-effort extraction of the main assistant text from a Responses API
        object.

        Order:
          1) resp.output_text (SDK convenience)
          2) first message item in resp.output with text parts
        """
        # 1) Try SDK convenience
        try:
            txt = getattr(resp, "output_text", None)
            if isinstance(txt, str) and txt.strip():
                return txt
        except Exception:
            pass

        # 2) Walk output items
        try:
            output_items = getattr(resp, "output", None) or []
            for item in output_items:
                # ResponseOutputMessage type
                if getattr(item, "type", None) == "message":
                    parts = getattr(item, "content", None) or []
                    chunks: List[str] = []
                    for p in parts:
                        p_type = getattr(p, "type", None)
                        if p_type in ("output_text", "input_text", "text"):
                            t = getattr(p, "text", None)
                            if isinstance(t, str):
                                chunks.append(t)
                    if chunks:
                        return "".join(chunks)
        except Exception:
            pass

        return ""

    @staticmethod
    def _extract_usage(resp: Any) -> Optional[Dict[str, Any]]:
        """
        Normalize usage metadata from Responses API object into:
          {
            "prompt_tokens": ...,
            "completion_tokens": ...,
            "total_tokens": ...,
            "raw_usage": {...}
          }
        """
        try:
            u = getattr(resp, "usage", None) or getattr(resp, "usage_metadata", None)
            if u is None:
                return None
            d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
            prompt = d.get("input_tokens") or d.get("prompt_tokens")
            completion = d.get("output_tokens") or d.get("completion_tokens")
            total = d.get("total_tokens") or (
                (prompt or 0) + (completion or 0)
            )
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "raw_usage": d,
            }
        except Exception:
            return None

    @staticmethod
    def _extract_reasoning_summary(resp: Any) -> Optional[str]:
        """
        Extract the first reasoning summary text from a Responses API object,
        if present. See official reasoning docs: reasoning items with `summary`.
        """
        try:
            output_items = getattr(resp, "output", None) or []
            for item in output_items:
                if getattr(item, "type", None) == "reasoning":
                    summaries = getattr(item, "summary", None) or []
                    if summaries:
                        t = getattr(summaries[0], "text", None)
                        if isinstance(t, str):
                            return t
        except Exception:
            pass
        return None

    # ---------------- main call ----------------
    def generate_one(
        self,
        messages: List[Dict[str, str]],
        task: Task,
        run_config: Dict[str, Any],
    ) -> Any:
        """
        Use the Responses API instead of Chat Completions.

        - `messages` are passed as `input`.
        - Structured-output schemas from task.get_response_format("openai", ...) are
        mapped to `text.format` as required by the Responses API.
        - Chat-only fields from build_openai_chat_body_params (e.g. messages,
        response_format, reasoning_effort) are stripped or remapped.
        """
        model = self.engine_cfg["model"]

        # Base params: sampling, etc. (originally designed for chat)
        params = build_openai_chat_body_params(run_config, task) or {}
        params["model"] = model

        # ---- Strip chat-specific fields from params ----
        # They are not valid top-level kwargs for Responses.create().
        params.pop("messages", None)          # we'll pass our own `input`
        params.pop("response_format", None)   # Responses uses text.format instead

        # Grab and remove reasoning_effort / verbosity if present; Responses expects
        # them nested under `reasoning`.
        reasoning_effort = params.pop("reasoning_effort", None)
        verbosity = params.pop("verbosity", None)  # not used by Responses; safe to drop

        # ---- Structured outputs: map response_format -> text.format ----
        # Responses API expects text.format = {type, name, strict, schema}
        # (flattened), NOT nested under a "json_schema" key.
        rf = task.get_response_format("openai", run_config)
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            schema_wrapper = rf.get("json_schema")
            if schema_wrapper:
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        **schema_wrapper,
                    }
                }

        # ---- Reasoning config: map reasoning_effort / reasoning_summary -> reasoning ----
        reasoning_cfg: Dict[str, Any] = {}

        # Prefer explicit per-call value from params (if builder put it there),
        # otherwise fall back to engine_cfg.
        if reasoning_effort is None:
            reasoning_effort = self.engine_cfg.get("reasoning_effort")
        if reasoning_effort:
            reasoning_cfg["effort"] = reasoning_effort  # low | medium | high

        summary_mode = (
            self.engine_cfg.get("reasoning_summary")
            or ((self.engine_cfg.get("provider_overrides", {}) or {}).get("openai", {}) or {}).get("reasoning_summary")
        )
        if summary_mode:
            reasoning_cfg["summary"] = summary_mode  # e.g. "auto" | "concise" | "detailed"

        if reasoning_cfg:
            params["reasoning"] = reasoning_cfg

        # ---- Token limits: map chat-style names -> max_output_tokens ----
        if "max_completion_tokens" in params:
            params["max_output_tokens"] = params.pop("max_completion_tokens")
        elif "max_tokens" in params:
            params["max_output_tokens"] = params.pop("max_tokens")

        # ---- Main call: /v1/responses ----
        # Per migration guide, you can pass the same messages array as `input`.
        params["input"] = messages

        # This is where the error was: `reasoning_effort` was still in params,
        # causing Responses.create(...) to reject it. Now it's removed and moved
        # into params["reasoning"]["effort"] instead.
        resp = self.client.responses.create(**params)

        # ... keep your existing extraction code here ...
        content = self._extract_output_text(resp)
        usage = self._extract_usage(resp)
        reasoning_summary = self._extract_reasoning_summary(resp)

        meta = {
            "id": getattr(resp, "id", None),
            "model": getattr(resp, "model", None) or model,
            "status": getattr(resp, "status", None),
            "usage": usage,
            "reasoning_summary": reasoning_summary,
        }

        return content, meta