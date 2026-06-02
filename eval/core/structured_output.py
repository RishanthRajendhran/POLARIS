# core/structured_output.py
from __future__ import annotations
from typing import Any, Dict

# Gemini models on Vertex that support responseSchema for structured output.
# Base-name prefixes; we accept dated/preview suffixed variants.
_VERTEX_GEMINI_STRUCTURED_BASES = [
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def _supports_openai_json_schema(model: str) -> bool:
    """
    All modern OpenAI chat models in your price table support response_format
    with json_schema (gpt-4.1*, gpt-4o*, gpt-5*, o*).
    If you ever add a truly legacy model, you can refine this.
    """
    m = (model or "").strip()
    if not m:
        return False
    # Cheap heuristics: all your configured OpenAI models use these prefixes.
    return m.startswith(("gpt-4", "gpt-5", "o"))


def _supports_vertex_schema(model: str) -> bool:
    """
    True if the Vertex model name looks like a Gemini text model that supports
    responseSchema according to Vertex docs.
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    return any(m.startswith(base) for base in _VERTEX_GEMINI_STRUCTURED_BASES)


def supports_native_schema(provider: str, run_config: Dict[str, Any]) -> bool:
    """
    Return True if this (provider, model) pair should use native structured
    output (JSON Schema / responseSchema) instead of prompt-only.

    This is intentionally conservative: anything unknown returns False.
    """
    engine_cfg = (run_config.get("engine") or {}) if isinstance(run_config, dict) else {}
    model = engine_cfg.get("model", "")

    if provider == "openai":
        return _supports_openai_json_schema(model)
    if provider == "vertex":
        return _supports_vertex_schema(model)

    # For HF, vLLM, openrouter, etc., assume no native schema.
    return False