# core/openai_params.py
from __future__ import annotations
from typing import Any, Dict

from core.sampling import get_engine_cfg


def _get_sampling(engine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return engine_cfg.get("sampling", {}) or {}


def _get_reasoning_effort(engine_cfg: Dict[str, Any]) -> str:
    sampling = _get_sampling(engine_cfg)
    return (
        engine_cfg.get("reasoning_effort")
        or sampling.get("reasoning_effort")
        or "medium"
    )


def _get_verbosity(engine_cfg: Dict[str, Any]) -> str:
    sampling = _get_sampling(engine_cfg)
    return (
        engine_cfg.get("verbosity")
        or sampling.get("verbosity")
        or "medium"
    )


def _get_max_completion_tokens(engine_cfg: Dict[str, Any]) -> int:
    sampling = _get_sampling(engine_cfg)
    return (
        sampling.get("max_completion_tokens")
        or sampling.get("max_tokens")
        or sampling.get("max_new_tokens")
        or 512
    )


def _get_legacy_chat_params(engine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    sampling = _get_sampling(engine_cfg)
    temperature = sampling.get("temperature", 1.0)
    top_p = sampling.get("top_p", 1.0)
    max_tokens = int(_get_max_completion_tokens(engine_cfg))

    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }


def build_openai_chat_body_params(
    run_config: Dict[str, Any],
    task: Any,
) -> Dict[str, Any]:
    """
    Build the *model-specific* chat parameters for OpenAI /v1/chat/completions.

    We DO NOT include 'model' or 'messages' here; caller must set those.

    Behavior mirrors your original script:

      - gpt-5, gpt-5-mini, gpt-5-nano:
          * max_completion_tokens
          * reasoning_effort
          * verbosity
          * response_format (if any)

      - models starting with "o" (o3-mini, o4-mini, etc.):
          * max_completion_tokens
          * reasoning_effort
          * response_format

      - everything else (gpt-4.1, gpt-4.1-mini, gpt-4o, etc.):
          * temperature, top_p, max_tokens, frequency_penalty, presence_penalty
          * response_format
    """
    engine_cfg = get_engine_cfg(run_config)
    model = engine_cfg.get("model", "")
    if not model:
        raise ValueError("engine.model must be set for OpenAI chat.")

    # Determine response_format precedence:
    provider_params = engine_cfg.get("provider_params", {}) or {}
    response_format = provider_params.get("response_format") or task.get_response_format(
        provider="openai",
        run_config=run_config,
    )

    params: Dict[str, Any] = {}

    # gpt-5 family
    if model.startswith("gpt-5"):
        params["max_completion_tokens"] = int(_get_max_completion_tokens(engine_cfg))
        params["reasoning_effort"] = _get_reasoning_effort(engine_cfg)
        params["verbosity"] = _get_verbosity(engine_cfg)
        if response_format:
            params["response_format"] = response_format
        return params

    # "o*" family (o3-mini, o4-mini, etc.)
    if model.startswith("o"):
        params["max_completion_tokens"] = int(_get_max_completion_tokens(engine_cfg))
        params["reasoning_effort"] = _get_reasoning_effort(engine_cfg)
        if response_format:
            params["response_format"] = response_format
        return params

    # Legacy chat models (gpt-4.1, gpt-4.1-mini, gpt-4o, etc.)
    params.update(_get_legacy_chat_params(engine_cfg))
    if response_format:
        params["response_format"] = response_format
    return params