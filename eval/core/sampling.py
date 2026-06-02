# core/sampling.py
from __future__ import annotations
from typing import Any, Dict
import inspect

try:
    from vllm import SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def get_engine_cfg(run_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience accessor. Expects 'engine' at the top level of run_config.
    """
    return run_config.get("engine", {}) or {}


def _merge_sampling_for_provider(engine_cfg: Dict[str, Any], provider_key: str) -> Dict[str, Any]:
    """
    Merge generic sampling config and provider-specific overrides for a given provider_key,
    e.g., 'hf', 'vllm', 'openai'.
    """
    base = engine_cfg.get("sampling", {}) or {}
    overrides = (engine_cfg.get("provider_overrides", {}) or {}).get(provider_key, {}) or {}
    merged: Dict[str, Any] = {**base, **overrides}
    return merged


# ---------- HF helpers ----------

def build_hf_generate_kwargs(engine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build kwargs for transformers' .generate().

    Behavior:
      - Merge engine_cfg["sampling"] with any HF-specific overrides in
        engine_cfg["provider_overrides"]["hf"]["sampling"] (if present),
        with provider-specific overrides winning.
      - Seed with HF-like defaults for core knobs (temperature, top_p, top_k, etc.),
        so that even when you don't specify them in config, they are explicit.
      - Normalize "max_tokens" -> "max_new_tokens" for convenience, so that
        configs that use the vLLM-style "max_tokens" still work for HF.

    We do NOT try to enumerate all generation parameters; we let unknown keys
    flow through and rely on HF to ignore/handle them via GenerationConfig.
    """
    # Base merged sampling config
    merged: Dict[str, Any] = _merge_sampling_for_provider(engine_cfg, provider_key="hf")  # type: ignore

    # HF GenerationConfig defaults for key knobs (as of v4.45.x)
    defaults: Dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "num_return_sequences": 1,
        "early_stopping": False,
        # We intentionally do NOT set max_new_tokens here; HF will derive
        # generation length from max_length/generation_config if none given.
    }

    # Start with defaults, then overlay merged user config
    kwargs: Dict[str, Any] = {**defaults, **merged}

    # Alias: max_tokens -> max_new_tokens, if user didn't specify max_new_tokens explicitly
    if "max_new_tokens" not in kwargs and "max_tokens" in kwargs:
        kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

    # You could optionally normalize stop_sequences->stopping_criteria, but HF
    # does not have a direct stop_sequences arg; users should set HF-native
    # stopping criteria instead. We leave any extra keys in kwargs; HF will
    # ignore those that don't match the generate() signature or GenerationConfig.

    return kwargs


# ---------- vLLM helpers ----------
def build_vllm_sampling_params(engine_cfg: Dict[str, Any]) -> SamplingParams:
    """
    Build vLLM SamplingParams from engine_cfg["sampling"], supporting all
    documented fields from vLLM 0.6.x.

    We read from:
      - engine_cfg["sampling"]
      - engine_cfg["provider_overrides"]["vllm"]["sampling"] (if present),
    with provider-specific overrides winning.

    Fields supported (matching vLLM 0.6.x SamplingParams signature):

      n: int = 1
      best_of: int | None = None
      presence_penalty: float = 0.0
      frequency_penalty: float = 0.0
      repetition_penalty: float = 1.0
      temperature: float = 1.0
      top_p: float = 1.0
      top_k: int = -1
      min_p: float = 0.0
      seed: int | None = None
      use_beam_search: bool = False
      length_penalty: float = 1.0
      early_stopping: bool | str = False
      stop: str | list[str] | None = None
      stop_token_ids: list[int] | None = None
      ignore_eos: bool = False
      max_tokens: int | None = 16
      min_tokens: int = 0
      logprobs: int | None = None
      prompt_logprobs: int | None = None
      detokenize: bool = True
      skip_special_tokens: bool = True
      spaces_between_special_tokens: bool = True
      logits_processors: Any | None = None
      include_stop_str_in_output: bool = False
      truncate_prompt_tokens: int | None = None
      output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
      output_text_buffer_length: int = 0

    Any field not provided in config will be left at its vLLM default.
    """
    # Merge generic sampling + vLLM-specific overrides
    merged: Dict[str, Any] = _merge_sampling_for_provider(engine_cfg, provider_key="vllm")  # type: ignore

    kwargs: Dict[str, Any] = {}

    # ---- max_tokens / max_new_tokens alias ----
    if "max_tokens" in merged and merged["max_tokens"] is not None:
        kwargs["max_tokens"] = merged["max_tokens"]
    elif "max_new_tokens" in merged and merged["max_new_tokens"] is not None:
        kwargs["max_tokens"] = merged["max_new_tokens"]
    # If neither is set, we do not pass max_tokens and let vLLM use its default (16).

    # ---- Core fields ----
    for key in [
        "n",
        "best_of",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "use_beam_search",
        "length_penalty",
        "early_stopping",
        "ignore_eos",
        "min_tokens",
        "logprobs",
        "prompt_logprobs",
        "detokenize",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "logits_processors",
        "include_stop_str_in_output",
        "truncate_prompt_tokens",
        "output_kind",
        "output_text_buffer_length",
    ]:
        if key in merged and merged[key] is not None:
            kwargs[key] = merged[key]

    # ---- stop / stop_token_ids ----
    if "stop" in merged and merged["stop"] is not None:
        kwargs["stop"] = merged["stop"]
    elif "stop_sequences" in merged and merged["stop_sequences"] is not None:
        # allow alias stop_sequences
        kwargs["stop"] = merged["stop_sequences"]

    if "stop_token_ids" in merged and merged["stop_token_ids"] is not None:
        kwargs["stop_token_ids"] = merged["stop_token_ids"]

    # Construct SamplingParams with only the keys the user actually set;
    # all other fields use vLLM's built-in defaults.
    return SamplingParams(**kwargs)

# ---------- OpenAI helpers ----------

def build_openai_params(
    engine_cfg: Dict[str, Any],
    task: Any,
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build kwargs for OpenAI chat.completions.create (or batch body) from engine_cfg.

    Precedence:
      - Start from engine_cfg["sampling"].
      - Overlay engine_cfg["provider_overrides"]["openai"].
      - Overlay engine_cfg["provider_params"] (for OpenAI-only fields).
      - Set/override "response_format" by:
          engine_cfg["provider_params"]["response_format"]
        OR task.get_response_format("openai", run_config)
      - Alias 'max_new_tokens' -> 'max_tokens' if needed.

    We DO NOT include 'model' here; caller must set it explicitly.
    """
    sampling = engine_cfg.get("sampling", {}) or {}
    overrides = (engine_cfg.get("provider_overrides", {}) or {}).get("openai", {}) or {}
    provider_params = engine_cfg.get("provider_params", {}) or {}

    merged: Dict[str, Any] = {**sampling, **overrides, **provider_params}

    # response_format precedence: explicit in provider_params > task default
    rf = provider_params.get("response_format") or task.get_response_format("openai", run_config)
    if rf:
        merged["response_format"] = rf

    # Alias: many people use 'max_new_tokens' (HF-style) but OpenAI wants 'max_tokens'
    if "max_new_tokens" in merged and "max_tokens" not in merged:
        merged["max_tokens"] = merged["max_new_tokens"]

    # Ensure we don't accidentally overwrite 'model' at call sites
    if "model" in merged:
        merged.pop("model")

    return merged