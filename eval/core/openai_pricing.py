# core/openai_pricing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Mapping, Union


@dataclass(frozen=True)
class TokenPrices:
    # USD per 1M tokens
    input_per_1m: float
    output_per_1m: float


# NOTE:
# Keep this table up to date with OpenAI public pricing.
# Keys should be "base model names"; matching allows date-suffixed variants.
#
# If a model has priority pricing and you know it, add "priority".
# If priority is requested but unknown, we fall back to "default".

# OPENAI_PRICE_TABLE: USD per 1M tokens (text tokens)
# tiers: batch | flex | default | priority
# fields: input_per_1m, cached_input_per_1m (None if not offered), output_per_1m
OPENAI_PRICE_TABLE = {
    "gpt-5.2": {
        "batch":    {"input_per_1m": 0.875, "cached_input_per_1m": 0.0875, "output_per_1m": 7.00},
        "flex":     {"input_per_1m": 0.875, "cached_input_per_1m": 0.0875, "output_per_1m": 7.00},
        "default": {"input_per_1m": 1.75,  "cached_input_per_1m": 0.175,  "output_per_1m": 14.00},
        "priority": {"input_per_1m": 3.50,  "cached_input_per_1m": 0.35,   "output_per_1m": 28.00},
    },
    "gpt-5.1": {
        "batch":    {"input_per_1m": 0.625, "cached_input_per_1m": 0.0625, "output_per_1m": 5.00},
        "flex":     {"input_per_1m": 0.625, "cached_input_per_1m": 0.0625, "output_per_1m": 5.00},
        "default": {"input_per_1m": 1.25,  "cached_input_per_1m": 0.125,  "output_per_1m": 10.00},
        "priority": {"input_per_1m": 2.50,  "cached_input_per_1m": 0.25,   "output_per_1m": 20.00},
    },
    "gpt-5": {
        "batch":    {"input_per_1m": 0.625, "cached_input_per_1m": 0.0625, "output_per_1m": 5.00},
        "flex":     {"input_per_1m": 0.625, "cached_input_per_1m": 0.0625, "output_per_1m": 5.00},
        "default": {"input_per_1m": 1.25,  "cached_input_per_1m": 0.125,  "output_per_1m": 10.00},
        "priority": {"input_per_1m": 2.50,  "cached_input_per_1m": 0.25,   "output_per_1m": 20.00},
    },
    "gpt-5-mini": {
        "batch":    {"input_per_1m": 0.125, "cached_input_per_1m": 0.0125, "output_per_1m": 1.00},
        "flex":     {"input_per_1m": 0.125, "cached_input_per_1m": 0.0125, "output_per_1m": 1.00},
        "default": {"input_per_1m": 0.25,  "cached_input_per_1m": 0.025,  "output_per_1m": 2.00},
        "priority": {"input_per_1m": 0.45,  "cached_input_per_1m": 0.045,  "output_per_1m": 3.60},
    },
    "gpt-5-nano": {
        "batch":    {"input_per_1m": 0.025, "cached_input_per_1m": 0.0025, "output_per_1m": 0.20},
        "flex":     {"input_per_1m": 0.025, "cached_input_per_1m": 0.0025, "output_per_1m": 0.20},
        "default": {"input_per_1m": 0.05,  "cached_input_per_1m": 0.005,  "output_per_1m": 0.40},
        # not listed under Priority in the table
    },

    "gpt-5.2-pro": {
        "batch":    {"input_per_1m": 10.50, "cached_input_per_1m": None, "output_per_1m": 84.00},
        "default": {"input_per_1m": 21.00, "cached_input_per_1m": None, "output_per_1m": 168.00},
    },
    "gpt-5-pro": {
        "batch":    {"input_per_1m": 7.50,  "cached_input_per_1m": None, "output_per_1m": 60.00},
        "default": {"input_per_1m": 15.00, "cached_input_per_1m": None, "output_per_1m": 120.00},
    },

    "gpt-4.1": {
        "batch":    {"input_per_1m": 1.00, "cached_input_per_1m": None,  "output_per_1m": 4.00},
        "default": {"input_per_1m": 2.00, "cached_input_per_1m": 0.50,  "output_per_1m": 8.00},
        "priority": {"input_per_1m": 3.50, "cached_input_per_1m": 0.875, "output_per_1m": 14.00},
    },
    "gpt-4.1-mini": {
        "batch":    {"input_per_1m": 0.20, "cached_input_per_1m": None,  "output_per_1m": 0.80},
        "default": {"input_per_1m": 0.40, "cached_input_per_1m": 0.10,  "output_per_1m": 1.60},
        "priority": {"input_per_1m": 0.70, "cached_input_per_1m": 0.175, "output_per_1m": 2.80},
    },
    "gpt-4.1-nano": {
        "batch":    {"input_per_1m": 0.05, "cached_input_per_1m": None,  "output_per_1m": 0.20},
        "default": {"input_per_1m": 0.10, "cached_input_per_1m": 0.025, "output_per_1m": 0.40},
        "priority": {"input_per_1m": 0.20, "cached_input_per_1m": 0.05,  "output_per_1m": 0.80},
    },

    "gpt-4o": {
        "batch":    {"input_per_1m": 1.25, "cached_input_per_1m": None,  "output_per_1m": 5.00},
        "default": {"input_per_1m": 2.50, "cached_input_per_1m": 1.25,  "output_per_1m": 10.00},
        "priority": {"input_per_1m": 4.25, "cached_input_per_1m": 2.125, "output_per_1m": 17.00},
    },
    "gpt-4o-mini": {
        "batch":    {"input_per_1m": 0.075, "cached_input_per_1m": None,  "output_per_1m": 0.30},
        "default": {"input_per_1m": 0.15,  "cached_input_per_1m": 0.075, "output_per_1m": 0.60},
        "priority": {"input_per_1m": 0.25,  "cached_input_per_1m": 0.125, "output_per_1m": 1.00},
    },

    # Specific dated variant explicitly listed on the page
    "gpt-4o-2024-05-13": {
        "batch":    {"input_per_1m": 2.50, "cached_input_per_1m": None, "output_per_1m": 7.50},
        "default": {"input_per_1m": 5.00, "cached_input_per_1m": None, "output_per_1m": 15.00},
        "priority": {"input_per_1m": 8.75, "cached_input_per_1m": None, "output_per_1m": 26.25},
    },

    "o1": {
        "batch":    {"input_per_1m": 7.50,  "cached_input_per_1m": None, "output_per_1m": 30.00},
        "default": {"input_per_1m": 15.00, "cached_input_per_1m": 7.50, "output_per_1m": 60.00},
    },
    "o1-pro": {
        "batch":    {"input_per_1m": 75.00,  "cached_input_per_1m": None, "output_per_1m": 300.00},
        "default": {"input_per_1m": 150.00, "cached_input_per_1m": None, "output_per_1m": 600.00},
    },

    "o3": {
        "batch":    {"input_per_1m": 1.00, "cached_input_per_1m": None,  "output_per_1m": 4.00},
        "flex":     {"input_per_1m": 1.00, "cached_input_per_1m": 0.25,  "output_per_1m": 4.00},
        "default": {"input_per_1m": 2.00, "cached_input_per_1m": 0.50,  "output_per_1m": 8.00},
        "priority": {"input_per_1m": 3.50, "cached_input_per_1m": 0.875, "output_per_1m": 14.00},
    },
    "o3-pro": {
        "batch":    {"input_per_1m": 10.00, "cached_input_per_1m": None, "output_per_1m": 40.00},
        "default": {"input_per_1m": 20.00, "cached_input_per_1m": None, "output_per_1m": 80.00},
    },
    "o3-deep-research": {
        "batch":    {"input_per_1m": 5.00,  "cached_input_per_1m": None, "output_per_1m": 20.00},
        "default": {"input_per_1m": 10.00, "cached_input_per_1m": 2.50, "output_per_1m": 40.00},
    },

    "o4-mini": {
        "batch":    {"input_per_1m": 0.55, "cached_input_per_1m": None,  "output_per_1m": 2.20},
        "flex":     {"input_per_1m": 0.55, "cached_input_per_1m": 0.138, "output_per_1m": 2.20},
        "default": {"input_per_1m": 1.10, "cached_input_per_1m": 0.275, "output_per_1m": 4.40},
        "priority": {"input_per_1m": 2.00, "cached_input_per_1m": 0.50,  "output_per_1m": 8.00},
    },
    "o4-mini-deep-research": {
        "batch":    {"input_per_1m": 1.00, "cached_input_per_1m": None, "output_per_1m": 4.00},
        "default": {"input_per_1m": 2.00, "cached_input_per_1m": 0.50, "output_per_1m": 8.00},
    },

    "o3-mini": {
        "batch":    {"input_per_1m": 0.55, "cached_input_per_1m": None, "output_per_1m": 2.20},
        "default": {"input_per_1m": 1.10, "cached_input_per_1m": 0.55, "output_per_1m": 4.40},
    },
    "o1-mini": {
        "batch":    {"input_per_1m": 0.55, "cached_input_per_1m": None, "output_per_1m": 2.20},
        "default": {"input_per_1m": 1.10, "cached_input_per_1m": 0.55, "output_per_1m": 4.40},
    },

    "computer-use-preview": {
        "batch":    {"input_per_1m": 1.50, "cached_input_per_1m": None, "output_per_1m": 6.00},
        "default": {"input_per_1m": 3.00, "cached_input_per_1m": None, "output_per_1m": 12.00},
    },

    # Models listed under Standard (text tokens) only
    "gpt-5.2-chat-latest":   {"default": {"input_per_1m": 1.75, "cached_input_per_1m": 0.175, "output_per_1m": 14.00}},
    "gpt-5.1-chat-latest":   {"default": {"input_per_1m": 1.25, "cached_input_per_1m": 0.125, "output_per_1m": 10.00}},
    "gpt-5-chat-latest":     {"default": {"input_per_1m": 1.25, "cached_input_per_1m": 0.125, "output_per_1m": 10.00}},
    "gpt-5.1-codex-max":     {"default": {"input_per_1m": 1.25, "cached_input_per_1m": 0.125, "output_per_1m": 10.00},
                              "priority": {"input_per_1m": 2.50, "cached_input_per_1m": 0.25,  "output_per_1m": 20.00}},
    "gpt-5.1-codex":         {"default": {"input_per_1m": 1.25, "cached_input_per_1m": 0.125, "output_per_1m": 10.00},
                              "priority": {"input_per_1m": 2.50, "cached_input_per_1m": 0.25,  "output_per_1m": 20.00}},
    "gpt-5-codex":           {"default": {"input_per_1m": 1.25, "cached_input_per_1m": 0.125, "output_per_1m": 10.00},
                              "priority": {"input_per_1m": 2.50, "cached_input_per_1m": 0.25,  "output_per_1m": 20.00}},
    "gpt-5.1-codex-mini":    {"default": {"input_per_1m": 0.25, "cached_input_per_1m": 0.025, "output_per_1m": 2.00}},
    "codex-mini-latest":     {"default": {"input_per_1m": 1.50, "cached_input_per_1m": 0.375, "output_per_1m": 6.00}},
    "gpt-5-search-api":      {"default": {"input_per_1m": 1.25, "cached_input_per_1m": 0.125, "output_per_1m": 10.00}},
    "gpt-4o-mini-search-preview": {"default": {"input_per_1m": 0.15, "cached_input_per_1m": None, "output_per_1m": 0.60}},
    "gpt-4o-search-preview":      {"default": {"input_per_1m": 2.50, "cached_input_per_1m": None, "output_per_1m": 10.00}},

    # Realtime/audo/image entries appear in the same "Text tokens" Standard table on the page;
    # include them here only if you actually bill them as "text tokens" in your pipeline.
    "gpt-realtime":          {"default": {"input_per_1m": 4.00, "cached_input_per_1m": 0.40, "output_per_1m": 16.00}},
    "gpt-realtime-mini":     {"default": {"input_per_1m": 0.60, "cached_input_per_1m": 0.06, "output_per_1m": 2.40}},
    "gpt-4o-realtime-preview":      {"default": {"input_per_1m": 5.00, "cached_input_per_1m": 2.50, "output_per_1m": 20.00}},
    "gpt-4o-mini-realtime-preview": {"default": {"input_per_1m": 0.60, "cached_input_per_1m": 0.30, "output_per_1m": 2.40}},
    "gpt-audio":            {"default": {"input_per_1m": 2.50, "cached_input_per_1m": None, "output_per_1m": 10.00}},
    "gpt-audio-mini":       {"default": {"input_per_1m": 0.60, "cached_input_per_1m": None, "output_per_1m": 2.40}},
    "gpt-4o-audio-preview":       {"default": {"input_per_1m": 2.50, "cached_input_per_1m": None, "output_per_1m": 10.00}},
    "gpt-4o-mini-audio-preview":  {"default": {"input_per_1m": 0.15, "cached_input_per_1m": None, "output_per_1m": 0.60}},
    "gpt-image-1.5":        {"default": {"input_per_1m": 5.00, "cached_input_per_1m": 1.25, "output_per_1m": 10.00}},
    "chatgpt-image-latest": {"default": {"input_per_1m": 5.00, "cached_input_per_1m": 1.25, "output_per_1m": 10.00}},
    "gpt-image-1":          {"default": {"input_per_1m": 5.00, "cached_input_per_1m": 1.25, "output_per_1m": None}},
    "gpt-image-1-mini":     {"default": {"input_per_1m": 2.00, "cached_input_per_1m": 0.20, "output_per_1m": None}},
}


def _normalize_model_name(model: str) -> str:
    return (model or "").strip()

def resolve_openai_prices(
    model: str,
    service_tier: str = "default",
    pricing_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[TokenPrices], str]:
    """
    Returns (TokenPrices or None, source_str).

    Supports overrides like:
      pricing_overrides = {
        "gpt-4o": {"default": {"input_per_1m": 5, "output_per_1m": 15}},
        "default": {"default": {"input_per_1m": 1.0, "output_per_1m": 1.0}}
      }
    """
    tier = (service_tier or "default").strip().lower()
    model_norm = _normalize_model_name(model)

    # 1) Overrides
    if isinstance(pricing_overrides, dict):
        # exact or prefix (base name)
        for key in [model_norm, *[k for k in pricing_overrides.keys() if model_norm.startswith(k + "-")]]:
            if key in pricing_overrides:
                per_model = pricing_overrides[key] or {}
                if tier in per_model:
                    t = per_model[tier] or {}
                    try:
                        return TokenPrices(
                            input_per_1m=float(t["input_per_1m"]),
                            output_per_1m=float(t["output_per_1m"]),
                        ), f"override:{key}:{tier}"
                    except Exception:
                        pass

        # default override
        if "default" in pricing_overrides:
            per_def = pricing_overrides["default"] or {}
            if tier in per_def:
                t = per_def[tier] or {}
                try:
                    return TokenPrices(
                        input_per_1m=float(t["input_per_1m"]),
                        output_per_1m=float(t["output_per_1m"]),
                    ), f"override:default:{tier}"
                except Exception:
                    pass
    
    # 2) Table: exact match
    if model_norm in OPENAI_PRICE_TABLE:
        per = OPENAI_PRICE_TABLE[model_norm]
        if tier in per:
            return per[tier], f"table:{model_norm}:{tier}"
        if "default" in per:
            return per["default"], f"table:{model_norm}:default(fallback)"

    # 3) Table: prefix match for dated variants (e.g., gpt-4o-2024-08-06)
    #    Use the *longest* matching base name so that "gpt-5-mini-2025-01-10"
    #    resolves to "gpt-5-mini", not just "gpt-5".
    candidates: list[tuple[str, Dict[str, Any]]] = []
    for base, per in OPENAI_PRICE_TABLE.items():
        if model_norm.startswith(base + "-"):
            candidates.append((base, per))

    if candidates:
        # pick the most specific (longest) base name
        base, per = max(candidates, key=lambda bp: len(bp[0]))
        if tier in per:
            return per[tier], f"table:{base}:{tier}"
        if "default" in per:
            return per["default"], f"table:{base}:default(fallback)"

    return None, "unknown"

def estimate_openai_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    prices: Union["TokenPrices", Mapping[str, Any]],
) -> float:
    if isinstance(prices, dict):
        input_per_1m = float(prices["input_per_1m"])
        output_per_1m = float(prices["output_per_1m"])
    else:
        input_per_1m = float(prices.input_per_1m)
        output_per_1m = float(prices.output_per_1m)

    return (prompt_tokens / 1_000_000.0) * input_per_1m + (completion_tokens / 1_000_000.0) * output_per_1m