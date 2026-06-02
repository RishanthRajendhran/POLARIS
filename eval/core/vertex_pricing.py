# core/vertex_pricing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Mapping, Union


@dataclass(frozen=True)
class TokenPrices:
    # USD per 1M tokens
    input_per_1m: float
    output_per_1m: float

# VERTEX_PRICE_TABLE:
#   base_model -> consumption_tier -> TokenPrices
#
# consumption_tier in: "standard", "priority", "flex"
#
# "standard"  -> Standard PayGo section for that model.
# "priority"  -> Priority PayGo section.
# "flex"      -> Flex/Batch section (Vertex uses same price for Flex online & Batch).
#
# All numbers are for <= 200K input tokens (short context); if your
# requests exceed that, you'd need a second table for long-context rates.

VERTEX_PRICE_TABLE: Dict[str, Dict[str, TokenPrices]] = {
    # =========================
    # Gemini 2.5 family
    # =========================
    # Source: Vertex AI Pricing – Standard, Priority, Flex/Batch sections for Gemini 2.5([nailsope.com](https://nailsope.com/?_=%2Fvertex-ai%2Fgenerative-ai%2Fpricing%23WLz4YT%2FfYOTTL7txvf4lFEWsonYjiiin0m67bf4%3D))

    # Gemini 2.5 Pro
    "gemini-2.5-pro": {
        # Standard PayGo (<=200K input tokens)
        "standard": TokenPrices(input_per_1m=1.25,  output_per_1m=10.00),
        # Priority PayGo
        "priority": TokenPrices(input_per_1m=2.25,  output_per_1m=18.00),
        # Flex/Batch (discounted)
        "flex":     TokenPrices(input_per_1m=0.625, output_per_1m=5.00),
    },

    # Gemini 2.5 Flash
    "gemini-2.5-flash": {
        # Standard
        "standard": TokenPrices(input_per_1m=0.30,  output_per_1m=2.50),
        # Priority
        "priority": TokenPrices(input_per_1m=0.54,  output_per_1m=4.50),
        # Flex/Batch
        "flex":     TokenPrices(input_per_1m=0.15,  output_per_1m=1.25),
    },

    # Gemini 2.5 Flash Lite
    "gemini-2.5-flash-lite": {
        # Standard
        "standard": TokenPrices(input_per_1m=0.10,  output_per_1m=0.40),
        # Priority
        "priority": TokenPrices(input_per_1m=0.18,  output_per_1m=0.72),
        # Flex/Batch
        "flex":     TokenPrices(input_per_1m=0.05,  output_per_1m=0.20),
    },

    # =========================
    # Gemini 2.0 family
    # =========================
    # Source: Vertex AI Pricing – Gemini 2.0 token-based pricing (Price vs Price with Batch API).([nailsope.com](https://nailsope.com/?_=%2Fvertex-ai%2Fgenerative-ai%2Fpricing%23WLz4YT%2FfYOTTL7txvf4lFEWsonYjiiin0m67bf4%3D))
    # Note: For 2.0 models, Google only publishes "Price" and "Price with Batch API";
    # there are no separate Priority or Flex tables. We treat:
    #   - standard = base "Price"
    #   - flex     = "Price with Batch API"
    #   - priority = same as standard (no distinct Priority tier published).

    # Gemini 2.0 Flash
    "gemini-2.0-flash": {
        "standard": TokenPrices(input_per_1m=0.15,   output_per_1m=0.60),
        "priority": TokenPrices(input_per_1m=0.15,   output_per_1m=0.60),  # no separate Priority pricing
        "flex":     TokenPrices(input_per_1m=0.075,  output_per_1m=0.30),  # Batch API price
    },

    # Gemini 2.0 Flash Lite
    "gemini-2.0-flash-lite": {
        "standard": TokenPrices(input_per_1m=0.075,  output_per_1m=0.30),
        "priority": TokenPrices(input_per_1m=0.075,  output_per_1m=0.30),  # no separate Priority pricing
        "flex":     TokenPrices(input_per_1m=0.0375, output_per_1m=0.15),  # Batch API price
    },

    # =========================
    # Gemini 3.0 family (Preview)
    # =========================
    # Source: Vertex AI Pricing – Gemini 3 Pro/Flash Preview: Standard, Priority, Flex/Batch.([nailsope.com](https://nailsope.com/?_=%2Fvertex-ai%2Fgenerative-ai%2Fpricing%23WLz4YT%2FfYOTTL7txvf4lFEWsonYjiiin0m67bf4%3D))

    # Gemini 3.0 Pro Preview
    "gemini-3-pro-preview": {
        # Standard
        "standard": TokenPrices(input_per_1m=2.00,  output_per_1m=12.00),
        # Priority
        "priority": TokenPrices(input_per_1m=3.60,  output_per_1m=21.60),
        # Flex/Batch
        "flex":     TokenPrices(input_per_1m=1.00,  output_per_1m=6.00),
    },

    # Gemini 3.1 Pro Preview
    "gemini-3.1-pro-preview": {
        # Standard
        "standard": TokenPrices(input_per_1m=2.00,  output_per_1m=12.00),
        # Priority
        "priority": TokenPrices(input_per_1m=3.60,  output_per_1m=21.60),
        # Flex/Batch
        "flex":     TokenPrices(input_per_1m=1.00,  output_per_1m=6.00),
    },

    # Gemini 3.0 Flash Preview
    "gemini-3-flash-preview": {
        # Standard
        "standard": TokenPrices(input_per_1m=0.50,  output_per_1m=3.00),
        # Priority
        "priority": TokenPrices(input_per_1m=0.90,  output_per_1m=5.40),
        # Flex/Batch
        "flex":     TokenPrices(input_per_1m=0.25,  output_per_1m=1.50),
    },
}


def _normalize_model_name(model: str) -> str:
    return (model or "").strip()


def resolve_vertex_prices(
    model: str,
    mode: str = "online",               # "online" | "batch"
    consumption: str = "standard",      # "standard" | "priority" | "flex"
    pricing_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[TokenPrices], str]:
    """
    Returns (TokenPrices or None, source_str) for Vertex/Gemini models.

    Dimensions:
      - mode:        "online" or "batch"
      - consumption: "standard", "priority", or "flex"

    Semantics:
      - For Standard / Priority online calls, use the corresponding section
        from the pricing table.
      - For Flex online calls, use Flex/Batch rates.
      - For Batch mode calls, Vertex documentation groups them under the
        Flex/Batch section; regardless of 'consumption', pricing is taken
        from the 'flex' tier.

    Overrides:
      pricing_overrides can override the built-in table, e.g.:

        {
          "gemini-2.5-flash": {
            "standard": {"input_per_1m": 0.35, "output_per_1m": 2.8},
            "flex":     {"input_per_1m": 0.18, "output_per_1m": 1.4}
          },
          "default": {
            "standard": {"input_per_1m": 1.0, "output_per_1m": 1.0}
          }
        }
    """
    mode_norm = (mode or "online").strip().lower()
    cons_req = (consumption or "standard").strip().lower()
    model_norm = _normalize_model_name(model)

    # Batch always uses Flex/Batch rates (per docs).
    # We keep 'consumption' and 'mode' distinct in the API, but for pricing
    # we must map batch -> flex tier.
    if mode_norm == "batch":
        cons_eff = "flex"
    else:
        cons_eff = cons_req if cons_req in ("standard", "priority", "flex") else "standard"

    # 1) Overrides
    if isinstance(pricing_overrides, dict):
        # Try exact or prefix match
        keys_to_try = [model_norm] + [k for k in pricing_overrides.keys() if model_norm.startswith(k + "-")]
        for key in keys_to_try:
            if key in pricing_overrides:
                per_model = pricing_overrides[key] or {}
                # Prefer effective consumption; fall back to standard/default
                if cons_eff in per_model:
                    t = per_model[cons_eff] or {}
                    try:
                        return TokenPrices(
                            input_per_1m=float(t["input_per_1m"]),
                            output_per_1m=float(t["output_per_1m"]),
                        ), f"override:{key}:{mode_norm}:{cons_eff}"
                    except Exception:
                        pass

        # Global default override
        if "default" in pricing_overrides:
            per_def = pricing_overrides["default"] or {}
            if cons_eff in per_def:
                t = per_def[cons_eff] or {}
                try:
                    return TokenPrices(
                        input_per_1m=float(t["input_per_1m"]),
                        output_per_1m=float(t["output_per_1m"]),
                    ), f"override:default:{mode_norm}:{cons_eff}"
                except Exception:
                    pass

    # 2) Table: exact base match
    if model_norm in VERTEX_PRICE_TABLE:
        per = VERTEX_PRICE_TABLE[model_norm]
        if cons_eff in per:
            return per[cons_eff], f"vertex_table:{model_norm}:{mode_norm}:{cons_eff}"
        if "standard" in per:
            return per["standard"], f"vertex_table:{model_norm}:{mode_norm}:standard(fallback)"

    # 3) Table: prefix match (e.g. gemini-3-flash-preview-001 -> gemini-3-flash-preview)
    candidates: list[tuple[str, Dict[str, TokenPrices]]] = []
    for base, per in VERTEX_PRICE_TABLE.items():
        if model_norm.startswith(base + "-"):
            candidates.append((base, per))

    if candidates:
        base, per = max(candidates, key=lambda bp: len(bp[0]))
        if cons_eff in per:
            return per[cons_eff], f"vertex_table:{base}:{mode_norm}:{cons_eff}"
        if "standard" in per:
            return per["standard"], f"vertex_table:{base}:{mode_norm}:standard(fallback)"

    return None, "unknown"


def estimate_vertex_cost_usd(
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