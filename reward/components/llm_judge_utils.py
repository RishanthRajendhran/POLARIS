# reward/components/llm_judge_utils.py
"""
Shared utilities for LLM-judge reward components
(StoryJudgeCoherenceComponent, StoryQualityComponent, SurfaceArtifactsJudgeComponent).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Vertex / OpenAI model capability heuristics
# ---------------------------------------------------------------------------

_VERTEX_GEMINI_STRUCTURED_BASES = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def _supports_openai_json_schema(model: str) -> bool:
    m = (model or "").strip().lower()
    return m.startswith(("gpt-4", "gpt-5", "o"))


def _supports_vertex_schema(model: str) -> bool:
    m = (model or "").strip().lower()
    return any(m.startswith(base) for base in _VERTEX_GEMINI_STRUCTURED_BASES)


def supports_native_schema(provider: str, model: str) -> bool:
    if provider == "openai":
        return _supports_openai_json_schema(model)
    if provider == "vertex":
        return _supports_vertex_schema(model)
    return False


# ---------------------------------------------------------------------------
# JSON-schema sanitisation for google.genai (Vertex)
# ---------------------------------------------------------------------------

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _sanitize_numeric_enum(node: Dict[str, Any], mode: str) -> None:
    """
    node is a JSON-schema-ish dict.
    mode:
      - "range":  replace enum=[nums] with minimum/maximum and delete enum
      - "drop":   delete enum
      - "string": cast enum values to strings and set type="string"
    """
    enum = node.get("enum", None)
    if not isinstance(enum, list) or not enum:
        return
    if all(isinstance(v, str) for v in enum):
        return  # already strings, nothing to do

    if all(_is_num(v) for v in enum):
        if mode == "drop":
            node.pop("enum", None)
            return
        if mode == "string":
            node["type"] = "string"
            node["enum"] = [str(v) for v in enum]
            return
        # default: "range"
        lo = float(min(enum))
        hi = float(max(enum))
        node.pop("enum", None)
        if "type" not in node:
            node["type"] = "integer" if all(float(v).is_integer() for v in enum) else "number"
        node["minimum"] = lo
        node["maximum"] = hi
        return

    # mixed types: safest fallback is string
    node["type"] = "string"
    node["enum"] = [str(v) for v in enum]


def _sanitize_numeric_const(node: Dict[str, Any], mode: str) -> None:
    """
    Handle const values that are numeric (Gemini Schema may not support const).
    mode:
      - "range": const=n -> minimum=n, maximum=n
      - "drop":  drop const
    """
    if "const" not in node:
        return
    c = node.get("const")
    if not _is_num(c):
        return
    if mode == "drop":
        node.pop("const", None)
        return
    node.pop("const", None)
    if "type" not in node:
        node["type"] = "integer" if float(c).is_integer() else "number"
    node["minimum"] = float(c)
    node["maximum"] = float(c)


def sanitize_json_schema_for_genai(
    schema: Any,
    numeric_enum_mode: str = "range",
    numeric_const_mode: str = "range",
    strip_pattern: bool = True,
) -> Any:
    """
    Recursively sanitise a JSON-schema-ish dict for google.genai.types.Schema.

    - Numeric enums  -> minimum/maximum (or drop/string)
    - Numeric const  -> minimum/maximum (or drop)
    - pattern fields -> removed (Vertex Schema validation often rejects regex constraints)
    """
    if isinstance(schema, list):
        return [
            sanitize_json_schema_for_genai(x, numeric_enum_mode, numeric_const_mode, strip_pattern)
            for x in schema
        ]

    if not isinstance(schema, dict):
        return schema

    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if strip_pattern and k == "pattern":
            continue

        if k == "properties" and isinstance(v, dict):
            out[k] = {
                kk: sanitize_json_schema_for_genai(vv, numeric_enum_mode, numeric_const_mode, strip_pattern)
                for kk, vv in v.items()
            }
        elif k in ("items", "anyOf", "oneOf", "allOf"):
            out[k] = sanitize_json_schema_for_genai(v, numeric_enum_mode, numeric_const_mode, strip_pattern)
        elif isinstance(v, (dict, list)):
            out[k] = sanitize_json_schema_for_genai(v, numeric_enum_mode, numeric_const_mode, strip_pattern)
        else:
            out[k] = v

    _sanitize_numeric_enum(out, numeric_enum_mode)
    _sanitize_numeric_const(out, numeric_const_mode)
    return out


# ---------------------------------------------------------------------------
# JSON extraction helpers (used when native schema is unavailable)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", flags=re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", (s or "").strip()).strip()


def _extract_first_json_object(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        return s
    a = s.find("{")
    b = s.rfind("}")
    if a >= 0 and b > a:
        return s[a:b + 1]
    return None
