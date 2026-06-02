# core/provider_registry.py
from __future__ import annotations
from typing import Dict, Any
import json
import threading

from core.engine import Engine
from providers.openai_sync import OpenAIChatEngine
from providers.vertex_engine import VertexAIEngine

# Cache engines per engine-config so each backend is instantiated once per process.
_ENGINE_CACHE: Dict[str, Engine] = {}
_ENGINE_CACHE_LOCK = threading.Lock()


def _engine_cache_key(engine_cfg: Dict[str, Any]) -> str:
    return json.dumps(engine_cfg, sort_keys=True, separators=(",", ":"))


def get_engine(run_config: Dict[str, Any]) -> Engine:
    """
    Factory for Engine instances based on run_config["engine"]["provider"].

    Supported providers in the public POLARIS eval:
      - "openai"  (GPT judges; Story Quality, EQ-Bench LongForm/Creative)
      - "vertex"  (Gemini judges; WritingBench, LongBench-Write, pairwise Elo)
    """
    engine_cfg = run_config.get("engine", {}) or {}
    provider = engine_cfg.get("provider")
    if not provider:
        raise ValueError("engine.provider must be set ('openai' or 'vertex').")

    cache_key = _engine_cache_key(engine_cfg)
    with _ENGINE_CACHE_LOCK:
        if cache_key in _ENGINE_CACHE:
            return _ENGINE_CACHE[cache_key]
        if provider == "openai":
            eng = OpenAIChatEngine(run_config)
        elif provider == "vertex":
            eng = VertexAIEngine(run_config)
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. The public eval ships 'openai' and 'vertex'."
            )
        _ENGINE_CACHE[cache_key] = eng
        return eng
