# reward/components/__init__.py
from __future__ import annotations
from typing import Any
from .engine import CompositeRewardEngine

_ENGINE = None

def composite_reward_fn(batch, return_dict: bool = False, **kwargs):
    """
    Entry point called by VERL's NaiveRewardManager.
    Must tolerate extra kwargs like data_source, return_dict, etc.
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CompositeRewardEngine()

    # Let manager override return_dict via kwargs if it passed it
    rd = bool(kwargs.get("return_dict", return_dict))

    # If manager passed data_source (or other metadata), keep it for debugging if you like
    try:
        if "data_source" in kwargs:
            batch.meta_info["__data_source__"] = kwargs["data_source"]
    except Exception:
        pass

    # We rely on our engine to read tokenizer/cfg from batch.meta_info if needed
    # (see earlier patch that stashes context into meta_info in the trainer).
    result = _ENGINE.compute(batch, tokenizer=None, cfg=None, return_dict=True)

    if rd:
        return result
    # Back-compat: some call sites expect a (tensor, extra_info) tuple
    return result["reward_tensor"], result.get("reward_extra_info", {})