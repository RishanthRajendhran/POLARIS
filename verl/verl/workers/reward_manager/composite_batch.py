# verl/workers/reward_manager/composite_batch.py
from __future__ import annotations
from typing import Any, Optional, Dict
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
import traceback

import torch

# Optional: if OmegaConf is around we’ll serialize a light view of cfg.algorithm
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None


@register("composite_batch")
class CompositeBatchRewardManager:
    """
    Batch-first RewardManager for VERL that calls your reward once per batch.
    Accepts 'reward_kwargs' (plain dict) and 'tokenizer' in the constructor.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        self.reward_fn = compute_score
        self.tokenizer = tokenizer
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.num_examine = num_examine
        self.rw = reward_kwargs or {}

    def _normalize_reward_extra(self, out: dict, B: int) -> dict:
        """
        Ensure reward_extra_info is a dict of per-sample lists with length B.
        - Broadcast singleton lists [x] -> [x]*B
        - Drop keys with wrong-sized lists
        - Drop non-list values (metadata not per-sample)
        """
        extra = out.get("reward_extra_info", {})
        if not isinstance(extra, dict):
            out["reward_extra_info"] = {}
            return out

        normalized = {}
        for k, v in extra.items():
            if not isinstance(v, list):
                # not per-sample; skip to satisfy VERL validate invariants
                continue
            if len(v) == B:
                normalized[k] = v
            elif len(v) == 1:
                normalized[k] = v * B
            else:
                # wrong-sized list; drop
                # optional: print(f"[reward_mgr] drop {k}: len {len(v)} != B {B}")
                continue
        out["reward_extra_info"] = normalized
        return out
        

    def __call__(self, data, return_dict: bool = False, **kwargs):
        """
        data: DataProto (full batch)
        returns: {"reward_tensor": Tensor[B,T], "reward_extra_info": dict[str, list]} (preferred)
                 or a legacy (tensor, dict) tuple
        """
        # Context: tokenizer and reward_kwargs into meta_info so your reward can read them
        try:
            if self.tokenizer is not None:
                data.meta_info["__tokenizer__"] = self.tokenizer
        except Exception:
            pass

        # Merge any algorithm view (if trainer stashes it) with reward_kwargs; rw wins
        try:
            alg_view = {}
            if OmegaConf is not None and hasattr(data, "meta_info"):
                # optional: if the trainer injected cfg.algorithm to meta_info, reuse it
                alg_view = data.meta_info.get("__reward_cfg_alg__", {}) or {}
        except Exception:
            alg_view = {}
        merged_alg = dict(alg_view)
        merged_alg.update(self.rw)  # reward_kwargs override anything from cfg view
        data.meta_info["__reward_cfg_alg__"] = merged_alg

        # Actor model path can also be supplied via reward_kwargs (ppl/rankgen models etc.)
        if "__actor_model_path__" not in data.meta_info:
            model_path = self.rw.get("actor_model_path", None)
            data.meta_info["__actor_model_path__"] = model_path

        # Swallow per-item kwargs the caller might pass (e.g., data_source)
        _ = kwargs

        # Call your composite reward once for the whole batch
        try:
            out = self.reward_fn(data, return_dict=True)
        except TypeError:
            # legacy: some rewards still return (tensor, dict)
            rt, extra = self.reward_fn(data, return_dict)
            out = {"reward_tensor": rt, "reward_extra_info": extra}
        except Exception as e:
            traceback.print_exc()
            B, T = data.batch["responses"].shape
            zero = torch.zeros((B, T), dtype=torch.float32, device=data.batch["responses"].device)
            return {"reward_tensor": zero, "reward_extra_info": {"error": [str(e)] * B}}

        B = data.batch["responses"].shape[0]
        if isinstance(out, dict) and "reward_tensor" in out:
            out = self._normalize_reward_extra(out, B)
            return out

        if isinstance(out, tuple) and len(out) == 2:
            rt, extra = out
            out = {"reward_tensor": rt, "reward_extra_info": extra}
            out = self._normalize_reward_extra(out, B)
            return out

        # Fallback: unexpected output
        B, T = data.batch["responses"].shape
        zero = torch.zeros((B, T), dtype=torch.float32, device=data.batch["responses"].device)
        return {"reward_tensor": zero, "reward_extra_info": {}}