# Composite-batch reward manager for the experimental reward_loop framework.
# Supports BOTH streaming (per-item) and batch reward computation paths.
#
# Streaming path: run_single() — called per-item during rollout generation.
# Batch path: compute_score_batch() — called on a chunk after all rollouts.

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("composite_batch")
class CompositeBatchRewardManager(RewardManagerBase):
    """
    Reward manager that calls ``compute_score(batch, return_dict=True)`` with
    a DataProto.  Works in both streaming (B=1 per call) and batch modes.

    The user's reward function must accept a DataProto and ``return_dict=True``
    kwarg and return ``{"reward_tensor": Tensor[B, T], "reward_extra_info": dict}``.
    """

    def __init__(self, config, tokenizer, compute_score, **reward_kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.is_async = inspect.iscoroutinefunction(compute_score) if compute_score else False

        # The actual reward_kwargs (component enables, judge config, etc.) live
        # in config.reward_model.reward_kwargs (or the migrated reward.reward_kwargs).
        # The **reward_kwargs passed here are just infrastructure args
        # (reward_router_address, reward_model_tokenizer) from load_reward_manager.
        from omegaconf import OmegaConf

        # Build reward_kwargs by merging:
        # 1. config.algorithm (training hyperparams that reward engine also reads)
        # 2. config.reward_model.reward_kwargs (reward-specific component config)
        # reward_kwargs keys override algorithm keys when both exist.
        def _resolve_node(cfg, dotpath):
            """Safely extract an OmegaConf node, returning a plain dict or None."""
            try:
                node = OmegaConf.select(cfg, dotpath, default=None)
                if node is None:
                    return None
                if hasattr(node, '_metadata'):
                    return OmegaConf.to_container(node, resolve=True)
                if isinstance(node, dict):
                    return dict(node)
                if isinstance(node, (str, int, float, bool)):
                    return None
                return dict(node)
            except Exception as e:
                print(f"[CompositeBatchRewardManager] _resolve_node('{dotpath}') failed: {e}", flush=True)
                return None

        cfg_rw = {}
        alg = _resolve_node(config, "algorithm")
        if alg:
            cfg_rw.update(alg)
        # Try migrated path first (reward.reward_kwargs), then legacy
        rw_found = False
        for dotpath in ["reward.reward_kwargs", "reward_model.reward_kwargs"]:
            rw = _resolve_node(config, dotpath)
            if rw:
                cfg_rw.update(rw)
                rw_found = True
                print(f"[CompositeBatchRewardManager] Loaded reward_kwargs from '{dotpath}', "
                      f"keys={list(rw.keys())[:8]}", flush=True)
                break  # first found wins
        if not rw_found:
            # Debug: show what keys exist at top level and under reward/reward_model
            top_keys = list(config.keys()) if hasattr(config, 'keys') else str(type(config))
            reward_keys = list(config.reward.keys()) if hasattr(config, 'reward') and hasattr(config.reward, 'keys') else 'N/A'
            rm_keys = list(config.reward_model.keys()) if hasattr(config, 'reward_model') and hasattr(config.reward_model, 'keys') else 'N/A'
            print(f"[CompositeBatchRewardManager] WARNING: No reward_kwargs found! "
                  f"top_keys={top_keys}, reward_keys={reward_keys}, rm_keys={rm_keys}", flush=True)
        self.reward_kwargs = cfg_rw

        # Build val_reward_kwargs: training as base + val overrides
        self.val_reward_kwargs = dict(cfg_rw)
        for dotpath in ["reward.val_reward_kwargs", "reward_model.val_reward_kwargs"]:
            vrw = _resolve_node(config, dotpath)
            if vrw:
                self.val_reward_kwargs.update(vrw)
                break

        # Sanity log
        rw_keys = list(self.reward_kwargs.keys())
        vrw_keys = list(self.val_reward_kwargs.keys())
        print(f"[CompositeBatchRewardManager] init OK. "
              f"compute_score={self.compute_score}, "
              f"reward_kwargs keys={rw_keys[:10]}, "
              f"val_reward_kwargs keys={vrw_keys[:10]}", flush=True)

    # ------------------------------------------------------------------
    # _inject_meta: ensure meta_info has the context the reward fn needs
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_bytes(data: DataProto) -> None:
        """Decode bytes→str in non_tensor_batch (Ray serialization artefact)."""
        import numpy as np
        for k, v in list(data.non_tensor_batch.items()):
            if isinstance(v, np.ndarray) and v.dtype.kind in ('S', 'O'):
                try:
                    decoded = []
                    for item in v.flat:
                        if isinstance(item, bytes):
                            decoded.append(item.decode("utf-8", errors="replace"))
                        elif isinstance(item, dict):
                            decoded.append({
                                dk: dv.decode("utf-8", errors="replace") if isinstance(dv, bytes) else dv
                                for dk, dv in item.items()
                            })
                        else:
                            decoded.append(item)
                    data.non_tensor_batch[k] = np.array(decoded, dtype=object).reshape(v.shape)
                except Exception:
                    pass

    def _inject_meta(self, data: DataProto) -> None:
        """Inject tokenizer, config, and reward_kwargs into data.meta_info.
        Uses val_reward_kwargs when data.meta_info["validate"] is True."""
        if data.meta_info is None:
            data.meta_info = {}
        if self.tokenizer is not None:
            data.meta_info.setdefault("__tokenizer__", self.tokenizer)
        is_val = bool(data.meta_info.get("validate", False))
        rw = self.val_reward_kwargs if is_val else self.reward_kwargs
        alg_view = data.meta_info.get("__reward_cfg_alg__", {}) or {}
        merged = dict(alg_view)
        merged.update(rw)
        data.meta_info["__reward_cfg_alg__"] = merged
        # Decode bytes→str in non_tensor_batch
        self._decode_bytes(data)
        # Flatten story_prompt from extra_info if not already present
        self._ensure_story_prompt(data)

    @staticmethod
    def _ensure_story_prompt(data: DataProto) -> None:
        """Extract story_prompt from extra_info or raw_prompt if not already present.
        Reward components like blank_length need a plain-text prompt string, but
        VERL datasets store chat messages in 'prompt' and plain text in extra_info."""
        import numpy as np
        ntb = data.non_tensor_batch
        if "story_prompt" in ntb:
            return
        B = len(data)
        prompts = []
        for i in range(B):
            sp = None
            # Try extra_info.prompt first (plain text string)
            ei = ntb.get("extra_info")
            if ei is not None:
                try:
                    ei_val = ei[i] if hasattr(ei, '__getitem__') else ei
                    if isinstance(ei_val, dict):
                        sp = ei_val.get("prompt", None)
                except Exception:
                    pass
            # Fallback: extract content from raw_prompt (chat messages)
            if sp is None:
                rp = ntb.get("raw_prompt")
                if rp is not None:
                    try:
                        rp_val = rp[i] if hasattr(rp, '__getitem__') else rp
                        # rp_val might be an array of message dicts
                        if hasattr(rp_val, '__iter__'):
                            for msg in rp_val:
                                if isinstance(msg, dict) and msg.get("role") == "user":
                                    sp = msg.get("content", "")
                                    break
                    except Exception:
                        pass
            prompts.append(sp or "")
        ntb["story_prompt"] = np.array(prompts, dtype=object)

    # ------------------------------------------------------------------
    # _call_reward_fn: call the reward fn (sync or async)
    # ------------------------------------------------------------------
    async def _call_reward_fn(self, data: DataProto) -> dict:
        """Call compute_score and normalise the return value.

        Sync reward fns are run in a thread via run_in_executor so that:
        (a) they don't block the Ray actor event loop, and
        (b) internal asyncio.run() calls work (the thread has no event loop).
        """
        compute_fn = self.compute_score
        try:
            if self.is_async:
                out = await compute_fn(data, return_dict=True)
            else:
                out = await self.loop.run_in_executor(
                    None, lambda: compute_fn(data, return_dict=True)
                )
        except Exception as exc:
            import traceback, sys
            err_msg = traceback.format_exc()
            # Log to both stdout and stderr (thread stdout may not reach Ray logs)
            msg = f"[CompositeBatchRewardManager] ERROR in reward fn: {exc}\n{err_msg}"
            print(msg, flush=True)
            print(msg, file=sys.stderr, flush=True)
            # Also write to a file as a fallback
            try:
                with open("/tmp/composite_batch_reward_error.log", "a") as ef:
                    ef.write(f"\n{'='*60}\n{msg}\n")
            except Exception:
                pass
            B, T = data.batch["responses"].shape
            zero = torch.zeros((B, T), dtype=torch.float32)
            out = {"reward_tensor": zero, "reward_extra_info": {"_reward_error": [str(exc)] * B}}

        # Handle legacy (tensor, dict) tuple returns
        if isinstance(out, tuple) and len(out) == 2:
            out = {"reward_tensor": out[0], "reward_extra_info": out[1]}
        return out

    # ------------------------------------------------------------------
    # _extract_per_item: split batch result into per-item dicts
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_per_item(data: DataProto, out: dict) -> list[dict]:
        reward_tensor = out.get("reward_tensor")  # [B, T]
        extra_info = out.get("reward_extra_info", {})
        B = len(data)

        prompt_length = data.batch["prompts"].size(1)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)

        per_item_results: list[dict] = []
        for i in range(B):
            vrl = int(valid_response_length[i].item())
            if reward_tensor is not None and vrl > 0:
                score = float(reward_tensor[i, vrl - 1].item())
            elif reward_tensor is not None:
                score = float(reward_tensor[i].sum().item())
            else:
                score = 0.0

            item_extra = {}
            for k, v in extra_info.items():
                val = None
                if isinstance(v, list) and len(v) == B:
                    val = v[i]
                elif isinstance(v, list) and len(v) == 1:
                    val = v[0]
                # Only include scalar values; dicts/lists/etc break downstream np.mean
                if val is not None and isinstance(val, (int, float)):
                    item_extra[k] = val

            per_item_results.append({
                "reward_score": score,
                "reward_extra_info": item_extra,
            })
        return per_item_results

    # ------------------------------------------------------------------
    # run_single: streaming path (B=1, called per-item during rollout)
    # ------------------------------------------------------------------
    async def run_single(self, data: DataProto) -> dict:
        """Process a single sample during streaming reward computation."""
        self._inject_meta(data)
        # Debug: log non_tensor_batch types on first call
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            import numpy as np
            ntb_info = {}
            for k, v in data.non_tensor_batch.items():
                if isinstance(v, np.ndarray):
                    sample = v.flat[0] if v.size > 0 else None
                    ntb_info[k] = f"ndarray({v.dtype}, shape={v.shape}, sample_type={type(sample).__name__})"
                    if isinstance(sample, bytes):
                        ntb_info[k] += f" BYTES! first20={sample[:20]}"
                else:
                    ntb_info[k] = f"{type(v).__name__}"
            print(f"[CompositeBatchRewardManager] run_single non_tensor_batch: {ntb_info}", flush=True)
        out = await self._call_reward_fn(data)
        results = self._extract_per_item(data, out)
        return results[0]

    # ------------------------------------------------------------------
    # compute_score_batch: batch path (called on a chunk after rollout)
    # ------------------------------------------------------------------
    async def compute_score_batch(self, data: DataProto) -> list[dict]:
        """Call the reward function once for the full chunk, then split."""
        self._inject_meta(data)
        out = await self._call_reward_fn(data)
        return self._extract_per_item(data, out)
