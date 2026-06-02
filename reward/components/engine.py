from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import regex as re

from reward.components.base import RewardComponent
from reward.components.self_repeat import SelfRepeatComponent
from reward.components.blank_length import BlankAndLengthComponent
from reward.components.story_quality import StoryQualityComponent
from reward.utils import parse_think_and_prediction


class CompositeRewardEngine:
    """Paper-path reward engine for POLARIS.

    This public copy intentionally supports only the main paper reward stack:
    - Story Quality
    - self repetition penalty
    - blank / length penalties

    It drops historical reward families and locator-driven token-level shaping.
    The output is still a token-shaped reward tensor so outcome-mode GRPO can run
    unchanged in the modified VERL fork.
    """

    def __init__(self):
        self._tokenizer = None
        self._components: List[RewardComponent] = [
            BlankAndLengthComponent(),
            SelfRepeatComponent(),
            StoryQualityComponent(),
        ]

    def _decode_responses(self, batch, tokenizer, cfg: Any, model_path_from_meta: Optional[str]) -> List[str]:
        if tokenizer is not None:
            tok = tokenizer
        elif self._tokenizer is not None:
            tok = self._tokenizer
        else:
            from transformers import AutoTokenizer

            mpath = None
            if cfg is not None and hasattr(cfg, "actor_rollout_ref"):
                try:
                    mpath = cfg.actor_rollout_ref.model.path
                except Exception:
                    mpath = None
            if mpath is None:
                mpath = model_path_from_meta or "Qwen/Qwen3.5-9B"
            tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
            self._tokenizer = tok

        ids = batch.batch["responses"].cpu()
        return [tok.decode(x, skip_special_tokens=True) for x in ids]

    def _thought_token_count(self, text: str, response_len: int, tokenizer) -> int:
        if tokenizer is None or not getattr(tokenizer, "is_fast", False):
            return 0
        tl = text.lower()
        pos = tl.find("</think>")
        if pos < 0:
            return 0
        end_pos = pos + len("</think>")
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_attention_mask=False)
        offs = enc.get("offset_mapping", None)
        if offs is None or len(offs) == 0:
            return 0
        cnt = 0
        for cs, ce in offs:
            cs = int(cs)
            ce = int(ce)
            if ce <= cs:
                continue
            if 0.5 * (cs + ce) < end_pos:
                cnt += 1
        return min(cnt, response_len)

    def _slice_non_tensor(self, d: Dict[str, Any], idxs: List[int], batch_size: int) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            try:
                if isinstance(v, (list, tuple)) and len(v) == batch_size:
                    out[k] = [v[i] for i in idxs]
                elif isinstance(v, np.ndarray) and v.shape[0] == batch_size:
                    out[k] = [v[i] for i in idxs]
                elif torch.is_tensor(v) and v.shape[0] == batch_size:
                    out[k] = [v[i].item() for i in idxs]
                else:
                    out[k] = v
            except Exception:
                out[k] = v
        return out

    def compute(self, batch, tokenizer, cfg: Any, return_dict: bool = False):
        tok_from_meta = batch.meta_info.get("__tokenizer__", None)
        tokenizer = tokenizer or tok_from_meta
        alg_view = batch.meta_info.get("__reward_cfg_alg__", {}) or {}
        model_path_from_meta = batch.meta_info.get("__actor_model_path__", None)

        from types import SimpleNamespace

        if cfg is None:
            cfg = SimpleNamespace(algorithm=SimpleNamespace(**alg_view))

        is_validation = bool(batch.meta_info.get("validate", False))
        if is_validation:
            print("[POLARIS engine] VALIDATION reward_components:", getattr(cfg.algorithm, "reward_components", {}))

        device = batch.batch["responses"].device
        resp_mask = batch.batch["responses"].ne(0).to(torch.float32)
        if "response_mask" in batch.batch:
            resp_mask = batch.batch["response_mask"].to(torch.float32)
        B, T = batch.batch["responses"].shape

        non_tensor = getattr(batch, "non_tensor_batch", {}) or {}
        texts = self._decode_responses(batch, tokenizer, cfg, model_path_from_meta)

        parsed = [parse_think_and_prediction(t) for t in texts]
        thoughts: List[Optional[str]] = [th for th, _ in parsed]
        preds_only: List[str] = [pr for _, pr in parsed]

        trim_marker = getattr(cfg.algorithm, "prediction_end_marker", None)
        if trim_marker:
            Ltrim = len(trim_marker)
            preds_only = [p[:-Ltrim] if p.endswith(trim_marker) else p for p in preds_only]

        is_gt_list = non_tensor.get("is_gt", [False] * B)
        is_gt = np.asarray(is_gt_list, dtype=bool) if len(is_gt_list) == B else np.zeros(B, dtype=bool)
        gt_idx = np.where(is_gt)[0].tolist()
        pol_idx = np.where(~is_gt)[0].tolist()

        gt_override_best = bool(getattr(cfg.algorithm, "gt_override_best", False))
        compute_idx = pol_idx if gt_override_best else list(range(B))

        comp_clip = float(getattr(cfg.algorithm, "composite_clip", 2.0))
        if gt_override_best and len(pol_idx) == 0 and len(gt_idx) == B:
            token_level = comp_clip * resp_mask
            extra: Dict[str, List[float]] = {
                "composite": [float(comp_clip)] * B,
                "pos_gate": [1.0] * B,
            }
            if return_dict:
                return {"reward_tensor": token_level, "reward_extra_info": extra}
            return token_level, extra

        comp_vals: Dict[str, List[float]] = {}
        raw_by_comp: Dict[str, List[Any]] = {c.name: [None] * B for c in self._components}

        for comp in self._components:
            if not comp.enabled(cfg):
                continue
            if len(compute_idx) == 0:
                continue

            texts_sub = [preds_only[i] for i in compute_idx]
            non_tensor_sub = self._slice_non_tensor(non_tensor, compute_idx, B)
            vals = comp.compute(texts_sub, non_tensor_sub, tokenizer, cfg, None)

            for k, v in vals.items():
                if k == "extra_info":
                    continue
                if k not in comp_vals:
                    comp_vals[k] = [0.0] * B
                if len(v) != len(compute_idx):
                    raise RuntimeError(f"Component {comp.name} key {k} returned length {len(v)} != {len(compute_idx)}")
                for j, ii in enumerate(compute_idx):
                    x = float(v[j])
                    if not np.isfinite(x):
                        x = 0.0
                    comp_vals[k][ii] = x

            raw_list = vals.get("extra_info", None)
            if isinstance(raw_list, list):
                for j, ii in enumerate(compute_idx):
                    raw_by_comp[comp.name][ii] = raw_list[j] if j < len(raw_list) else None
            elif isinstance(raw_list, dict):
                for ii in compute_idx:
                    raw_by_comp[comp.name][ii] = raw_list

        sq_fallback_enable = bool(getattr(cfg.algorithm, "story_quality_fallback_to_mean", True))
        if sq_fallback_enable and "story_quality" in comp_vals:
            sq_raw = raw_by_comp.get("story_quality", [None] * B)
            failed_mask = []
            for i in range(B):
                info = sq_raw[i] if i < len(sq_raw) else None
                failed = False
                if isinstance(info, dict):
                    failed = bool(info.get("error") or info.get("timed_out"))
                failed_mask.append(failed)
            n_failed = sum(failed_mask)
            if 0 < n_failed < B:
                good_scores = [comp_vals["story_quality"][i] for i in range(B) if not failed_mask[i]]
                fallback_val = float(np.mean(good_scores)) if good_scores else 0.0
                for i in range(B):
                    if failed_mask[i]:
                        comp_vals["story_quality"][i] = fallback_val
                print(f"[POLARIS engine] story_quality: {n_failed}/{B} judge failures, replaced with batch mean={fallback_val:.3f}")

        def W(name: str, default: float = 0.0) -> float:
            w = getattr(cfg.algorithm, "reward_weights", None)
            if w and (name in w):
                return float(w[name])
            return float(default)

        storyq_arr = np.asarray(comp_vals.get("story_quality", [0.0] * B), dtype=np.float32)
        selfrep_arr = np.asarray(comp_vals.get("self_rep", [0.0] * B), dtype=np.float32)
        lenpen_arr = np.asarray(comp_vals.get("length_pen", [0.0] * B), dtype=np.float32)
        blank_arr = np.asarray(comp_vals.get("blank_pen", [0.0] * B), dtype=np.float32)
        lenrew_arr = np.asarray(comp_vals.get("length_reward", [0.0] * B), dtype=np.float32)

        for arr in (storyq_arr, selfrep_arr, lenpen_arr, blank_arr, lenrew_arr):
            arr[~np.isfinite(arr)] = 0.0

        pg_cfg = getattr(cfg.algorithm, "pos_gate", {}) if cfg is not None else {}
        if bool(pg_cfg.get("enable", False)):
            gate = np.ones((B,), dtype=np.float32)
            if bool(pg_cfg.get("require_non_blank", True)):
                gate *= (blank_arr < 0.5).astype(np.float32)
            max_sr = float(pg_cfg.get("max_self_rep_for_pos", 0.10))
            if max_sr > 0:
                gate *= np.clip(1.0 - (selfrep_arr / max_sr), 0.0, 1.0)
            max_lp = float(pg_cfg.get("max_length_pen_for_pos", 0.50))
            if max_lp > 0:
                gate *= (lenpen_arr <= max_lp).astype(np.float32)
        else:
            gate = np.ones((B,), dtype=np.float32)

        score_pos_seq = gate * (W("story_quality", 0.0) * storyq_arr + W("length_reward", 0.0) * lenrew_arr)
        score_neg_seq = (
            W("self_rep", 0.0) * selfrep_arr
            + W("length_pen", 0.0) * lenpen_arr
            + W("blank_pen", 0.0) * blank_arr
        )
        composite = score_pos_seq - score_neg_seq

        len_scale_arr = np.ones((B,), dtype=np.float32)
        len_scale_cfg = getattr(cfg.algorithm, "length_scale", None) or {}
        len_scale_enable = bool(len_scale_cfg.get("enable", False) if isinstance(len_scale_cfg, dict) else False)
        if len_scale_enable:
            ls_fallback = int(len_scale_cfg.get("fallback_target_words", 600))
            ls_alpha = float(len_scale_cfg.get("alpha", 0.5))
            ls_floor = float(len_scale_cfg.get("floor", 0.15))
            wc_re = re.compile(r"\w+")
            len_re = re.compile(
                str(getattr(cfg.algorithm, "len_wordcount_regex", r"(?P<word_count>\d+)\s+words\s+long")),
                flags=re.IGNORECASE,
            )
            prompts = None
            for pk in ("story_prompt", "prompt", "raw_prompt"):
                v = non_tensor.get(pk, None)
                if v is not None and (isinstance(v, (list, tuple)) or (hasattr(v, "__len__") and len(v) > 0)):
                    prompts = v
                    break
            if prompts is None:
                prompts = [""] * B
            for i in range(B):
                target_w = ls_fallback
                pm = len_re.search(prompts[i] if i < len(prompts) else "")
                if pm:
                    try:
                        target_w = max(1, int(pm.group("word_count")))
                    except Exception:
                        pass
                wc = len(wc_re.findall(preds_only[i]))
                ratio = min(float(wc) / max(1.0, float(target_w)), 1.0)
                len_scale_arr[i] = max(ls_floor, ratio ** ls_alpha)
            for gi in gt_idx:
                len_scale_arr[gi] = 1.0
            composite = composite * len_scale_arr

        if gt_override_best and len(gt_idx) > 0:
            composite = composite.astype(np.float32, copy=True)
            composite[gt_idx] = comp_clip

        composite = np.clip(composite, -comp_clip, comp_clip)
        composite[~np.isfinite(composite)] = 0.0

        think_mode = str(getattr(cfg.algorithm, "think_reward_mode", "story_mean")).lower()
        think_scale = float(getattr(cfg.algorithm, "think_reward_scale", 1.0))
        think_cap = int(getattr(cfg.algorithm, "think_len_cap", 0))
        think_pen = float(getattr(cfg.algorithm, "think_len_penalty", 0.0))

        overlong_enable = bool(getattr(cfg.algorithm, "overlong_buffer_enable", False))
        overlong_buffer_len = int(getattr(cfg.algorithm, "overlong_buffer_len", 0))
        overlong_penalty_factor = float(getattr(cfg.algorithm, "overlong_penalty_factor", 1.0))
        overlong_max_resp_len = int(getattr(cfg.algorithm, "overlong_max_response_length", 0))

        story_scalar_arr = (
            W("story_quality", 0.0) * storyq_arr
            - W("self_rep", 0.0) * selfrep_arr
            - W("length_pen", 0.0) * lenpen_arr
            - W("blank_pen", 0.0) * blank_arr
        )
        story_scalar_arr[~np.isfinite(story_scalar_arr)] = 0.0

        token_level = torch.zeros((B, T), dtype=torch.float32, device=device)
        for i in range(B):
            L = int(resp_mask[i].sum().item())
            if L <= 0:
                continue

            s0 = self._thought_token_count(texts[i], L, tokenizer)
            S = max(0, L - s0)
            story_scalar = float(story_scalar_arr[i])
            story_mean = story_scalar if S > 0 else 0.0

            if think_mode == "zero":
                thought_val = 0.0
            elif think_mode == "composite":
                thought_val = float(composite[i])
            else:
                thought_val = story_mean
            thought_val *= think_scale

            if s0 > 0:
                token_level[i, :s0] = thought_val
            if S > 0:
                token_level[i, s0:s0 + S] = story_scalar

            if s0 > 0 and think_cap > 0 and think_pen != 0.0:
                over = max(0, s0 - think_cap)
                if over > 0:
                    token_level[i, think_cap:s0] -= think_pen

            token_level[i, :] = token_level[i, :] * resp_mask[i]

            if overlong_enable and S > 0 and overlong_max_resp_len > 0:
                expected_len = overlong_max_resp_len - overlong_buffer_len
                if L >= expected_len:
                    eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer else None
                    if eos_id is not None:
                        has_eos = bool((batch.batch["responses"][i, :L] == eos_id).any().item())
                    else:
                        has_eos = (L < overlong_max_resp_len)
                    is_truncated = not has_eos
                    if is_truncated:
                        exceed = L - expected_len
                        if overlong_buffer_len > 0:
                            penalty_total = min(-(exceed / float(overlong_buffer_len)) * overlong_penalty_factor, 0.0)
                        else:
                            penalty_total = -float(overlong_penalty_factor)
                        if penalty_total < 0.0:
                            token_level[i, s0:s0 + S] += penalty_total / float(S)

        token_level = torch.clamp(token_level, -comp_clip, comp_clip)
        token_level = torch.nan_to_num(token_level, nan=0.0, posinf=0.0, neginf=0.0)

        extra: Dict[str, List[float]] = {
            "composite": [float(x) for x in composite.tolist()],
            "pos_gate": [float(x) for x in gate.tolist()],
        }
        if len_scale_enable:
            extra["len_scale"] = [float(x) for x in len_scale_arr.tolist()]

        for k, v in comp_vals.items():
            if isinstance(v, list) and len(v) == B:
                extra[k] = [float(x) if np.isfinite(float(x)) else 0.0 for x in v]

        extra["weights_used"] = {
            "story_quality": W("story_quality", 0.0),
            "self_rep": W("self_rep", 0.0),
            "length_pen": W("length_pen", 0.0),
            "length_reward": W("length_reward", 0.0),
            "blank_pen": W("blank_pen", 0.0),
        }

        if return_dict:
            return {"reward_tensor": token_level, "reward_extra_info": extra}
        return token_level, extra
