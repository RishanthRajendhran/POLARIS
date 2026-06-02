# reward/components/blank_length.py
from __future__ import annotations
from typing import Dict, Any, List
import html, re

from .base import RewardComponent

def _visible_char_ratio(s: str) -> float:
    t = html.unescape(s)
    return len(re.findall(r"[^\s]", t)) / max(1, len(t))

def _word_count(s: str) -> int:
    t = html.unescape(s)
    return len(re.findall(r"\w+", t))

class BlankAndLengthComponent(RewardComponent):
    name = "blank_len"

    def enabled(self, cfg: Any) -> bool:
        return bool(getattr(cfg.algorithm, "reward_components", {}).get("enable_blank_len", False))

    def needs_gpu(self) -> bool:
        return False

    def keys(self) -> List[str]:
        return ["blank_pen", "length_pen", "length_reward"]

    def compute(
        self,
        texts: List[str],
        batch_non_tensor: Dict[str, Any],
        tokenizer,
        cfg: Any,
        gpu_actor=None
    ) -> Dict[str, List[float]]:
        # Base knobs
        min_vis   = float(getattr(cfg.algorithm, "min_visible_ratio", 0.05))
        min_words_default = int(getattr(cfg.algorithm, "min_words", 500))
        max_words_cfg = int(getattr(cfg.algorithm, "max_words", -1))
        target_w_cfg = int(getattr(cfg.algorithm, "target_words", -1))

        # Length tolerances: support absolute (int), fractional (float), or unset (-1).
        #   int N (N != -1): absolute tolerance in words (e.g. 500)
        #   float F: fractional tolerance = F * target_wc per sample (e.g. 0.4)
        #   -1 (int): unset / use default
        # Note: 2 is int (absolute), 2.0 is float (fractional).
        _raw_tol       = getattr(cfg.algorithm, "len_tolerance", 300)
        _raw_tol_lower = getattr(cfg.algorithm, "len_tolerance_lower", _raw_tol)
        _raw_tol_upper = getattr(cfg.algorithm, "len_tolerance_upper", _raw_tol)

        # Absolute min/max bounds for tolerances (used with fractional mode).
        # When unset: min=0, max=infinity (no clamping).
        _tol_lower_abs_min = float(getattr(cfg.algorithm, "len_tolerance_lower_abs_min", 0))
        _tol_lower_abs_max = float(getattr(cfg.algorithm, "len_tolerance_lower_abs_max", float("inf")))
        _tol_upper_abs_min = float(getattr(cfg.algorithm, "len_tolerance_upper_abs_min", 0))
        _tol_upper_abs_max = float(getattr(cfg.algorithm, "len_tolerance_upper_abs_max", float("inf")))

        def _resolve_tolerance(raw, target_wc, abs_min, abs_max):
            """Resolve a tolerance value given target word count.
            int -1 => unset (returns 0); int N => absolute N;
            float F => F * target_wc clamped to [abs_min, abs_max]."""
            if isinstance(raw, int):
                # -1 means unset
                return 0 if raw == -1 else raw
            elif isinstance(raw, float):
                # Fractional: compute from target, clamp to bounds
                val = raw * target_wc
                return int(max(abs_min, min(abs_max, val)))
            else:
                # Try to parse: if it has a decimal point, treat as float
                s = str(raw)
                if '.' in s:
                    return int(max(abs_min, min(abs_max, float(s) * target_wc)))
                v = int(s)
                return 0 if v == -1 else v

        # Optional: extract a target length from the prompt via regex
        use_prompt_len = bool(getattr(cfg.algorithm, "len_prompt_use", True))
        len_regex = getattr(cfg.algorithm, "len_wordcount_regex", r"(?P<word_count>\d+)\s+words\s+long")
        prompt_re = re.compile(len_regex, flags=re.IGNORECASE)

        # Cues we may consult
        targets:  List[str] = batch_non_tensor.get("target", [""]*len(texts))
        prompts:  List[str] = batch_non_tensor.get("story_prompt", None) \
                           or batch_non_tensor.get("prompt", None) \
                           or batch_non_tensor.get("raw_prompt", None)
        if not isinstance(prompts, list):
            prompts = [""] * len(texts)

        blank_pen: List[float] = []
        length_pen: List[float] = []
        length_reward: List[float] = []

        for idx, (t, tgt) in enumerate(zip(texts, targets)):
            vis = _visible_char_ratio(t)
            wc  = _word_count(t)
            tgt_wc  = _word_count(tgt)

            # Determine min_words and target_words from config and prompt
            min_words = min_words_default
            # Priority for target_w:
            # 1) Prompt-specified word count (if enabled and found)
            # 2) config.target_words (if >0)
            # 3) fallback to target text length
            target_w = -1
            if use_prompt_len:
                pr = prompts[idx] if idx < len(prompts) else ""
                m = prompt_re.search(pr or "")
                if m:
                    try:
                        target_w = int(m.group("word_count"))
                    except Exception:
                        target_w = -1

            if target_w <= 0:
                if target_w_cfg > 0:
                    target_w = int(target_w_cfg)
                else:
                    target_w = max(min_words, tgt_wc)
            else:
                target_w = max(min_words, target_w)

            # Resolve per-sample tolerances (may be fractional)
            len_tolerance_lower = _resolve_tolerance(_raw_tol_lower, target_w, _tol_lower_abs_min, _tol_lower_abs_max)
            len_tolerance_upper = _resolve_tolerance(_raw_tol_upper, target_w, _tol_upper_abs_min, _tol_upper_abs_max)

            # max_words:
            # if user specified max_words in config use it; else allow a soft ceiling at target+len_tolerance
            max_words = max_words_cfg
            if max_words <= 0:
                max_words = target_w + len_tolerance_upper

            # Blank penalty
            is_blank = (vis < min_vis) or (wc < min_words)
            blank_pen.append(1.0 if is_blank else 0.0)

            # Length penalty (piecewise; asymmetric tolerant band)
            if wc >= (target_w - len_tolerance_lower) and wc <= (target_w + len_tolerance_upper):
                lp = 0.0
            elif wc < (target_w - len_tolerance_lower):
                lp = min(1.0, 1.0 - (wc / max(1.0, float(target_w - len_tolerance_lower))))
            else:
                # Overly long
                if max_words <= 0 or (max_words <= (target_w + len_tolerance_upper)):
                    lp = min(1.0, (wc / max(1.0, float(target_w + len_tolerance_upper))) - 1.0)
                else:
                    lp = 1.0 - ((max_words - wc) / max(1.0, float(max_words - (target_w + len_tolerance_upper))))
                    lp = max(0.0, min(1.0, lp))

            length_pen.append(float(lp))

            # Length reward: continuous [0, 1] signal within tolerance only
            #   - 1.0 at exactly target_w
            #   - Decays linearly to 0.0 at tolerance edges
            #   - 0.0 outside tolerance band
            #   - 0.0 if below min_words (blank)
            if is_blank:
                lr = 0.0
            elif wc >= (target_w - len_tolerance_lower) and wc <= (target_w + len_tolerance_upper):
                # Within tolerance: linear from 1.0 (at target) to 0.0 (at edge)
                if wc <= target_w:
                    dist_frac = (target_w - wc) / max(1.0, float(len_tolerance_lower))
                else:
                    dist_frac = (wc - target_w) / max(1.0, float(len_tolerance_upper))
                lr = max(0.0, 1.0 - dist_frac)
            else:
                lr = 0.0

            length_reward.append(float(lr))

        return {"blank_pen": blank_pen, "length_pen": length_pen, "length_reward": length_reward}
