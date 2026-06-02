# verl/trainer/utils/auto_weight.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Deque, Tuple, Optional
from collections import deque
import numpy as np
    
@dataclass
class AWSchedConfig:
    keys: List[str]
    update_every: int = 10
    window: int = 100
    min_points: int = 30
    eta: float = 0.5
    w_min: float = 0.05
    w_max: float = 0.95
    deg: int = 1
    ema_beta: float = 0.0
    start_step: int = 0
    normalize_domain: str = "subset"  # "subset" or "global"
    sum_target: Optional[float] = None  # None or "auto" or float
    log_to_wandb: bool = True
    freeze_if_clamped: bool = True
    key_overrides: Optional[Dict[str, Dict[str, float]]] = None  # e.g., {"pangram": {"w_min":0.25,"w_max":0.55,"eta_scale":0.8}}
    # Keys where lower values = better (penalties). Their slopes are negated
    # so the exponentiated gradient update correctly increases weight when
    # the penalty stagnates and decreases weight when it improves.
    penalty_keys: Optional[List[str]] = None  # e.g., ["self_rep", "length_pen", "blank_pen"]

class AutoWeightScheduler:
    """
    Implements AW-GRPO-like adaptive weighting for a subset of reward keys.
    Slope ŝ_k is estimated from recent per-step means, then weights update:
        w_k <- clip(alpha_k * exp(-eta * slope_k), w_min, w_max); alpha <- softmax-normalized over chosen keys
    If normalize_domain == "subset", we re-scale alphas so the chosen keys sum to `sum_target`
    (either provided or auto=their initial total from reward_weights).
    """
    def __init__(self, init_weights: Dict[str, float], cfg: AWSchedConfig):
        self.cfg = cfg
        self.keys = list(cfg.keys)
        # initialize alphas from current reward weights, restricted to keys
        self.alpha: Dict[str, float] = {k: float(init_weights.get(k, 0.0)) for k in self.keys}
        total = sum(self.alpha.values()) + 1e-12
        for k in self.keys:
            self.alpha[k] = self.alpha[k] / total  # internal normalized domain
        # target sum in the outer config domain
        if cfg.sum_target == "auto":
            self.sum_target = float(sum(init_weights.get(k, 0.0) for k in self.keys))
        elif isinstance(cfg.sum_target, (float, int)):
            self.sum_target = float(cfg.sum_target)
        else:
            self.sum_target = float(sum(init_weights.get(k, 0.0) for k in self.keys))

        # ring buffers of (step, value)
        self.hist: Dict[str, Deque[Tuple[int, float]]] = {k: deque(maxlen=max(10, cfg.window)) for k in self.keys}
        # EMA state
        self.ema: Dict[str, float] = {}
        self.over = cfg.key_overrides or {}

    def _add_point(self, step: int, means: Dict[str, float]):
        beta = float(self.cfg.ema_beta)
        for k in self.keys:
            v = float(means.get(k, np.nan))
            if not np.isfinite(v):
                continue
            if beta > 0.0:
                m_prev = self.ema.get(k, v)
                m = beta * m_prev + (1.0 - beta) * v
                self.ema[k] = m
                self.hist[k].append((step, float(m)))
            else:
                self.hist[k].append((step, v))

    def _slope(self, pts: Deque[Tuple[int, float]]) -> float:
        if len(pts) < max(2, self.cfg.deg + 1):
            return 0.0
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)
        try:
            coef = np.polyfit(xs, ys, deg=int(self.cfg.deg))  # deg=1 => [slope, intercept]
            return float(coef[-2]) if self.cfg.deg >= 1 else 0.0
        except Exception:
            return 0.0

    def maybe_update(self, step: int, step_means: Dict[str, float]) -> Optional[Dict[str, float]]:
        # Always collect point
        self._add_point(step, step_means)

        # Update cadence and warmup
        if step < int(self.cfg.start_step):
            return None
        if (step % max(1, int(self.cfg.update_every))) != 0:
            return None
        if any(len(self.hist[k]) < int(self.cfg.min_points) for k in self.keys):
            return None

        # Compute slopes (negate for penalty keys so the update direction is correct)
        penalty_set = set(self.cfg.penalty_keys or [])
        slopes = {}
        for k in self.keys:
            s = self._slope(self.hist[k])
            if k in penalty_set:
                s = -s  # For penalties: decreasing (good) → positive slope → weight decreases
            slopes[k] = s

        # Exponentiated gradient update with clipping
        w = {}
        for k in self.keys:
            eta_k = float(self.cfg.eta) * float(self.over.get(k, {}).get("eta_scale", 1.0))
            wk = self.alpha[k] * np.exp(-eta_k * float(slopes[k]))
            wmin_k = float(self.over.get(k, {}).get("w_min", self.cfg.w_min))
            wmax_k = float(self.over.get(k, {}).get("w_max", self.cfg.w_max))
            wk = float(np.clip(wk, wmin_k, wmax_k))
            w[k] = wk

        # Normalize to sum 1 within subset
        s = sum(w.values()) + 1e-12
        for k in self.keys:
            self.alpha[k] = w[k] / s

        # Optionally freeze if all clamped (prevents thrashing when no gradient info).
        # Use per-key bounds (from key_overrides) to match what was used during clipping.
        if self.cfg.freeze_if_clamped:
            all_min = all(
                np.isclose(w[k], float(self.over.get(k, {}).get("w_min", self.cfg.w_min)))
                for k in self.keys
            )
            all_max = all(
                np.isclose(w[k], float(self.over.get(k, {}).get("w_max", self.cfg.w_max)))
                for k in self.keys
            )
            if all_min or all_max:
                return None

        # Return outer-domain weights for these keys (rescaled to sum_target if subset)
        if self.cfg.normalize_domain == "subset":
            scale = self.sum_target if self.sum_target is not None else 1.0
            return {k: float(self.alpha[k] * scale) for k in self.keys}
        else:
            # global domain: caller should renormalize overall reward_weights as desired
            return {k: float(self.alpha[k]) for k in self.keys}

    def get_debug(self) -> Dict[str, float]:
        return dict(self.alpha)