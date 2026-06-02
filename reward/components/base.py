# reward/components/base.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Protocol

class GPUActorLike(Protocol):
    # Protocol for our Ray GPU actor; duck-typed in components that need it
    ...

class RewardComponent:
    """
    Base component: produce per-sequence scalar(s). Higher is better for "positive" components;
    for penalties, return positive penalty to be subtracted by the engine.

    .name            stable component name used for logging and weights
    .enabled(cfg)    gate via yaml
    .needs_gpu()     hint to the engine
    .keys()          list of output scalar keys (for logging)
    .compute(...)    return dict[str, List[float]] per sequence (length B)
    """

    name: str = "base"

    def enabled(self, cfg: Any) -> bool:
        return True

    def needs_gpu(self) -> bool:
        return False

    def keys(self) -> List[str]:
        return [self.name]

    def compute(
        self,
        texts: List[str],
        batch_non_tensor: Dict[str, Any],
        tokenizer,
        cfg: Any,
        gpu_actor: Optional[GPUActorLike] = None
    ) -> Dict[str, List[float]]:
        raise NotImplementedError