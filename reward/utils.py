# reward/utils.py
from __future__ import annotations
import re
from typing import Tuple, Optional

# Non-greedy, DOTALL, optional <think>...</think>, capture the rest as prediction
_THINK_RE = re.compile(
    # r"^\s*(?:<think>\s*(?P<thoughts>.*?)\s*</think>\s*)?(?P<prediction>.*)\s*$",
    r"^\s*(?:(?:<think>)?\s*(?P<thoughts>.*?)\s*</think>\s*)?(?P<prediction>.*)\s*$",
    re.DOTALL,
)

def parse_think_and_prediction(text: str) -> Tuple[Optional[str], str]:
    """
    Returns (thoughts, prediction). thoughts is None if the <think> block is absent.
    The 'prediction' is everything after </think> (or the whole string if absent).
    """
    m = _THINK_RE.match(text)
    if not m:
        return None, text
    thoughts = m.group("thoughts")
    if thoughts is not None and thoughts.strip() == "":
        thoughts = None
    prediction = m.group("prediction") or ""
    return thoughts, prediction