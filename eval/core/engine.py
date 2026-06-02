# core/engine.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from core.task_base import Task, Example


class Engine(ABC):
    """
    Abstract base for synchronous providers (HF, vLLM, OpenAI sync, etc.).

    For OpenAI Batch we will *not* use this interface, but a separate builder
    that writes JSONL using Task.build_messages + sampling config.
    """

    name: str = "base"

    @abstractmethod
    def generate_one(
        self,
        messages: List[Dict[str, str]],
        task: Task,
        run_config: Dict[str, Any],
    ) -> Any:
        """
        Run one inference call for the given messages.

        Returns a provider-specific output (string, JSON dict, etc.)
        that Task.parse_response() will interpret.
        """
        ...