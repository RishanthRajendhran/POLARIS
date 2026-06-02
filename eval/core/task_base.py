# core/task_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Sequence
import json

class Example(NamedTuple):
    """
    Minimal unit of work for a Task.

    - id: unique identifier (string) used to track this example across providers/batches.
    - data: arbitrary dict of fields the Task needs to build prompts and parse responses.
    """
    id: str
    data: Dict[str, Any]


class Task(ABC):
    """
    Abstract base for all tasks (win-rate, RM, etc.).

    Responsibilities:
      1. Load raw examples (from JSONL, HF, multiple files, etc).
      2. Build messages for the provider (OpenAI-style messages).
      3. Optionally provide provider-specific response_format (e.g., OpenAI json_schema).
      4. Parse provider outputs into structured records.
      5. Optionally aggregate records into metrics, CSVs, plots, etc.
    """

    # Short identifier for registry / config
    name: str = "base"

    # ---- 1. Load / prepare examples ----
    @abstractmethod
    def load_examples(self, task_config: Dict[str, Any]) -> Sequence[Example]:
        """
        Task-specific data loading.

        task_config is a free-form dict taken from the top-level run_config["task_config"].
        Example patterns:
          - For win-rate: read 1 JSONL where each row already has prompt, story_a, story_b.
          - Later: read 2 JSONLs and align by prompt_id.
          - For RM: read 1 JSONL with 'story' field.
        """
        ...

    # ---- 2. Build prompts/messages ----
    @abstractmethod
    def build_messages(self, ex: Example, run_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert Example -> OpenAI-style messages.

        Returns a list of dicts like:
          [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."}
          ]
        """
        ...

    # ---- 3. Provider-specific response_format (optional) ----
    def get_response_format(
        self,
        provider: str,
        run_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        For providers that support response_format (e.g., OpenAI json_schema).

        This is a *default* that can be overridden by run_config["engine"]["provider_params"]["response_format"].
        """
        return None

    # ---- 4. Parse one response ----
    @abstractmethod
    def parse_response(
        self,
        ex: Example,
        provider_output: Any,
        run_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert provider output into a serializable record (e.g., JSON line/dict).

        provider_output is provider-specific:
          - For HF/vLLM/etc. in this phase, usually a plain string.
          - Later phases: could be JSON objects (e.g., OpenAI json_schema).
        """
        ...

    # ---- 5. Aggregate over all records ----
    def aggregate(
        self,
        records: List[Dict[str, Any]],
        run_config: Dict[str, Any],
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Optional: compute metrics, write CSVs/plots, etc.
        Default: no-op, returns a small summary.

        Implementations are free to ignore run_config/output_dir or use them.
        """
        return {"num_records": len(records)}

    # ---------- Optional debug/preview helpers ----------

    def format_input_preview(
        self,
        ex: Example,
        messages: List[Dict[str, str]],
        run_config: Dict[str, Any],
        max_len: int = 400,
    ) -> str:
        """
        Optional: return a human-readable preview of the input for debugging.
        Default: generic truncated dump of messages.
        """
        def shorten(text: str) -> str:
            text = text.replace("\n", "\\n")
            return text if len(text) <= max_len else text[: max_len - 1] + "…"

        lines = [f"Example id: {ex.id}", "Messages:"]
        for i, m in enumerate(messages):
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"  [{i}] role={role}")
            lines.append("    " + shorten(content))
        return "\n".join(lines)

    def format_record_preview(
        self,
        record: Dict[str, Any],
        run_config: Dict[str, Any],
        max_len: int = 400,
    ) -> str:
        """
        Optional: return a human-readable preview of a parsed record.
        Default: truncated JSON.
        """
        s = json.dumps(record, indent=2)
        s = s.replace("\n", "\\n")
        return s if len(s) <= max_len else s[: max_len - 1] + "…"
    
    def format_summary(
        self,
        summary: Dict[str, Any],
        run_config: Dict[str, Any],
        max_len: int = 4000,
    ) -> str:
        """
        Optional: pretty-print the aggregate summary for this task.

        Default: JSON with indentation, truncated if very long.
        Tasks are encouraged to override for more human-friendly output.
        """
        s = json.dumps(summary, indent=2)
        s = s.replace("\n", "\n  ")  # small indentation for console
        return s if len(s) <= max_len else s[: max_len - 1] + "…"