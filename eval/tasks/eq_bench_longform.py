# tasks/eq_bench_longform.py
"""
EQ-Bench LongForm Writing scoring task — adapted for single-shot stories.

Uses the 16-dimension rubric from the EQ-Bench LongForm Writing benchmark
(final-judging variant) to evaluate stories on a 0-20 scale per dimension.
Negative dimensions are inverted so higher is always better. "Forced Poetry
or Metaphor" receives a 5× weight multiplier with a (score/20)^1.7 power
transform before inversion.

Config shape:
  task: eq_bench_longform
  task_config:
    input_path: <generated stories JSONL>
    story_field: story
    prompt_field: prompt
    id_field: id
  task_params:
    judge_prompt_path: <path to judge prompt .txt>

Output record fields:
  id, prompt, story_word_count, analysis,
  raw_<dim> (0-20 from judge),
  score_<dim> (adjusted: inverted negatives, power-transformed forced_poetry),
  weighted_score (0-100),
  num_dims_rated
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.task_base import Task, Example

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

POSITIVE_DIMS: List[str] = [
    "Nuanced Characters",
    "Emotionally Engaging",
    "Compelling Plot",
    "Coherent",
    "Well-earned Lightness or Darkness",
    "Faithful to Writing Prompt",
]

NEGATIVE_DIMS: List[str] = [
    "Weak Dialogue",
    "Tell-Don't-Show",
    "Unsurprising or Uncreative",
    "Amateurish",
    "Purple Prose",
    "Forced Poetry or Metaphor",
]

ALL_DIMS: List[str] = POSITIVE_DIMS + NEGATIVE_DIMS
_NEGATIVE_SET = set(NEGATIVE_DIMS)

# Special dimension with 5× weight and power transform
_FORCED_POETRY = "Forced Poetry or Metaphor"
_FORCED_POETRY_EXPONENT = 1.7
_DIM_WEIGHTS = {d: 1.0 for d in ALL_DIMS}
_DIM_WEIGHTS[_FORCED_POETRY] = 5.0

_DEFAULT_JUDGE_PROMPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "prompts", "eq_bench_longform_judge.txt")
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_SCORE_LINE_RE = re.compile(r"^(.+?)\s*:\s*\[?(\d+(?:\.\d+)?)\]?\s*$")


def _key(dim: str) -> str:
    return (
        dim.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
    )


def _parse_scores(raw_text: str) -> Dict[str, float]:
    m = re.search(r"\[Scores\](.*?)(?:---|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
    search_text = m.group(1) if m else raw_text

    scores: Dict[str, float] = {}
    for line in search_text.splitlines():
        lm = _SCORE_LINE_RE.match(line.strip())
        if lm:
            metric = lm.group(1).strip()
            score = float(lm.group(2))
            score = min(20.0, max(0.0, score))
            scores[metric] = score
    return scores


def _match_dim(dim: str, raw_scores: Dict[str, float]) -> Optional[float]:
    if dim in raw_scores:
        return raw_scores[dim]
    dim_lower = dim.lower()
    for k, v in raw_scores.items():
        if k.lower() == dim_lower:
            return v
    prefix = dim_lower[:12]
    for k, v in raw_scores.items():
        if k.lower().startswith(prefix):
            return v
    return None


def _extract_analysis(raw_text: str) -> str:
    m = re.search(r"\[Analysis\](.*?)(?:\[Scores\]|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _adjust_score(dim: str, raw_val: float) -> float:
    """Invert negatives, apply power transform for forced poetry.

    Official EQ-Bench order: invert FIRST, then power transform.
    This penalizes forced poetry more heavily than transform-then-invert.
    """
    if dim == _FORCED_POETRY:
        # Step 1: invert (since it's a negative dim)
        inverted = 20.0 - raw_val
        # Step 2: power transform on the inverted value
        return (inverted / 20.0) ** _FORCED_POETRY_EXPONENT * 20.0
    elif dim in _NEGATIVE_SET:
        return 20.0 - raw_val
    else:
        return raw_val


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class EqBenchLongformTask(Task):
    name = "eq_bench_longform"

    def _judge_prompt_template(self, task_params: Dict[str, Any]) -> str:
        path = task_params.get("judge_prompt_path") or _DEFAULT_JUDGE_PROMPT_PATH
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_examples(self, task_config: Dict[str, Any]) -> Sequence[Example]:
        input_path = task_config["input_path"]
        story_field = task_config.get("story_field", "story")
        prompt_field = task_config.get("prompt_field", "prompt")
        id_field = task_config.get("id_field", "id")

        examples: List[Example] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                examples.append(Example(
                    id=str(rec[id_field]),
                    data={
                        "story": rec[story_field],
                        "prompt": rec.get(prompt_field, ""),
                        "word_count": rec.get("word_count"),
                    },
                ))
        return examples

    def build_messages(self, ex: Example, run_config: Dict[str, Any]) -> List[Dict[str, str]]:
        task_params = run_config.get("task_params") or {}
        template = self._judge_prompt_template(task_params)

        filled = template
        filled = filled.replace("{writing_prompt}", ex.data["prompt"])
        filled = filled.replace("{story_text}", ex.data["story"])

        dims_text = "\n".join(
            f"  {'[NEGATIVE]' if d in _NEGATIVE_SET else '[POSITIVE]'} {d} (weight: {_DIM_WEIGHTS[d]}×)"
            for d in ALL_DIMS
        )
        filled = filled.replace("{dimensions_list}", dims_text)

        return [{"role": "user", "content": filled}]

    def parse_response(self, ex: Example, provider_output: Any, run_config: Dict[str, Any]) -> Dict[str, Any]:
        raw = str(provider_output)
        raw_scores = _parse_scores(raw)
        analysis = _extract_analysis(raw)

        dim_raw: Dict[str, Optional[float]] = {}
        dim_adjusted: Dict[str, Optional[float]] = {}

        for d in ALL_DIMS:
            raw_val = _match_dim(d, raw_scores)
            dim_raw[d] = raw_val
            if raw_val is not None:
                dim_adjusted[d] = round(_adjust_score(d, raw_val), 4)
            else:
                dim_adjusted[d] = None

        # Weighted aggregate
        rated = {d: v for d, v in dim_adjusted.items() if v is not None}
        if rated:
            w_sum = sum(v * _DIM_WEIGHTS[d] for d, v in rated.items())
            w_total = sum(_DIM_WEIGHTS[d] for d in rated)
            weighted_score = round(w_sum / w_total / 20.0 * 100.0, 2)  # scale to 0-100
        else:
            weighted_score = None

        record: Dict[str, Any] = {
            "id": ex.id,
            "prompt": ex.data["prompt"],
            "story_word_count": ex.data.get("word_count"),
            "analysis": analysis,
            "raw_judge_text": raw,
        }
        for d in ALL_DIMS:
            record[f"raw_{_key(d)}"] = dim_raw[d]
        for d in ALL_DIMS:
            record[f"score_{_key(d)}"] = dim_adjusted[d]
        record["weighted_score"] = weighted_score
        record["num_dims_rated"] = len(rated)

        return record

    def aggregate(
        self,
        records: List[Dict[str, Any]],
        run_config: Dict[str, Any],
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        valid = [r for r in records if r.get("weighted_score") is not None]
        if not valid:
            return {"num_records": len(records), "num_valid": 0}

        scores = [r["weighted_score"] for r in valid]

        per_dim: Dict[str, Any] = {}
        for d in ALL_DIMS:
            k = f"score_{_key(d)}"
            vals = [r[k] for r in valid if r.get(k) is not None]
            if vals:
                per_dim[d] = {
                    "mean": round(float(np.mean(vals)), 2),
                    "std": round(float(np.std(vals)), 2),
                    "is_negative": d in _NEGATIVE_SET,
                    "weight": _DIM_WEIGHTS[d],
                }

        return {
            "num_records": len(records),
            "num_valid": len(valid),
            "weighted_score": {
                "mean": round(float(np.mean(scores)), 2),
                "median": round(float(np.median(scores)), 2),
                "std": round(float(np.std(scores)), 2),
            },
            "per_dimension": per_dim,
        }

    def format_record_preview(self, record: Dict[str, Any], run_config: Dict[str, Any], max_len: int = 600) -> str:
        lines = [
            f"id: {record.get('id')}",
            f"weighted_score: {record.get('weighted_score')} / 100",
            f"dims rated: {record.get('num_dims_rated')} / {len(ALL_DIMS)}",
            "",
            "Positive dimensions:",
        ]
        for d in POSITIVE_DIMS:
            lines.append(f"  {d}: {record.get(f'score_{_key(d)}')}")
        lines.append("\nNegative dimensions (adjusted; higher=better):")
        for d in NEGATIVE_DIMS:
            raw = record.get(f"raw_{_key(d)}")
            adj = record.get(f"score_{_key(d)}")
            tag = " [5× weight]" if d == _FORCED_POETRY else ""
            lines.append(f"  {d}: {adj}  (raw: {raw}){tag}")
        analysis = (record.get("analysis") or "")[:300]
        if analysis:
            lines.append(f"\nAnalysis (truncated):\n  {analysis}...")
        return "\n".join(lines)

    def format_summary(self, summary: Dict[str, Any], run_config: Dict[str, Any], max_len: int = 4000) -> str:
        if not summary:
            return "(no summary)"
        ws = summary.get("weighted_score") or {}
        lines = [
            f"EQ-Bench LongForm Score (0-100): "
            f"mean={ws.get('mean')}  median={ws.get('median')}  std={ws.get('std')}",
            f"Records: {summary.get('num_valid')} / {summary.get('num_records')} valid",
            "",
            "Per-dimension means (adjusted; higher always better):",
        ]
        for d, info in summary.get("per_dimension", {}).items():
            tag = f"  [neg→inv, {info['weight']}×]" if info.get("is_negative") else ""
            lines.append(f"  {d}: {info['mean']} ±{info['std']}{tag}")
        return "\n".join(lines)
