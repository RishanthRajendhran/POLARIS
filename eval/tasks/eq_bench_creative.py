# tasks/eq_bench_creative.py
"""
EQ-Bench Creative Writing v3 judge task.

Scores a generated story against 22 EQ-Bench criteria using a plain-text
judge prompt. Negative criteria (lower raw score = worse writing) are
inverted so that higher is always better after adjustment.

Config shape:
  task: eq_bench_creative
  task_config:
    input_path: <path to generated stories JSONL>
    story_field: story          # default: "story"
    prompt_field: prompt        # default: "prompt"
    id_field: id                # default: "id"
  task_params:
    judge_prompt_path: <optional override; defaults to eq_bench_creative_judge.txt>

Output record fields:
  id, prompt, story_word_count, analysis, raw_judge_text,
  raw_<criterion> (0-20, raw from judge),
  score_<criterion> (0-20, inverted for negatives; higher always better),
  rubric_score (mean of score_* for rated criteria, 0-20),
  eqbench_creative_score (rubric_score × 5, 0-100),
  num_criteria_rated
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.task_base import Task, Example

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

POSITIVE_CRITERIA: List[str] = [
    "Adherence to Instructions",
    "Believable Character Actions",
    "Nuanced Characters",
    "Consistent Voice/Tone of Writing",
    "Imagery and Descriptive Quality",
    "Elegant Prose",
    "Emotionally Engaging",
    "Emotionally Complex",
    "Coherent",
    "Well-earned Lightness or Darkness",
    "Sentences Flow Naturally",
    "Overall Reader Engagement",
    "Overall Impression",
]

NEGATIVE_CRITERIA: List[str] = [
    "Meandering",
    "Weak Dialogue",
    "Tell-Don't-Show",
    "Unsurprising or Uncreative",
    "Amateurish",
    "Purple Prose",
    "Overwrought",
    "Incongruent Ending Positivity",
    "Unearned Transformations",
]

ALL_CRITERIA: List[str] = POSITIVE_CRITERIA + NEGATIVE_CRITERIA
_NEGATIVE_SET = set(NEGATIVE_CRITERIA)

_CRITERIA_TEXT = "\n".join(ALL_CRITERIA)
_LOWER_IS_BETTER_TEXT = "\n".join(NEGATIVE_CRITERIA)

# Default judge prompt path relative to this file
_DEFAULT_JUDGE_PROMPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "prompts", "eq_bench_creative_judge.txt")
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches lines like:  "Metric Name: [15]"  or  "Metric Name: 15"
_SCORE_LINE_RE = re.compile(r"^(.+?)\s*:\s*\[?(\d+(?:\.\d+)?)\]?\s*$")


def _key(criterion: str) -> str:
    """Convert a criterion name to a safe snake_case dict key."""
    return (
        criterion.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
    )


def _parse_scores(raw_text: str) -> Dict[str, float]:
    """Extract metric scores from the [Scores] section of judge output."""
    # Isolate [Scores] block first; fall back to full text
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


def _match_criterion(criterion: str, raw_scores: Dict[str, float]) -> Optional[float]:
    """Match a canonical criterion name against the judge's raw score keys."""
    # Exact match
    if criterion in raw_scores:
        return raw_scores[criterion]
    # Case-insensitive exact
    crit_lower = criterion.lower()
    for k, v in raw_scores.items():
        if k.lower() == crit_lower:
            return v
    # Prefix match on first 10 chars
    prefix = crit_lower[:10]
    for k, v in raw_scores.items():
        if k.lower().startswith(prefix):
            return v
    return None


def _extract_analysis(raw_text: str) -> str:
    m = re.search(r"\[Analysis\](.*?)(?:\[Scores\]|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class EqBenchCreativeTask(Task):
    name = "eq_bench_creative"

    def _judge_prompt_template(self, task_params: Dict[str, Any]) -> str:
        path = task_params.get("judge_prompt_path") or _DEFAULT_JUDGE_PROMPT_PATH
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ---- 1. Load examples ----

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

    # ---- 2. Build messages ----

    def build_messages(self, ex: Example, run_config: Dict[str, Any]) -> List[Dict[str, str]]:
        task_params = run_config.get("task_params") or {}
        template = self._judge_prompt_template(task_params)

        # Use explicit replace to avoid issues with curly braces in story text
        filled = template
        filled = filled.replace("{writing_prompt}", ex.data["prompt"])
        filled = filled.replace("{test_model_response}", ex.data["story"])
        filled = filled.replace("{creative_writing_criteria}", _CRITERIA_TEXT)
        filled = filled.replace("{lower_is_better_criteria}", _LOWER_IS_BETTER_TEXT)

        return [{"role": "user", "content": filled}]

    # ---- 4. Parse response ----

    def parse_response(self, ex: Example, provider_output: Any, run_config: Dict[str, Any]) -> Dict[str, Any]:
        raw = str(provider_output)
        raw_scores = _parse_scores(raw)
        analysis = _extract_analysis(raw)

        # Per-criterion raw and adjusted scores
        criterion_raw: Dict[str, Optional[float]] = {}
        criterion_adjusted: Dict[str, Optional[float]] = {}

        for c in ALL_CRITERIA:
            raw_val = _match_criterion(c, raw_scores)
            criterion_raw[c] = raw_val
            if raw_val is not None:
                criterion_adjusted[c] = (20.0 - raw_val) if c in _NEGATIVE_SET else raw_val
            else:
                criterion_adjusted[c] = None

        rated = {c: v for c, v in criterion_adjusted.items() if v is not None}
        rubric_score = round(sum(rated.values()) / len(rated), 4) if rated else None
        eqbench_creative_score = round(rubric_score * 5, 2) if rubric_score is not None else None

        record: Dict[str, Any] = {
            "id": ex.id,
            "prompt": ex.data["prompt"],
            "story_word_count": ex.data.get("word_count"),
            "analysis": analysis,
            "raw_judge_text": raw,
        }
        for c in ALL_CRITERIA:
            record[f"raw_{_key(c)}"] = criterion_raw[c]
        for c in ALL_CRITERIA:
            record[f"score_{_key(c)}"] = criterion_adjusted[c]
        record["rubric_score"] = rubric_score
        record["eqbench_creative_score"] = eqbench_creative_score
        record["num_criteria_rated"] = len(rated)

        return record

    # ---- 5. Aggregate ----

    def aggregate(
        self,
        records: List[Dict[str, Any]],
        run_config: Dict[str, Any],
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        valid = [r for r in records if r.get("eqbench_creative_score") is not None]
        if not valid:
            return {"num_records": len(records), "num_valid": 0}

        eq_scores = [r["eqbench_creative_score"] for r in valid]
        rubric_scores = [r["rubric_score"] for r in valid]

        per_criterion: Dict[str, Any] = {}
        for c in ALL_CRITERIA:
            k = f"score_{_key(c)}"
            vals = [r[k] for r in valid if r.get(k) is not None]
            if vals:
                per_criterion[c] = {
                    "mean": round(float(np.mean(vals)), 2),
                    "is_negative": c in _NEGATIVE_SET,
                }

        return {
            "num_records": len(records),
            "num_valid": len(valid),
            "eqbench_creative_score": {
                "mean": round(float(np.mean(eq_scores)), 2),
                "median": round(float(np.median(eq_scores)), 2),
                "std": round(float(np.std(eq_scores)), 2),
            },
            "rubric_score": {
                "mean": round(float(np.mean(rubric_scores)), 4),
                "median": round(float(np.median(rubric_scores)), 4),
                "std": round(float(np.std(rubric_scores)), 4),
            },
            "per_criterion": per_criterion,
        }

    # ---- Preview helpers ----

    def format_record_preview(self, record: Dict[str, Any], run_config: Dict[str, Any], max_len: int = 600) -> str:
        lines = [
            f"id: {record.get('id')}",
            f"eqbench_creative_score: {record.get('eqbench_creative_score')} / 100  "
            f"(rubric: {record.get('rubric_score')} / 20)",
            f"criteria rated: {record.get('num_criteria_rated')} / {len(ALL_CRITERIA)}",
            "",
            "Positive criteria:",
        ]
        for c in POSITIVE_CRITERIA:
            lines.append(f"  {c}: {record.get(f'score_{_key(c)}')}")
        lines.append("\nNegative criteria (inverted, higher=better after inversion):")
        for c in NEGATIVE_CRITERIA:
            raw = record.get(f"raw_{_key(c)}")
            adj = record.get(f"score_{_key(c)}")
            lines.append(f"  {c}: {adj}  (raw: {raw})")
        analysis = (record.get("analysis") or "")[:300]
        if analysis:
            lines.append(f"\nAnalysis (truncated):\n  {analysis}...")
        return "\n".join(lines)

    def format_summary(self, summary: Dict[str, Any], run_config: Dict[str, Any], max_len: int = 4000) -> str:
        if not summary:
            return "(no summary)"
        eq = summary.get("eqbench_creative_score") or {}
        lines = [
            f"EQ-Bench Creative Score (0-100): "
            f"mean={eq.get('mean')}  median={eq.get('median')}  std={eq.get('std')}",
            f"Records: {summary.get('num_valid')} / {summary.get('num_records')} valid",
            "",
            "Per-criterion means (adjusted; higher always better):",
        ]
        for c, info in summary.get("per_criterion", {}).items():
            tag = "  [neg→inv]" if info.get("is_negative") else ""
            lines.append(f"  {c}: {info['mean']}{tag}")
        return "\n".join(lines)
