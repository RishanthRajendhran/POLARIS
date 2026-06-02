# tasks/longbench_write.py
"""
LongBench-Write evaluation task.

6 fixed dimensions scored 1-5 each. Overall score = mean of all 6.
Adapted from THUDM/LongWriter evaluation/eval_quality.py.
"""
from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List, Optional, Sequence

from core.task_base import Task, Example

DIMENSIONS = [
    "Relevance", "Accuracy", "Coherence", "Clarity",
    "Breadth and Depth", "Reading Experience",
]

JUDGE_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "prompts", "longbench_write_judge.txt"
)

LBW_JSON_SCHEMA = {
    "name": "longbench_write_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "Analysis": {"type": "string"},
            "Relevance": {"type": "integer", "minimum": 1, "maximum": 5},
            "Accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
            "Coherence": {"type": "integer", "minimum": 1, "maximum": 5},
            "Clarity": {"type": "integer", "minimum": 1, "maximum": 5},
            "Breadth_and_Depth": {"type": "integer", "minimum": 1, "maximum": 5},
            "Reading_Experience": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": [
            "Analysis", "Relevance", "Accuracy", "Coherence",
            "Clarity", "Breadth_and_Depth", "Reading_Experience",
        ],
        "additionalProperties": False,
    },
}


class LongBenchWriteTask(Task):
    name = "longbench_write"

    def __init__(self):
        super().__init__()
        self._prompts_index: Dict[str, Dict] = {}

    # ── 1. Load examples ──
    def load_examples(self, task_config: Dict[str, Any]) -> Sequence[Example]:
        path = task_config["input_path"]
        story_field = task_config.get("story_field", "story")
        prompt_field = task_config.get("prompt_field", "prompt")
        id_field = task_config.get("id_field", "id")

        examples: List[Example] = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                examples.append(Example(
                    id=str(row.get(id_field, len(examples))),
                    data={
                        "prompt": row.get(prompt_field, ""),
                        "story": row.get(story_field, ""),
                        **{k: v for k, v in row.items() if k not in (id_field, prompt_field, story_field)},
                    },
                ))
        return examples

    # ── 2. Build messages ──
    def build_messages(self, ex: Example, run_config: Dict[str, Any]) -> List[Dict[str, str]]:
        tp = run_config.get("task_params", {}) or {}
        d = ex.data

        # Load prompts index for target_length lookup
        if not self._prompts_index:
            prompts_path = tp.get("prompts_path", "")
            prompts_id_field = tp.get("prompts_id_field", "uid")
            if prompts_path and os.path.isfile(prompts_path):
                with open(prompts_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        pid = str(row.get(prompts_id_field, ""))
                        self._prompts_index[pid] = row

        # Attach target_length from prompts if not already in data
        if "target_length" not in d or d["target_length"] is None:
            prompt_row = self._prompts_index.get(ex.id)
            if prompt_row and "target_length" in prompt_row:
                d["target_length"] = prompt_row["target_length"]

        judge_prompt_path = tp.get("judge_prompt_path", JUDGE_PROMPT_PATH)
        if os.path.isfile(judge_prompt_path):
            with open(judge_prompt_path) as f:
                template = f.read()
        else:
            template = "Evaluate:\n{prompt}\n\n{response}"

        user_content = template.replace("{prompt}", d.get("prompt", ""))
        user_content = user_content.replace("{response}", d.get("story", ""))

        system_msg = (
            "You are an expert text quality evaluator. Score each dimension strictly 1-5. "
            "Output ONLY a JSON object. No extra text."
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

    # ── 3. Response format ──
    def get_response_format(self, provider: str, run_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if provider == "vertex":
            return {
                "response_mime_type": "application/json",
                "response_schema": LBW_JSON_SCHEMA["schema"],
            }
        return None

    # ── 4. Parse response ──
    def parse_response(self, ex: Example, provider_output: Any, run_config: Dict[str, Any]) -> Dict[str, Any]:
        d = ex.data

        if isinstance(provider_output, dict):
            parsed = provider_output
        elif isinstance(provider_output, str):
            txt = provider_output.strip()
            if "```" in txt:
                import re
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.DOTALL)
                if m:
                    txt = m.group(1)
            try:
                parsed = json.loads(txt)
            except Exception:
                parsed = {}
        else:
            parsed = {}

        dim_scores = {}
        # Handle both "Breadth and Depth" and "Breadth_and_Depth" key variants
        key_map = {
            "Relevance": ["Relevance"],
            "Accuracy": ["Accuracy"],
            "Coherence": ["Coherence"],
            "Clarity": ["Clarity"],
            "Breadth and Depth": ["Breadth_and_Depth", "Breadth and Depth"],
            "Reading Experience": ["Reading_Experience", "Reading Experience"],
        }

        for dim, keys in key_map.items():
            for k in keys:
                v = parsed.get(k)
                if isinstance(v, (int, float)) and 1 <= v <= 5:
                    dim_scores[dim] = float(v)
                    break

        valid = list(dim_scores.values())
        # Sq per dimension: (raw_score - 1) * 25, maps 1-5 → 0-100
        # Matches LongWriter eval_quality.py formula exactly
        sq_per_dim = {dim: (s - 1) * 25 for dim, s in dim_scores.items()}
        sq = statistics.mean(sq_per_dim.values()) if sq_per_dim else None

        # Length score (Sl) — matches LongWriter eval_length.py formula
        word_count = d.get("word_count")
        target_length = d.get("target_length")
        sl = None
        if word_count and target_length and target_length > 0:
            x, y = target_length, word_count
            if y > x:
                sl = 100 * max(0, 1.0 - (y / x - 1) / 3)
            else:
                sl = 100 * max(0, 1.0 - (x / y - 1) / 2)

        # Combined score: S = S_Q × (S_L / 100)
        # Matches LongWriter paper: quality penalized by length adherence
        combined_score = None
        if sq is not None and sl is not None:
            combined_score = round(sq * (sl / 100.0), 2)

        rec = {
            "id": ex.id,
            "prompt": d.get("prompt", ""),
            "story": d.get("story", "")[:200],
            "analysis": parsed.get("Analysis", ""),
            "dim_scores": dim_scores,
            "sq_per_dim": sq_per_dim,
            "quality_score_sq": sq,
            "length_score_sl": sl,
            "combined_score": combined_score,
            "word_count": word_count,
            "target_length": target_length,
            "longbench_write_score": statistics.mean(valid) if valid else None,
            "num_dims_scored": len(valid),
        }
        return rec

    # ── 5. Aggregate ──
    def aggregate(self, records: List[Dict[str, Any]], run_config: Dict[str, Any],
                  output_dir: str) -> Optional[Dict[str, Any]]:
        if not records:
            return {"num_records": 0}

        all_scores = [r["longbench_write_score"] for r in records if r.get("longbench_write_score") is not None]

        per_dim: Dict[str, List[float]] = {}
        for r in records:
            for dim, score in r.get("dim_scores", {}).items():
                per_dim.setdefault(dim, []).append(score)

        dim_stats = {}
        for dim in DIMENSIONS:
            vals = per_dim.get(dim, [])
            if vals:
                dim_stats[dim] = {
                    "mean": statistics.mean(vals),
                    "median": statistics.median(vals),
                    "stdev": statistics.stdev(vals) if len(vals) > 1 else 0,
                    "count": len(vals),
                }

        # Quality score (S_Q) aggregates
        sq_vals = [r["quality_score_sq"] for r in records if r.get("quality_score_sq") is not None]
        # Length score (S_L) aggregates
        sl_vals = [r["length_score_sl"] for r in records if r.get("length_score_sl") is not None]
        # Combined score aggregates
        combined_vals = [r["combined_score"] for r in records if r.get("combined_score") is not None]

        return {
            "num_records": len(records),
            "num_scored": len(all_scores),
            "longbench_write_score_mean": statistics.mean(all_scores) if all_scores else None,
            "longbench_write_score_median": statistics.median(all_scores) if all_scores else None,
            "longbench_write_score_stdev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
            "quality_score_sq_mean": round(statistics.mean(sq_vals), 2) if sq_vals else None,
            "length_score_sl_mean": round(statistics.mean(sl_vals), 2) if sl_vals else None,
            "combined_score_mean": round(statistics.mean(combined_vals), 2) if combined_vals else None,
            "combined_score_median": round(statistics.median(combined_vals), 2) if combined_vals else None,
            "combined_score_stdev": round(statistics.stdev(combined_vals), 2) if len(combined_vals) > 1 else 0,
            "per_dimension": dim_stats,
        }
