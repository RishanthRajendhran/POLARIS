# tasks/writingbench.py
"""
WritingBench evaluation task.

Follows the official WritingBench evaluation method (X-PLUG/WritingBench):
each prompt has a per-query checklist of 5 criteria. The judge is called
once per criterion (5 calls per story), returning {"score": int, "reason": str}.
The overall WritingBench score is the mean of all criterion scores.
"""
from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List, Optional, Sequence

from core.task_base import Task, Example


JUDGE_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "prompts", "writingbench_judge.txt"
)


class WritingBenchTask(Task):
    name = "writingbench"

    def __init__(self):
        super().__init__()
        self._prompts_index: Dict[str, Dict] = {}

    # ── 1. Load examples ──
    # Each story × criterion pair becomes a separate example so the judge
    # evaluates one criterion at a time (matching official WritingBench).
    def load_examples(self, task_config: Dict[str, Any]) -> Sequence[Example]:
        path = task_config["input_path"]
        story_field = task_config.get("story_field", "story")
        prompt_field = task_config.get("prompt_field", "prompt")
        id_field = task_config.get("id_field", "id")

        # Load prompts index for checklist lookup
        prompts_path = task_config.get("prompts_path", "")
        prompts_id_field = task_config.get("prompts_id_field", "uid")
        if prompts_path and os.path.isfile(prompts_path):
            with open(prompts_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    pid = str(row.get(prompts_id_field, ""))
                    self._prompts_index[pid] = row

        examples: List[Example] = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                story_id = str(row.get(id_field, len(examples)))
                prompt_text = row.get(prompt_field, "")
                story_text = row.get(story_field, "")

                # Find checklist
                checklist = row.get("checklist")
                if not checklist:
                    prompt_row = self._prompts_index.get(story_id)
                    if not prompt_row:
                        # Try matching by prompt text prefix
                        for pid, prow in self._prompts_index.items():
                            if prow.get("sft_prompt", "")[:100] == prompt_text[:100]:
                                prompt_row = prow
                                break
                    if prompt_row:
                        checklist = prompt_row.get("checklist", [])

                if not checklist:
                    checklist = []

                # Create one example per criterion
                for ci, criterion in enumerate(checklist):
                    ex_id = f"{story_id}__c{ci}"
                    examples.append(Example(
                        id=ex_id,
                        data={
                            "prompt": prompt_text,
                            "story": story_text,
                            "story_id": story_id,
                            "criterion_index": ci,
                            "criterion_name": criterion["name"],
                            "criterion": criterion,
                            "num_criteria": len(checklist),
                            **{k: v for k, v in row.items()
                               if k not in (id_field, prompt_field, story_field, "checklist")},
                        },
                    ))
        return examples

    # ── 2. Build messages ──
    def build_messages(self, ex: Example, run_config: Dict[str, Any]) -> List[Dict[str, str]]:
        tp = run_config.get("task_params", {}) or {}
        d = ex.data

        criterion = d["criterion"]

        # Build single-criterion block
        criteria_block = f"Criterion: {criterion['name']}\n"
        criteria_block += f"  Description: {criterion['criteria_description']}\n"
        for anchor in ["1-2", "3-4", "5-6", "7-8", "9-10"]:
            if anchor in criterion:
                criteria_block += f"  Score {anchor}: {criterion[anchor]}\n"

        # Load judge prompt template
        judge_prompt_path = tp.get("judge_prompt_path", JUDGE_PROMPT_PATH)
        if os.path.isfile(judge_prompt_path):
            with open(judge_prompt_path) as f:
                template = f.read()
        else:
            template = (
                "Evaluate the response to the writing task on the given criterion.\n\n"
                "WRITING TASK:\n{prompt}\n\nRESPONSE:\n{response}\n\nCRITERION:\n{criteria_block}\n"
            )

        prompt_text = d.get("prompt", "")
        response_text = d.get("story", "")

        user_content = template.replace("{prompt}", prompt_text)
        user_content = user_content.replace("{response}", response_text)
        user_content = user_content.replace("{criteria_block}", criteria_block)

        system_msg = (
            "You are an expert evaluator with extensive experience in evaluating "
            "response of given query. "
            "Output ONLY a JSON object with keys \"score\" (integer 1-10) and "
            "\"reason\" (string with specific justification). No extra text."
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
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                },
            }
        elif provider == "openai":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "writingbench_criterion_score",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["score", "reason"],
                        "additionalProperties": False,
                    },
                },
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
            if txt.startswith("{"):
                try:
                    parsed = json.loads(txt)
                except Exception:
                    parsed = {}
            else:
                parsed = {}
        else:
            parsed = {}

        score = parsed.get("score")
        reason = parsed.get("reason", "")

        # Validate score range
        if isinstance(score, (int, float)) and 1 <= score <= 10:
            score = int(score)
        else:
            score = None

        rec = {
            "id": ex.id,
            "story_id": d.get("story_id", ""),
            "criterion_index": d.get("criterion_index", 0),
            "criterion_name": d.get("criterion_name", ""),
            "score": score,
            "reason": reason,
            "raw_output": parsed,
        }
        return rec

    # ── 5. Aggregate ──
    def aggregate(self, records: List[Dict[str, Any]], run_config: Dict[str, Any],
                  output_dir: str) -> Optional[Dict[str, Any]]:
        if not records:
            return {"num_records": 0}

        # Group by story_id
        by_story: Dict[str, List[Dict]] = {}
        for r in records:
            sid = r.get("story_id", r.get("id", ""))
            by_story.setdefault(sid, []).append(r)

        # Per-story score = mean of criterion scores
        story_scores = []
        for sid, recs in by_story.items():
            valid = [r["score"] for r in recs if r.get("score") is not None]
            if valid:
                story_scores.append(statistics.mean(valid))

        # Per-criterion aggregation
        crit_scores: Dict[str, List[float]] = {}
        for r in records:
            name = r.get("criterion_name", "")
            if r.get("score") is not None and name:
                crit_scores.setdefault(name, []).append(r["score"])

        per_criterion = {}
        for name, vals in sorted(crit_scores.items()):
            per_criterion[name] = {
                "mean": round(statistics.mean(vals), 2),
                "median": round(statistics.median(vals), 2),
                "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
                "count": len(vals),
            }

        num_scored = sum(1 for r in records if r.get("score") is not None)

        summary = {
            "num_records": len(records),
            "num_scored": num_scored,
            "num_stories": len(by_story),
            "writingbench_score_mean": round(statistics.mean(story_scores), 2) if story_scores else None,
            "writingbench_score_median": round(statistics.median(story_scores), 2) if story_scores else None,
            "writingbench_score_stdev": round(statistics.stdev(story_scores), 2) if len(story_scores) > 1 else 0,
            "per_criterion": per_criterion,
        }
        return summary

    # ── 6. Format record preview ──
    def format_record_preview(self, record: Dict[str, Any], run_config: Dict[str, Any], **kwargs) -> str:
        lines = []
        lines.append(f"ID: {record.get('id', '?')}")
        lines.append(f"Story: {record.get('story_id', '?')}")
        lines.append(f"Criterion: {record.get('criterion_name', '?')}")
        lines.append(f"Score: {record.get('score', '?')}")
        reason = record.get("reason", "")
        if reason:
            lines.append(f"Reason: {reason[:200]}...")
        return "\n".join(lines)
