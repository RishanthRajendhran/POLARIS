# tasks/story_elo.py
"""
EQ-Bench style per-dimension pairwise evaluation for Elo ranking.

Each pair of stories is compared across 9 dimensions using a plus-count
system (A+, B+++, etc.). The verdict is derived from the aggregate
plus_diff across dimensions.

Designed for full round-robin Elo computation with Glicko-2.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence

from core.task_base import Task, Example


ELO_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "prompts", "eqbench_pairwise.txt"
)

# Verdict mapping from plus_diff
def _verdict_from_diff(diff: int) -> str:
    if diff >= 10:
        return "A>>B"
    elif diff > 0:
        return "A>B"
    elif diff == 0:
        return "tie"
    elif diff > -10:
        return "B>A"
    else:
        return "B>>A"


VERDICT_SCORES: Dict[str, float] = {
    "A>>B": 1.0,
    "A>B": 0.75,
    "tie": 0.5,
    "B>A": 0.25,
    "B>>A": 0.0,
}

# Keys that are not dimension scores
_SKIP_KEYS = frozenset({
    "chain_of_thought_reasoning", "verdict", "justification",
    "per_dimension", "raw_output",
})


class StoryEloTask(Task):
    name = "story_elo"

    def __init__(self):
        super().__init__()

    # ── 1. Load examples ──
    def load_examples(self, task_config: Dict[str, Any]) -> Sequence[Example]:
        mode = task_config.get("mode", "two_files")
        if mode != "two_files":
            raise ValueError(f"story_elo only supports mode=two_files, got {mode}")

        file_a = task_config["file_a"]
        file_b = task_config["file_b"]
        pairing_key = task_config.get("pairing_key", "id")
        story_field = task_config.get("story_field", "story")
        prompt_field = task_config.get("prompt_field", "prompt")
        model_a_name = task_config.get("default_model_a_name", "model_a")
        model_b_name = task_config.get("default_model_b_name", "model_b")
        dual_position = task_config.get("dual_position", True)
        strip_thinking = task_config.get("strip_thinking", True)

        # Load both files keyed by pairing_key
        def _load(path):
            records = {}
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = str(row.get(pairing_key, ""))
                    records[key] = row
            return records

        recs_a = _load(file_a)
        recs_b = _load(file_b)
        common_keys = sorted(set(recs_a.keys()) & set(recs_b.keys()))

        def _get_story(row):
            text = str(row.get(story_field, ""))
            if strip_thinking:
                # Strip <think>...</think>
                text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
                # Strip Gemma4 <|channel>thought...<channel|>
                text = re.sub(r"<\|channel>thought.*?<channel\|>\s*", "", text, flags=re.DOTALL)
            return text

        examples: List[Example] = []
        for key in common_keys:
            ra, rb = recs_a[key], recs_b[key]
            story_a = _get_story(ra)
            story_b = _get_story(rb)
            prompt = ra.get(prompt_field, rb.get(prompt_field, ""))

            # Position 0: A=file_a, B=file_b
            data_pos0 = {
                "prompt": prompt,
                "story_a": story_a,
                "story_b": story_b,
                "model_a": model_a_name,
                "model_b": model_b_name,
                "prompt_id": key,
            }
            examples.append(Example(id=f"{key}__pos0", data=data_pos0))

            if dual_position:
                # Position 1: A=file_b, B=file_a (swapped)
                data_pos1 = {
                    "prompt": prompt,
                    "story_a": story_b,
                    "story_b": story_a,
                    "model_a": model_b_name,
                    "model_b": model_a_name,
                    "prompt_id": key,
                    "position_swapped": True,
                }
                examples.append(Example(id=f"{key}__pos1", data=data_pos1))

        return examples

    # ── 2. Build messages ──
    def build_messages(self, ex: Example, run_config: Dict[str, Any]) -> List[Dict[str, str]]:
        d = ex.data
        tp = run_config.get("task_params", {}) or {}

        prompt = d.get("prompt", "")
        story_a = d.get("story_a", "")
        story_b = d.get("story_b", "")

        # Load prompt template
        prompt_path = tp.get("prompt_path", ELO_PROMPT_PATH)
        if os.path.isfile(prompt_path):
            with open(prompt_path) as f:
                template = f.read()
        else:
            template = (
                "[WRITING PROMPT]\n{prompt}\n[/WRITING PROMPT]\n\n"
                "[WRITER A]\n{story_a}\n[/WRITER A]\n\n"
                "[WRITER B]\n{story_b}\n[/WRITER B]\n"
            )

        user_content = template.replace("{prompt}", prompt)
        user_content = user_content.replace("{story_a}", story_a)
        user_content = user_content.replace("{story_b}", story_b)

        return [{"role": "user", "content": user_content}]

    # ── 3. Response format ──
    def get_response_format(self, provider: str, run_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # EQ-Bench format relies on prompting, not schema enforcement
        return None

    # ── 4. Parse response ──
    def parse_response(self, ex: Example, provider_output: Any, run_config: Dict[str, Any]) -> Dict[str, Any]:
        d = ex.data

        # Parse JSON from output
        if isinstance(provider_output, dict):
            parsed = provider_output
        elif isinstance(provider_output, str):
            txt = provider_output.strip()
            if "```" in txt:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.DOTALL)
                if m:
                    txt = m.group(1).strip()
            if txt.startswith("{"):
                try:
                    parsed = json.loads(txt)
                except Exception:
                    parsed = {}
            else:
                parsed = {}
        else:
            parsed = {}

        # Extract per-dimension scores from top-level keys
        per_dimension = {}
        plus_a = 0
        plus_b = 0

        for key, val in parsed.items():
            if key in _SKIP_KEYS:
                continue
            if not isinstance(val, str):
                continue
            val_stripped = val.strip()
            if val_stripped.startswith("A") or val_stripped.startswith("B"):
                per_dimension[key] = val_stripped
                if val_stripped.startswith("A"):
                    plus_a += val_stripped.count("+")
                elif val_stripped.startswith("B"):
                    plus_b += val_stripped.count("+")

        plus_diff = plus_a - plus_b
        verdict = _verdict_from_diff(plus_diff)
        score_a = VERDICT_SCORES.get(verdict, 0.5)

        cot = parsed.get("chain_of_thought_reasoning", "")

        rec = {
            "id": ex.id,
            "prompt_id": d.get("prompt_id", ""),
            "model_a": d.get("model_a", "model_a"),
            "model_b": d.get("model_b", "model_b"),
            "verdict": verdict,
            "score_a": score_a,
            "plus_a": plus_a,
            "plus_b": plus_b,
            "plus_diff": plus_diff,
            "per_dimension": per_dimension,
            "chain_of_thought": cot,
            "raw_output": parsed,
        }

        if d.get("position_swapped"):
            rec["position_swapped"] = True

        return rec

    # ── 5. Aggregate ──
    def aggregate(self, records: List[Dict[str, Any]], run_config: Dict[str, Any],
                  output_dir: str) -> Optional[Dict[str, Any]]:
        if not records:
            return {"num_records": 0}

        task_config = (run_config.get("task_config") or {})
        dual = task_config.get("dual_position", True)

        if dual:
            return self._aggregate_dual(records)

        # Simple aggregation
        a_wins = sum(1 for r in records if r["verdict"] in ("A>>B", "A>B"))
        b_wins = sum(1 for r in records if r["verdict"] in ("B>>A", "B>A"))
        ties = sum(1 for r in records if r["verdict"] == "tie")
        total = len(records)

        return {
            "num_records": total,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "a_win_pct": round(a_wins / total * 100, 1) if total else 0,
            "avg_plus_diff": round(statistics.mean([r["plus_diff"] for r in records]), 2),
        }

    def _aggregate_dual(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate dual-position results, correcting for position swap."""
        # Group by prompt_id
        by_prompt: Dict[str, Dict[str, Any]] = {}
        for r in records:
            pid = r.get("prompt_id", r["id"].rsplit("__", 1)[0])
            pos = "pos1" if r.get("position_swapped") or "__pos1" in r["id"] else "pos0"
            by_prompt.setdefault(pid, {})[pos] = r

        # For each prompt, determine the original model_a winner
        # pos0: A=orig_a, B=orig_b. "A wins" means orig_a wins.
        # pos1: A=orig_b, B=orig_a. "A wins" means orig_b wins.
        orig_a_wins = 0
        orig_b_wins = 0
        ties = 0
        plus_diffs = []

        for pid, positions in by_prompt.items():
            for pos_key, r in positions.items():
                pd = r["plus_diff"]
                if pos_key == "pos1":
                    pd = -pd  # Flip for swapped position
                plus_diffs.append(pd)

                if pd > 0:
                    orig_a_wins += 1
                elif pd < 0:
                    orig_b_wins += 1
                else:
                    ties += 1

        total = orig_a_wins + orig_b_wins + ties
        model_a = records[0].get("model_a", "model_a") if records else "?"
        model_b = records[0].get("model_b", "model_b") if records else "?"
        # Find original model names (from pos0)
        for r in records:
            if not r.get("position_swapped") and "__pos0" in r["id"]:
                model_a = r["model_a"]
                model_b = r["model_b"]
                break

        return {
            "num_records": len(records),
            "num_prompts": len(by_prompt),
            "model_a": model_a,
            "model_b": model_b,
            "orig_a_wins": orig_a_wins,
            "orig_b_wins": orig_b_wins,
            "ties": ties,
            "orig_a_win_pct": round(orig_a_wins / total * 100, 1) if total else 0,
            "orig_b_win_pct": round(orig_b_wins / total * 100, 1) if total else 0,
            "avg_plus_diff": round(statistics.mean(plus_diffs), 2) if plus_diffs else 0,
        }

    # ── 6. Format record preview ──
    def format_record_preview(self, record: Dict[str, Any], run_config: Dict[str, Any], **kwargs) -> str:
        lines = [
            f"ID: {record.get('id', '?')}",
            f"Verdict: {record.get('verdict', '?')} (plus_diff={record.get('plus_diff', '?')})",
            f"Dims: {record.get('per_dimension', {})}",
        ]
        cot = record.get("chain_of_thought", "")
        if cot:
            lines.append(f"CoT: {cot[:200]}...")
        return "\n".join(lines)
