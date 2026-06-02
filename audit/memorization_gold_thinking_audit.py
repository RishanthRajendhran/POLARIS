#!/usr/bin/env python3
import argparse
import json
import random
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


THINK_TEMPLATE = "<think>\n{thoughts}\n</think>\n"


@dataclass
class AuditResult:
    attack_mode: str
    row_idx: int
    prefix_tokens: int
    title: str
    author: str
    anthology: str
    prompt_chars: int
    gold_thinking_chars: int
    gold_story_tokens: int
    gold_continuation_tokens: int
    generated_tokens: int
    exact_prefix_match_tokens: int
    longest_common_substring_tokens: int
    lcs_start_in_generation: int
    lcs_start_in_gold: int
    generated_text: str
    gold_continuation_text: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Gold-thinking memorization stress test for POLARIS checkpoints.')
    p.add_argument('--model-path', type=str, required=True)
    p.add_argument('--parquet-path', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--sample-size', type=int, default=100)
    p.add_argument('--attack-mode', type=str, default='gold_thinking', choices=['gold_thinking', 'prompt_only'])
    p.add_argument('--seed', type=int, default=3479)
    p.add_argument('--prefix-token-counts', type=int, nargs='+', default=[50, 100, 200])
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--torch-dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    return p.parse_args()


def get_dtype(name: str):
    return {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }[name]


def load_rows(parquet_path: str) -> List[Dict[str, Any]]:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    return rows


def exact_prefix_match_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def longest_common_substring_match(a: List[int], b: List[int]):
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size, match.a, match.b


def build_prompt_text(tokenizer, row: Dict[str, Any], story_prefix_text: str, attack_mode: str) -> str:
    messages = row['prompt']
    base = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if attack_mode == 'prompt_only':
        return base
    gold_thinking = row['ground_truth_reasoning'].rstrip()
    return base + THINK_TEMPLATE.format(thoughts=gold_thinking) + story_prefix_text


def extract_generated_story_text(text: str) -> str:
    m = re.match(r"\s*<think>.*?</think>\s*(.*)\Z", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return text


def load_completed_keys(results_path: Path) -> Set[Tuple[str, int, int]]:
    completed: Set[Tuple[str, int, int]] = set()
    if not results_path.exists():
        return completed
    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                completed.add((r.get("attack_mode", "gold_thinking"), int(r["row_idx"]), int(r["prefix_tokens"])))
            except Exception:
                continue
    return completed


def sample_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if sample_size >= len(indexed):
        return indexed
    rng = random.Random(seed)
    return rng.sample(indexed, sample_size)


def summarize(results: List[AuditResult]) -> Dict[str, Any]:
    by_prefix: Dict[int, List[AuditResult]] = {}
    for r in results:
        by_prefix.setdefault(r.prefix_tokens, []).append(r)

    summary: Dict[str, Any] = {
        'num_results': len(results),
        'by_prefix_tokens': {},
    }
    for k, vals in sorted(by_prefix.items()):
        prefix_stats = {
            'count': len(vals),
            'mean_exact_prefix_match_tokens': sum(v.exact_prefix_match_tokens for v in vals) / len(vals),
            'mean_longest_common_substring_tokens': sum(v.longest_common_substring_tokens for v in vals) / len(vals),
            'max_exact_prefix_match_tokens': max(v.exact_prefix_match_tokens for v in vals),
            'max_longest_common_substring_tokens': max(v.longest_common_substring_tokens for v in vals),
            'copied_span_ge_50_frac': sum(v.longest_common_substring_tokens >= 50 for v in vals) / len(vals),
            'copied_span_ge_100_frac': sum(v.longest_common_substring_tokens >= 100 for v in vals) / len(vals),
            'copied_span_ge_200_frac': sum(v.longest_common_substring_tokens >= 200 for v in vals) / len(vals),
        }
        summary['by_prefix_tokens'][str(k)] = prefix_stats
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        torch_dtype=get_dtype(args.torch_dtype),
        device_map='auto',
    )
    model.eval()

    rows = load_rows(args.parquet_path)
    sampled = sample_rows(rows, args.sample_size, args.seed)

    jsonl_path = out_dir / 'results.jsonl'
    completed_keys = load_completed_keys(jsonl_path)
    all_results: List[AuditResult] = []
    if completed_keys:
        print(f'Resuming from {len(completed_keys)} completed evaluations in {jsonl_path}', flush=True)
    with jsonl_path.open('a') as f_out:
        for row_idx, row in sampled:
            story_text = row.get('story') or row.get('extra_info', {}).get('story')
            if not story_text:
                continue
            story_ids = tokenizer.encode(story_text, add_special_tokens=False)
            meta = row.get('extra_info', {}) or {}
            prefix_values = args.prefix_token_counts if args.attack_mode == 'gold_thinking' else [0]
            for prefix_k in prefix_values:
                key = (args.attack_mode, row_idx, prefix_k)
                if key in completed_keys:
                    continue
                if prefix_k >= len(story_ids):
                    continue
                if args.attack_mode == 'prompt_only':
                    prefix_ids = []
                    gold_continuation_ids = story_ids
                    story_prefix_text = ''
                else:
                    prefix_ids = story_ids[:prefix_k]
                    gold_continuation_ids = story_ids[prefix_k:]
                    story_prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
                gold_continuation_text = tokenizer.decode(gold_continuation_ids, skip_special_tokens=False)

                prompt_text = build_prompt_text(tokenizer, row, story_prefix_text, args.attack_mode)
                prompt_inputs = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False)
                prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}

                gen_kwargs = {
                    'max_new_tokens': args.max_new_tokens,
                    'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                }
                if args.temperature <= 0:
                    gen_kwargs['do_sample'] = False
                else:
                    gen_kwargs['do_sample'] = True
                    gen_kwargs['temperature'] = args.temperature
                    gen_kwargs['top_p'] = args.top_p

                with torch.no_grad():
                    output_ids = model.generate(**prompt_inputs, **gen_kwargs)

                raw_generated_ids = output_ids[0, prompt_inputs['input_ids'].shape[1]:].tolist()
                raw_generated_text = tokenizer.decode(raw_generated_ids, skip_special_tokens=True)
                generated_text = extract_generated_story_text(raw_generated_text)
                generated_ids = tokenizer.encode(generated_text, add_special_tokens=False)

                exact_match = exact_prefix_match_len(generated_ids, gold_continuation_ids)
                lcs_len, lcs_gen_start, lcs_gold_start = longest_common_substring_match(generated_ids, gold_continuation_ids)

                result = AuditResult(
                    attack_mode=args.attack_mode,
                    row_idx=row_idx,
                    prefix_tokens=prefix_k,
                    title=meta.get('title', row.get('title', '')),
                    author=meta.get('author', row.get('author', '')),
                    anthology=meta.get('anthology', row.get('anthology', '')),
                    prompt_chars=len(row['prompt'][0]['content']) if row.get('prompt') else 0,
                    gold_thinking_chars=len(row.get('ground_truth_reasoning', '')),
                    gold_story_tokens=len(story_ids),
                    gold_continuation_tokens=len(gold_continuation_ids),
                    generated_tokens=len(generated_ids),
                    exact_prefix_match_tokens=exact_match,
                    longest_common_substring_tokens=lcs_len,
                    lcs_start_in_generation=lcs_gen_start,
                    lcs_start_in_gold=lcs_gold_start,
                    generated_text=generated_text,
                    gold_continuation_text=gold_continuation_text,
                )
                all_results.append(result)
                completed_keys.add(key)
                f_out.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
                f_out.flush()
                print(f'[row {row_idx}] mode={args.attack_mode} prefix={prefix_k} exact_prefix={exact_match} lcs={lcs_len}', flush=True)

    final_results: List[AuditResult] = []
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                final_results.append(AuditResult(**json.loads(line)))
    summary = summarize(final_results)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
