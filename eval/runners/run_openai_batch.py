# runners/run_openai_batch.py
"""
OpenAI Batch API runner — end-to-end.

Builds requests, uploads, submits batch, polls, downloads, parses results.
Same interface as run_vertex_batch.py.

Usage:
  python3 -m runners.run_openai_batch \
    --config <config.json> \
    --output_file <output.jsonl> \
    [--poll_interval 60]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.task_registry import get_task
from core.task_base import Task, Example
from core.sampling import get_engine_cfg
from core.openai_params import build_openai_chat_body_params


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_batch_requests(
    examples: List[Example],
    task: Task,
    run_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build OpenAI batch request dicts."""
    engine_cfg = get_engine_cfg(run_cfg)
    model = engine_cfg.get("model", "")
    sampling = engine_cfg.get("sampling", {}) or {}

    requests = []
    for ex in examples:
        messages = task.build_messages(ex, run_cfg)

        # Use the same param builder as the old batch runner —
        # this handles response_format, max_completion_tokens, reasoning_effort, etc.
        params = build_openai_chat_body_params(run_cfg, task)

        body = {"model": model, "messages": messages, **params}

        requests.append({
            "custom_id": str(ex.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    return requests


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API runner — end-to-end."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--poll_interval", type=int, default=60)
    args = parser.parse_args()

    run_cfg = load_config(args.config)
    task_name = run_cfg["task"]
    task_cfg = run_cfg.get("task_config", {}) or {}
    engine_cfg = get_engine_cfg(run_cfg)
    model = engine_cfg.get("model", "")

    print(f"\n=== OpenAI Batch Runner ===")
    print(f"Model: {model}")
    print(f"Task: {task_name}")
    print(f"Sampling: {json.dumps(engine_cfg.get('sampling', {}), indent=2)}")
    print()

    # Load task + examples
    task: Task = get_task(task_name)
    examples: List[Example] = list(task.load_examples(task_cfg))
    ex_by_id = {str(ex.id): ex for ex in examples}

    # Resume support
    output_path = args.output_file
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    processed_ids: set = set()
    existing_records: List[Dict[str, Any]] = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    existing_records.append(rec)
                    rid = rec.get("id")
                    if rid is not None:
                        processed_ids.add(str(rid))
                except Exception:
                    continue

    remaining = [ex for ex in examples if str(ex.id) not in processed_ids]
    print(f"Total: {len(examples)}, done: {len(processed_ids)}, remaining: {len(remaining)}")

    if not remaining:
        print("All examples already processed.")
        all_records = existing_records
    else:
        # Build batch requests
        print(f"\nBuilding {len(remaining)} batch requests...")
        batch_reqs = build_batch_requests(remaining, task, run_cfg)

        # Write to temp JSONL
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for req in batch_reqs:
            tmp.write(json.dumps(req, ensure_ascii=False) + "\n")
        tmp.close()
        print(f"Wrote {len(batch_reqs)} requests to {tmp.name}")

        # Preview
        print(f"\n--- Request preview ---")
        preview = json.dumps(batch_reqs[0], indent=2, ensure_ascii=False)[:600]
        print(preview + "...")

        # Create client
        api_key = engine_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        # Upload file
        print(f"\nUploading batch file...")
        with open(tmp.name, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        os.unlink(tmp.name)
        print(f"Uploaded: {file_obj.id}")

        # Submit batch
        provider_params = engine_cfg.get("provider_params", {}) or {}
        completion_window = provider_params.get("completion_window", "24h")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={"description": f"cotlmeval_{task_name}_{len(remaining)}ex"},
        )
        print(f"Batch submitted: {batch.id}")
        print(f"Status: {batch.status}")

        # Poll
        print(f"\nPolling every {args.poll_interval}s...")
        start = time.time()
        while True:
            batch = client.batches.retrieve(batch.id)
            elapsed = (time.time() - start) / 60
            counts = getattr(batch.request_counts, '__dict__', {}) if batch.request_counts else {}
            total_c = counts.get('total', '?')
            completed_c = counts.get('completed', '?')
            failed_c = counts.get('failed', '?')
            print(f"  [{elapsed:.1f}min] {batch.status} — completed={completed_c}/{total_c}, failed={failed_c}", flush=True)

            if batch.status in ("completed", "expired", "failed", "cancelled"):
                break
            time.sleep(args.poll_interval)

        if batch.status != "completed":
            print(f"Batch ended with status: {batch.status}")
            if batch.error_file_id:
                err = client.files.content(batch.error_file_id)
                print(f"Errors: {err.text[:2000]}")
            return

        # Check if all requests failed
        if batch.request_counts and batch.request_counts.failed == batch.request_counts.total:
            print(f"All {batch.request_counts.total} requests failed!")
            if batch.error_file_id:
                err = client.files.content(batch.error_file_id)
                print(f"Error sample: {err.text[:2000]}")
            return

        if not batch.output_file_id:
            print("No output file ID. Checking error file...")
            if batch.error_file_id:
                err = client.files.content(batch.error_file_id)
                print(f"Errors: {err.text[:2000]}")
            return

        # Download results
        print(f"\nDownloading results...")
        out_file = client.files.content(batch.output_file_id)
        result_lines = out_file.text.strip().split("\n")
        print(f"Got {len(result_lines)} results")

        # Parse results
        batch_results = {}
        for line in result_lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            custom_id = rec.get("custom_id", "")
            response = rec.get("response", {})
            body = response.get("body", {})

            # Extract text from choices
            text = ""
            choices = body.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "")

            # Usage
            usage = body.get("usage", {})

            batch_results[custom_id] = {
                "response_text": text,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
            }

        # Safety check
        expected_ids = {str(ex.id) for ex in remaining}
        unexpected = set(batch_results.keys()) - expected_ids
        if unexpected:
            print(f"  [WARN] {len(unexpected)} unexpected custom_ids")

        # Parse through task and write
        new_records = []
        with open(output_path, "a", encoding="utf-8") as f_out:
            for ex in remaining:
                ex_id = str(ex.id)
                br = batch_results.get(ex_id)
                if br is None:
                    print(f"  [WARN] No result for id={ex_id}")
                    continue

                record = task.parse_response(ex, br["response_text"], run_cfg)
                record["_order_idx"] = len(new_records)
                record["_is_new"] = True

                usage = br.get("usage", {})
                if usage:
                    record["_provider_meta"] = {"model": model, "usage": usage}
                    pt = usage.get("prompt_tokens")
                    ct = usage.get("completion_tokens")
                    if isinstance(pt, int) and isinstance(ct, int):
                        record["openai_model"] = model
                        record["openai_prompt_tokens"] = pt
                        record["openai_completion_tokens"] = ct
                        record["openai_total_tokens"] = pt + ct

                new_records.append(record)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Wrote {len(new_records)} new records")
        all_records = existing_records + new_records

    # Summary
    print(f"\nTotal records: {len(all_records)}")
    total_pt = sum(r.get("openai_prompt_tokens", 0) or 0 for r in all_records)
    total_ct = sum(r.get("openai_completion_tokens", 0) or 0 for r in all_records)
    if total_pt or total_ct:
        print(f"\n=== Token usage ===")
        print(f"Prompt: {total_pt}, Completion: {total_ct}, Total: {total_pt + total_ct}")

    summary = task.aggregate(all_records, run_cfg, os.path.dirname(output_path))
    if summary is not None:
        print("\n=== Summary ===")
        print(task.format_summary(summary, run_cfg, max_len=100000))

    # Meta
    meta_path = output_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump({
            "num_records": len(all_records),
            "model": model,
            "mode": "openai_batch",
        }, mf, indent=2)

    print(f"\nDone. Output: {output_path}")


if __name__ == "__main__":
    main()
