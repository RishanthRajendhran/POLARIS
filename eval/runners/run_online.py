# runners/run_online.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import argparse
import asyncio
import hashlib
import json
import os
import random  # NEW

from core.task_registry import get_task
from core.task_base import Task, Example
from core.provider_registry import get_engine
from core.openai_pricing import resolve_openai_prices, estimate_openai_cost_usd
from core.vertex_pricing import resolve_vertex_prices, estimate_vertex_cost_usd


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    cfg_bytes = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(cfg_bytes).hexdigest()


# -------------------------------------------------------------------
# Core per-example runner (SYNC)
# -------------------------------------------------------------------

def _run_single_example(
    global_idx: int,
    ex: Example,
    run_cfg: Dict[str, Any],
    task: Task,
) -> Tuple[int, Dict[str, Any]]:
    """
    Synchronous worker for a single example:
      - build messages
      - call engine.generate_one
      - parse_response
      - compute per-record cost metadata
    Returns (global_idx, record).
    """
    engine = get_engine(run_cfg)  # engines are lightweight wrappers around clients

    messages = task.build_messages(ex, run_cfg)
    engine_out = engine.generate_one(messages, task, run_cfg)

    meta: Dict[str, Any] = {}
    provider_output: Any = engine_out
    if isinstance(engine_out, tuple) and len(engine_out) == 2 and isinstance(engine_out[1], dict):
        provider_output, meta = engine_out

    record = task.parse_response(ex, provider_output, run_cfg)

    if meta:
        record["_provider_meta"] = meta

    engine_cfg = run_cfg.get("engine", {}) or {}

    # ---- OpenAI costing ----
    if engine_cfg.get("provider") == "openai":
        model = meta.get("model") or engine_cfg.get("model")

        provider_params = engine_cfg.get("provider_params", {}) or {}
        provider_overrides = (engine_cfg.get("provider_overrides", {}) or {}).get("openai", {}) or {}
        service_tier = (
            provider_params.get("service_tier")
            or provider_params.get("priority")
            or meta.get("service_tier")
            or "standard"
        )

        usage = meta.get("usage") or {}
        prompt_toks = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_toks = usage.get("completion_tokens", usage.get("output_tokens"))

        if isinstance(prompt_toks, int) and isinstance(completion_toks, int):
            pricing_overrides = provider_overrides.get("pricing") or provider_params.get("pricing")
            prices, src = resolve_openai_prices(
                model=model,
                service_tier=service_tier,
                pricing_overrides=pricing_overrides,
            )

            record["openai_model"] = model
            record["openai_service_tier"] = service_tier
            record["openai_prompt_tokens"] = prompt_toks
            record["openai_completion_tokens"] = completion_toks
            record["openai_total_tokens"] = prompt_toks + completion_toks
            record["openai_pricing_source"] = src

            if prices is not None:
                cost = estimate_openai_cost_usd(prompt_toks, completion_toks, prices)
                record["openai_cost_usd"] = cost
            else:
                record["openai_cost_usd"] = None

    # ---- Vertex / Gemini costing ----
    if engine_cfg.get("provider") == "vertex":
        model = meta.get("model") or engine_cfg.get("model")
        usage = meta.get("usage") or {}
        prompt_toks = usage.get("prompt_tokens", usage.get("promptTokenCount"))
        completion_toks = usage.get("completion_tokens", usage.get("candidatesTokenCount"))

        if isinstance(prompt_toks, int) and isinstance(completion_toks, int):
            provider_overrides = (engine_cfg.get("provider_overrides", {}) or {}).get("vertex", {}) or {}
            provider_params = engine_cfg.get("provider_params", {}) or {}
            pricing_overrides = provider_overrides.get("pricing") or provider_params.get("pricing")

            vertex_mode = provider_params.get("vertex_mode") or provider_params.get("mode") or "online"
            vertex_consumption = (provider_params.get("vertex_consumption") or "standard").strip().lower()

            prices, src = resolve_vertex_prices(
                model=model,
                mode=vertex_mode,
                consumption=vertex_consumption,
                pricing_overrides=pricing_overrides,
            )

            record["vertex_model"] = model
            record["vertex_mode"] = vertex_mode
            record["vertex_consumption"] = vertex_consumption
            record["vertex_prompt_tokens"] = prompt_toks
            record["vertex_completion_tokens"] = completion_toks
            record["vertex_total_tokens"] = (prompt_toks or 0) + (completion_toks or 0)
            record["vertex_pricing_source"] = src

            if prices is not None:
                cost = estimate_vertex_cost_usd(
                    prompt_tokens=prompt_toks,
                    completion_tokens=completion_toks,
                    prices=prices,
                )
                record["vertex_cost_usd"] = cost
            else:
                record["vertex_cost_usd"] = None

    record["_order_idx"] = global_idx
    record["_is_new"] = True
    return global_idx, record


# -------------------------------------------------------------------
# Async orchestration
# -------------------------------------------------------------------
async def run_examples_async(
    run_cfg: Dict[str, Any],
    task: Task,
    examples: List[Example],
    processed_ids: set,
    output_path: str,
    print_examples: int,
    *,
    sample_ids_for_joint_preview: set[str],
    ex_by_id: Dict[str, Example],
) -> List[Dict[str, Any]]:
    """
    Run remaining examples concurrently using asyncio.to_thread around _run_single_example.
    Checkpoints records to `output_path` as they complete.
    Returns only the NEW records (old ones already loaded).

    Additionally:
      - For any example whose id is in sample_ids_for_joint_preview, print
        both an INPUT and RECORD preview as soon as its record is available.
    """
    remaining: List[Tuple[int, Example]] = [
        (idx, ex)
        for idx, ex in enumerate(examples)
        if str(ex.id) not in processed_ids
    ]

    if not remaining:
        return []

    total = len(remaining)
    concurrency = int(run_cfg.get("runner_concurrency", 1))
    concurrency = max(1, concurrency)
    sem = asyncio.Semaphore(concurrency)

    async def _worker(global_idx: int, ex: Example) -> Dict[str, Any]:
        async with sem:
            _, rec = await asyncio.to_thread(
                _run_single_example,
                global_idx,
                ex,
                run_cfg,
                task,
            )
            # Checkpoint immediately
            with open(output_path, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(rec, ensure_ascii=False))
                f_out.write("\n")
                f_out.flush()

            # If this example is in the random sample, print joint input+record preview
            ex_id_str = str(ex.id)
            if ex_id_str in sample_ids_for_joint_preview:
                print(f"\n--- Sample joint preview for id={ex_id_str} ---", flush=True)

                # INPUT PREVIEW
                msgs = task.build_messages(ex, run_cfg)
                preview_input = getattr(task, "format_input_preview", None)
                if callable(preview_input):
                    print("\n### INPUT PREVIEW ###", flush=True)
                    print(preview_input(ex, msgs, run_cfg, max_len=100000), flush=True)
                else:
                    print("\n### INPUT PREVIEW (fallback) ###", flush=True)
                    print(json.dumps(ex.data, ensure_ascii=False)[:100000], flush=True)

                # RECORD PREVIEW
                print("\n### RECORD PREVIEW ###", flush=True)
                print(task.format_record_preview(rec, run_cfg, max_len=100000), flush=True)

            return rec

    print(f"Running {total} new examples with concurrency={concurrency}")

    tasks = [asyncio.create_task(_worker(idx, ex)) for idx, ex in remaining]
    new_records: List[Dict[str, Any]] = []

    done = 0

    # As each example finishes, we:
    #   - update progress
    #   - optionally print a record-only preview for the first `print_examples` new examples
    for t in asyncio.as_completed(tasks):
        rec = await t
        new_records.append(rec)
        done += 1

        # Lightweight progress line
        print(f"[progress] completed {done}/{total} examples", flush=True)

    return new_records

# -------------------------------------------------------------------
# Main runner
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generic synchronous/async runner (HF, vLLM, OpenAI, OpenRouter, Vertex/Gemini)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    parser.add_argument("--output_file", type=str, required=True, help="Where to write JSONL records.")
    parser.add_argument(
        "--service_tier",
        type=str,
        default=None,
        choices=["auto", "default", "flex", "priority", "scale"],
        help=(
            "Optional OpenAI service_tier override. "
            "If set and engine.provider == 'openai', this will be passed via provider_params.service_tier."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Maximum number of examples to process concurrently. "
            "If >1, examples are run via asyncio.to_thread; prints are deferred to end."
        ),
    )
    args = parser.parse_args()

    run_cfg = load_config(args.config)

    # CLI override for OpenAI service_tier
    if args.service_tier is not None:
        engine_cfg_cli = run_cfg.get("engine", {}) or {}
        provider = engine_cfg_cli.get("provider")
        if provider == "openai":
            provider_params = engine_cfg_cli.get("provider_params") or {}
            provider_params.setdefault("service_tier", args.service_tier)
            engine_cfg_cli["provider_params"] = provider_params
            run_cfg["engine"] = engine_cfg_cli

    # Also expose concurrency to run_cfg so tasks can see it if needed
    run_cfg["runner_concurrency"] = max(1, args.concurrency)

    # Config hash for checkpointing
    config_hash = compute_config_hash(run_cfg)

    task_name = run_cfg["task"]
    task_cfg = run_cfg.get("task_config", {}) or {}
    task_params = run_cfg.get("task_params", {}) or {}
    engine_cfg = run_cfg.get("engine", {}) or {}

    print("\n=== Engine / model config ===")
    print(json.dumps(engine_cfg, indent=2, sort_keys=True))
    print("\n=== Task / data config ===")
    print(json.dumps(task_cfg, indent=2, sort_keys=True))
    print(json.dumps(task_params, indent=2, sort_keys=True))
    print("========================================\n")

    # Load task and examples
    task: Task = get_task(task_name)
    examples: List[Example] = list(task.load_examples(task_cfg))
    # Map id -> Example for later joint input+record previews
    ex_by_id: Dict[str, Example] = {str(ex.id): ex for ex in examples}

    # Choose up to 100 random example IDs for streaming joint (input+record) previews.
    # Use all example ids here; you could restrict to remaining_to_run later if you prefer.
    all_ids_for_sampling = [str(ex.id) for ex in examples]
    sample_size = min(100, len(all_ids_for_sampling))
    sample_ids_for_joint_preview: set[str] = set()
    if sample_size > 0:
        sample_ids_for_joint_preview = set(random.sample(all_ids_for_sampling, sample_size))

    output_path = args.output_file
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    meta_path = output_path + ".meta.json"

    records: List[Dict[str, Any]] = []
    processed_ids: set = set()
    resuming = False

    # Resume logic
    if os.path.exists(output_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta_json = json.load(mf)
        except Exception:
            meta_json = {}

        if meta_json.get("config_hash") == config_hash:
            resuming = True
            print(f"Resuming from existing output_file={output_path} (config_hash match).")
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    records.append(rec)
                    rid = rec.get("id")
                    if rid is not None:
                        processed_ids.add(str(rid))
        else:
            print(
                f"Existing output/meta found but config_hash mismatch.\n"
                f"  existing hash: {meta_json.get('config_hash')}\n"
                f"  current hash:  {config_hash}\n"
                f"Backing up old outputs and starting a fresh run."
            )
            os.rename(output_path, output_path + ".bak")
            os.rename(meta_path, meta_path + ".bak")

    elif os.path.exists(output_path) and not os.path.exists(meta_path):
        print(
            f"Existing output_file found at {output_path} but no meta file.\n"
            "Backing it up and starting a fresh run."
        )
        os.rename(output_path, output_path + ".bak")

    # If we're not resuming, ensure the output file is empty/created
    if not resuming:
        with open(output_path, "w", encoding="utf-8"):
            pass  # just truncate/create
        # Write meta file immediately so chained jobs can resume if this one
        # is preempted or times out before completing all examples.
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump({"config_hash": config_hash, "num_records": 0, "config": run_cfg},
                      mf, indent=2, sort_keys=True)

    print(f"Total examples: {len(examples)}; already processed: {len(processed_ids)}")

    # Optional: show input previews for the first few unprocessed examples (unchanged)
    max_input_previews = min(5, len(examples))
    if max_input_previews > 0:
        print(f"\n=== Input previews (first {max_input_previews} examples) ===")
        for ex in examples[:max_input_previews]:
            # Build messages so Tasks that depend on them (e.g. for truncation) can show them
            msgs = task.build_messages(ex, run_cfg)
            preview = getattr(task, "format_input_preview", None)
            if callable(preview):
                print("\n" + preview(ex, msgs, run_cfg, max_len=10000000))
            else:
                # Fallback: minimal manual preview
                print(f"\nExample id: {ex.id}")
                print(f"Data: {json.dumps(ex.data, ensure_ascii=False)[:1000000]}")

    # Run remaining examples (async, with concurrency)
    remaining_to_run = [ex for ex in examples if str(ex.id) not in processed_ids]
    print_examples = min(100, len(remaining_to_run))  # still used for inline previews during run

    if remaining_to_run:
        asyncio.run(
            run_examples_async(
                run_cfg=run_cfg,
                task=task,
                examples=examples,
                processed_ids=processed_ids,
                output_path=output_path,
                print_examples=print_examples,
                sample_ids_for_joint_preview=sample_ids_for_joint_preview,
                ex_by_id=ex_by_id,
            )
        )

        # Reload all records from output file to have a single consistent view
        records = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                records.append(rec)
    else:
        print("All examples already processed; skipping inference.")

    # Update meta file
    meta_payload = {
        "config_hash": config_hash,
        "num_records": len(records),
        "config": run_cfg,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta_payload, mf, indent=2, sort_keys=True)

    # ----------------- Token usage / cost summary -----------------
    engine_cfg = run_cfg.get("engine", {}) or {}
    provider = engine_cfg.get("provider")
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    num_costed = 0
    pricing_source = None

    if provider == "openai":
        for r in records:
            pt = r.get("openai_prompt_tokens")
            ct = r.get("openai_completion_tokens")
            c = r.get("openai_cost_usd")
            if isinstance(pt, int) and isinstance(ct, int):
                total_prompt_tokens += pt
                total_completion_tokens += ct
                if isinstance(c, (int, float)):
                    total_cost_usd += float(c)
                    num_costed += 1
                pricing_source = pricing_source or r.get("openai_pricing_source")

        print("\n=== OpenAI token usage / cost summary ===")
        print(f"Costed requests: {num_costed}/{len(records)}")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Total tokens: {total_prompt_tokens + total_completion_tokens}")
        print(f"Estimated cost (USD): {total_cost_usd:.6f}")
        if pricing_source:
            print(f"Pricing source: {pricing_source}")
        print("========================================\n")

    if provider == "vertex":
        for r in records:
            pt = r.get("vertex_prompt_tokens")
            ct = r.get("vertex_completion_tokens")
            c = r.get("vertex_cost_usd")
            if isinstance(pt, int) and isinstance(ct, int):
                total_prompt_tokens += pt
                total_completion_tokens += ct
                if isinstance(c, (int, float)):
                    total_cost_usd += float(c)
                    num_costed += 1
                pricing_source = pricing_source or r.get("vertex_pricing_source")

        print("\n=== Vertex AI token usage / cost summary ===")
        print(f"Costed requests: {num_costed}/{len(records)}")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Total tokens: {total_prompt_tokens + total_completion_tokens}")
        print(f"Estimated cost (USD): {total_cost_usd:.6f}")
        print("Note: pricing based on core/vertex_pricing.py (Gemini text-token rates).")
        print("========================================\n")

    # ----------------- Aggregate -----------------
    summary = task.aggregate(records, run_cfg, os.path.dirname(output_path))
    if summary is not None:
        print("\nSummary:")
        print(task.format_summary(summary, run_cfg, max_len=100000))


if __name__ == "__main__":
    main()