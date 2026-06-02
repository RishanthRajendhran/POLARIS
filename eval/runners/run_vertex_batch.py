# runners/run_vertex_batch.py
"""
Vertex AI Batch Prediction runner for Gemini models.

Uploads requests as JSONL to GCS, submits a batch job, polls for
completion, downloads and parses results into the same output format
as run_online.py.

Usage:
  python3 -m runners.run_vertex_batch \
    --config <config.json> \
    --output_file <output.jsonl> \
    [--poll_interval 60] \
    [--gcs_prefix gs://bucket/prefix]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types

from core.task_registry import get_task
from core.task_base import Task, Example
from core.sampling import get_engine_cfg
from providers.vertex_engine import _build_http_options_with_consumption


DEFAULT_GCS_BUCKET = "gs://creative-eval-batch-20260317-0966014990"
DEFAULT_GCS_PREFIX = "cotlmeval_batch"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _messages_to_gemini_batch(messages: List[Dict[str, str]]):
    """Convert OpenAI-style messages to Vertex batch format dicts."""
    sys_parts = []
    contents = []

    for m in messages:
        role = (m.get("role") or "user").strip()
        text = str(m.get("content", "") or "")
        if role == "system":
            sys_parts.append({"text": text})
        elif role == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": text}],
            })

    system_instruction = {"parts": sys_parts} if sys_parts else None
    return system_instruction, contents


def _build_generation_config(engine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build generationConfig dict for the batch request."""
    sampling = engine_cfg.get("sampling", {}) or {}
    overrides = (engine_cfg.get("provider_overrides", {}) or {}).get("vertex", {}) or {}
    merged = {**sampling, **overrides}

    cfg = {}
    if "temperature" in merged:
        cfg["temperature"] = float(merged["temperature"])
    if "top_p" in merged:
        cfg["topP"] = float(merged["top_p"])
    if "top_k" in merged:
        cfg["topK"] = int(merged["top_k"])
    if "max_output_tokens" in merged:
        cfg["maxOutputTokens"] = int(merged["max_output_tokens"])
    elif "max_tokens" in merged:
        cfg["maxOutputTokens"] = int(merged["max_tokens"])
    if "stop_sequences" in merged:
        cfg["stopSequences"] = list(merged["stop_sequences"])
    if "seed" in merged:
        cfg["seed"] = int(merged["seed"])
    if "presence_penalty" in merged:
        cfg["presencePenalty"] = float(merged["presence_penalty"])
    if "frequency_penalty" in merged:
        cfg["frequencyPenalty"] = float(merged["frequency_penalty"])
    if "thinking_config" in merged:
        tc = merged["thinking_config"]
        batch_tc = {}
        # Gemini 3+: thinkingLevel (LOW/MEDIUM/HIGH)
        if "thinkingLevel" in tc:
            batch_tc["thinkingLevel"] = tc["thinkingLevel"]
        # Gemini 2.5: thinkingBudget (token count)
        if "thinkingBudget" in tc:
            batch_tc["thinkingBudget"] = tc["thinkingBudget"]
        if "includeThoughts" in tc:
            batch_tc["includeThoughts"] = tc["includeThoughts"]
        cfg["thinkingConfig"] = batch_tc
    if "response_mime_type" in merged:
        cfg["responseMimeType"] = merged["response_mime_type"]
    if "response_schema" in merged:
        cfg["responseSchema"] = merged["response_schema"]

    return cfg


def build_batch_jsonl(
    examples: List[Example],
    task: Task,
    run_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build list of batch request dicts (one per line in JSONL)."""
    engine_cfg = get_engine_cfg(run_cfg)
    gen_config = _build_generation_config(engine_cfg)

    # Check if task provides structured output
    rf = task.get_response_format("vertex", run_cfg)
    if isinstance(rf, dict):
        mime = rf.get("response_mime_type") or rf.get("mime_type")
        if mime and "responseMimeType" not in gen_config:
            gen_config["responseMimeType"] = mime
        schema = rf.get("response_schema") or rf.get("schema")
        if schema and "responseSchema" not in gen_config:
            gen_config["responseSchema"] = schema

    requests = []
    for ex in examples:
        messages = task.build_messages(ex, run_cfg)
        system_instruction, contents = _messages_to_gemini_batch(messages)

        # For tasks with per-example schemas (e.g. WritingBench), we can't
        # pass different responseSchema per request in a single batch.
        # Instead, ensure responseMimeType=application/json is set so the
        # judge at least returns JSON, then rely on fuzzy parsing.
        ex_schema = (ex.data or {}).get("_schema")
        if ex_schema and "responseMimeType" not in gen_config:
            gen_config = dict(gen_config)
            gen_config["responseMimeType"] = "application/json"

        request_body = {
            "contents": contents,
            "generationConfig": gen_config,
        }
        if system_instruction:
            request_body["systemInstruction"] = system_instruction

        requests.append({
            "custom_id": str(ex.id),
            "request": request_body,
        })

    return requests


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a local file to GCS using gcloud."""
    print(f"  Uploading {local_path} -> {gcs_path}")
    subprocess.run(
        ["gcloud", "storage", "cp", local_path, gcs_path],
        check=True, capture_output=True, text=True,
    )


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download a file/dir from GCS."""
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", gcs_path, local_path],
        check=True, capture_output=True, text=True,
    )


class _RestBatchJobAdapter:
    """Holds the v1beta1 BatchPredictionJob resource JSON and exposes the
    .name/.state/.dest/.batch_stats interface that the rest of this runner
    expects (mirroring genai.BatchJob shape) so downstream code is unchanged."""

    class _Dest:
        def __init__(self, uri: str):
            self.gcs_uri = uri

    class _Stats:
        def __init__(self, success: int, fail: int):
            self.success_count = success
            self.fail_count = fail
            self.total_count = success + fail

    def __init__(self, resource: Dict[str, Any], creds, host: str):
        self._resource = resource
        self._creds = creds
        self._host = host  # e.g. "aiplatform.googleapis.com" or "us-central1-aiplatform.googleapis.com"

    def _ensure_token(self):
        from google.auth.transport.requests import Request as AuthRequest
        if not self._creds.valid:
            self._creds.refresh(AuthRequest())
        return self._creds.token

    def refresh(self):
        import requests
        token = self._ensure_token()
        url = f"https://{self._host}/v1beta1/{self.name}"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        r.raise_for_status()
        self._resource = r.json()

    @property
    def name(self) -> str:
        return self._resource.get("name", "")

    @property
    def state(self) -> str:
        return self._resource.get("state", "JOB_STATE_UNSPECIFIED")

    @property
    def dest(self) -> Optional["_RestBatchJobAdapter._Dest"]:
        oc = self._resource.get("outputInfo", {}) or {}
        uri = oc.get("gcsOutputDirectory")
        return _RestBatchJobAdapter._Dest(uri) if uri else None

    @property
    def batch_stats(self) -> Optional["_RestBatchJobAdapter._Stats"]:
        cs = self._resource.get("completionStats") or {}
        if not cs:
            return None
        return _RestBatchJobAdapter._Stats(
            success=int(cs.get("successfulCount", 0) or 0),
            fail=int(cs.get("failedCount", 0) or 0),
        )

    @property
    def labels(self) -> Dict[str, str]:
        return self._resource.get("labels") or {}


def submit_batch_job(
    client: Optional[genai.Client],
    model: str,
    gcs_input_uri: str,
    gcs_output_uri: str,
    display_name: str,
    user_label: str = "polaris",
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> "_RestBatchJobAdapter":
    """Submit a Vertex batch prediction job via direct REST POST to the
    v1beta1 endpoint. We do this (instead of google-genai's batches.create)
    so we can attach a billing label (`labels.user=<user>`) — google-genai's
    CreateBatchJobConfig does not expose `labels`. The v1beta1 endpoint also
    supports preview Gemini models that the legacy aiplatform v1 SDK rejects
    (`gemini-3.1-pro-preview` is not visible to v1).

    Online calls continue to use google-genai (with labels via
    GenerateContentConfig).
    """
    import requests
    from google.auth import default as google_auth_default
    from google.auth.transport.requests import Request as AuthRequest

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    # Default to "global" — matches genai's batch routing where preview Gemini
    # models are visible. Regional (e.g. us-central1) is also supported.
    location = location or os.environ.get("AIPLATFORM_BATCH_LOCATION") or "global"

    if user_label and not display_name.startswith(f"{user_label}_"):
        display_name = f"{user_label}_{display_name}"

    model_name = model if "/" in model else f"publishers/google/models/{model}"

    body: Dict[str, Any] = {
        "displayName": display_name,
        "model": model_name,
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [gcs_input_uri]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": gcs_output_uri.rstrip("/")},
        },
    }
    if user_label:
        body["labels"] = {"user": user_label}

    creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(AuthRequest())

    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    url = f"https://{host}/v1beta1/projects/{project}/locations/{location}/batchPredictionJobs"
    r = requests.post(
        url,
        json=body,
        headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"batchPredictionJobs create failed [{r.status_code}]: {r.text[:1500]}")
    resource = r.json()
    return _RestBatchJobAdapter(resource, creds, host)


def poll_batch_job(
    client: Optional[genai.Client],
    job_name: str,
    poll_interval: int = 60,
    job_handle: Optional["_RestBatchJobAdapter"] = None,
) -> "_RestBatchJobAdapter":
    """Poll the v1beta1 batch job until completion. `job_handle` is the
    adapter returned by submit_batch_job; `client` and `job_name` are kept
    for signature compatibility but unused."""
    if job_handle is None:
        raise RuntimeError(
            "poll_batch_job now requires the REST job handle returned by submit_batch_job"
        )

    start = time.time()
    while True:
        try:
            job_handle.refresh()
        except Exception as e:
            print(f"  [poll] refresh error: {e!r}; retrying in {poll_interval}s", flush=True)
            time.sleep(poll_interval)
            continue
        state = job_handle.state
        elapsed = (time.time() - start) / 60

        stats = job_handle.batch_stats
        if stats is not None:
            print(
                f"  [{elapsed:.1f}min] {state} — success={stats.success_count}/{stats.total_count}, "
                f"failed={stats.fail_count}",
                flush=True,
            )
        else:
            print(f"  [{elapsed:.1f}min] {state}", flush=True)

        if "SUCCEEDED" in str(state):
            return job_handle
        if "FAILED" in str(state) or "CANCELLED" in str(state) or "EXPIRED" in str(state):
            raise RuntimeError(f"Batch job {job_handle.name} ended with state: {state}")

        time.sleep(poll_interval)


def parse_batch_output(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse batch output JSONL files downloaded from GCS.
    Returns dict[custom_id] -> {"response_text": str, "usage": dict}.
    """
    results = {}
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    custom_id = rec.get("custom_id", "")
                    response = rec.get("response", {})

                    # Extract text from response candidates
                    text = ""
                    candidates = response.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        text_parts = []
                        for p in parts:
                            if "text" in p and not p.get("thought", False):
                                text_parts.append(p["text"])
                        text = "".join(text_parts)

                    # Extract usage
                    usage_meta = response.get("usageMetadata", {})
                    usage = {
                        "prompt_tokens": usage_meta.get("promptTokenCount"),
                        "completion_tokens": usage_meta.get("candidatesTokenCount"),
                        "total_tokens": usage_meta.get("totalTokenCount"),
                        "thoughts_tokens": usage_meta.get("thoughtsTokenCount"),
                    }

                    results[custom_id] = {
                        "response_text": text,
                        "usage": usage,
                    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Vertex AI Batch Prediction runner for Gemini models."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--poll_interval", type=int, default=60)
    parser.add_argument("--gcs_prefix", type=str, default=None,
                        help="GCS prefix for input/output (default: auto)")
    parser.add_argument("--display_name", type=str, default=None)
    args = parser.parse_args()

    run_cfg = load_config(args.config)
    task_name = run_cfg["task"]
    task_cfg = run_cfg.get("task_config", {}) or {}
    engine_cfg = get_engine_cfg(run_cfg)
    model = engine_cfg.get("model", "")

    print(f"\n=== Vertex Batch Runner ===")
    print(f"Model: {model}")
    print(f"Task: {task_name}")
    print(f"Sampling: {json.dumps(engine_cfg.get('sampling', {}), indent=2)}")
    overrides = (engine_cfg.get("provider_overrides", {}) or {}).get("vertex", {})
    if overrides:
        print(f"Overrides: {json.dumps(overrides, indent=2)}")
    print()

    # Load task and examples
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
        # Build batch JSONL
        print(f"\nBuilding {len(remaining)} batch requests...")
        batch_lines = build_batch_jsonl(remaining, task, run_cfg)

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for line in batch_lines:
            tmp.write(json.dumps(line, ensure_ascii=False) + "\n")
        tmp.close()
        print(f"Wrote {len(batch_lines)} requests to {tmp.name}")

        # Preview
        print(f"\n--- Request preview (first) ---")
        preview = json.dumps(batch_lines[0], indent=2, ensure_ascii=False)[:800]
        print(preview + "...")

        # GCS paths — use UUID to guarantee uniqueness across concurrent jobs
        import uuid
        uid = uuid.uuid4().hex[:12]
        ts = int(time.time())
        job_tag = args.display_name or f"{task_name}_{len(remaining)}ex"
        gcs_prefix = args.gcs_prefix or f"{DEFAULT_GCS_BUCKET}/{DEFAULT_GCS_PREFIX}/{job_tag}_{ts}_{uid}"
        gcs_input = f"{gcs_prefix}/input/requests.jsonl"
        gcs_output = f"{gcs_prefix}/output/"

        print(f"\nGCS input:  {gcs_input}")
        print(f"GCS output: {gcs_output}")

        # Upload
        upload_to_gcs(tmp.name, gcs_input)
        os.unlink(tmp.name)

        # Create client
        http_options = _build_http_options_with_consumption(engine_cfg)
        api_key = (
            engine_cfg.get("api_key")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        client = genai.Client(api_key=api_key, http_options=http_options) if api_key else genai.Client(http_options=http_options)

        # Submit
        display_name = job_tag
        # User label for billing tracking (default: polaris)
        sampling = engine_cfg.get("sampling", {}) or {}
        overrides = (engine_cfg.get("provider_overrides", {}) or {}).get("vertex", {}) or {}
        merged = {**sampling, **overrides}
        user_label = str(merged.get("vertexai_user", "polaris"))
        print(f"\nSubmitting batch job: {display_name} (user={user_label})")
        job = submit_batch_job(client, model, gcs_input, gcs_output, display_name, user_label=user_label)
        job_name = getattr(job, "name", "")
        print(f"Job name: {job_name}")
        print(f"State: {getattr(job, 'state', 'UNKNOWN')}")

        # Poll
        print(f"\nPolling every {args.poll_interval}s...")
        completed_job = poll_batch_job(client, job_name, args.poll_interval, job_handle=job)
        print(f"\nBatch job completed!")

        # Download results — use the actual output URI from the completed job,
        # not our guessed gcs_output, to avoid mixing results from different jobs
        actual_dest = None
        try:
            job_dest = getattr(completed_job, "dest", None)
            if job_dest:
                actual_dest = getattr(job_dest, "gcs_uri", None)
                if isinstance(actual_dest, list):
                    actual_dest = actual_dest[0] if actual_dest else None
        except Exception:
            pass
        download_uri = actual_dest or gcs_output
        local_output_dir = tempfile.mkdtemp(prefix="vertex_batch_output_")
        print(f"Downloading results from {download_uri} -> {local_output_dir}")
        download_from_gcs(download_uri, local_output_dir)

        # Parse results
        batch_results = parse_batch_output(local_output_dir)
        print(f"Parsed {len(batch_results)} results")

        # Safety check: verify custom_ids match our examples
        expected_ids = {str(ex.id) for ex in remaining}
        got_ids = set(batch_results.keys())
        unexpected = got_ids - expected_ids
        if unexpected:
            print(f"  [WARN] {len(unexpected)} unexpected custom_ids in results (possible cross-job contamination):")
            for uid in list(unexpected)[:5]:
                print(f"    {uid}")
            print(f"  Only using results that match our example IDs.")

        # Match with examples and run task.parse_response
        new_records = []
        with open(output_path, "a", encoding="utf-8") as f_out:
            for i, ex in enumerate(remaining):
                ex_id = str(ex.id)
                br = batch_results.get(ex_id)

                if br is None:
                    print(f"  [WARN] No result for id={ex_id}")
                    continue

                response_text = br.get("response_text", "")
                record = task.parse_response(ex, response_text, run_cfg)
                record["_order_idx"] = i
                record["_is_new"] = True

                usage = br.get("usage", {})
                if usage:
                    record["_provider_meta"] = {"model": model, "usage": usage}
                    pt = usage.get("prompt_tokens")
                    ct = usage.get("completion_tokens")
                    if isinstance(pt, int) and isinstance(ct, int):
                        record["vertex_model"] = model
                        record["vertex_mode"] = "batch"
                        record["vertex_prompt_tokens"] = pt
                        record["vertex_completion_tokens"] = ct
                        record["vertex_total_tokens"] = pt + ct

                new_records.append(record)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Wrote {len(new_records)} new records")
        all_records = existing_records + new_records

        # Clean up temp dir
        import shutil
        shutil.rmtree(local_output_dir, ignore_errors=True)

    # Summary
    print(f"\nTotal records: {len(all_records)}")

    total_prompt = sum(r.get("vertex_prompt_tokens", 0) or 0 for r in all_records)
    total_comp = sum(r.get("vertex_completion_tokens", 0) or 0 for r in all_records)
    if total_prompt or total_comp:
        print(f"\n=== Token usage ===")
        print(f"Prompt: {total_prompt}, Completion: {total_comp}, Total: {total_prompt + total_comp}")
        print(f"(Batch = 50% of online pricing)")

    summary = task.aggregate(all_records, run_cfg, os.path.dirname(output_path))
    if summary is not None:
        print("\n=== Summary ===")
        print(task.format_summary(summary, run_cfg, max_len=100000))

    # Meta
    meta_path = output_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump({
            "num_records": len(all_records),
            "batch_job_name": job_name if "job_name" in dir() else None,
            "model": model,
            "mode": "batch",
        }, mf, indent=2)

    print(f"\nDone. Output: {output_path}")


if __name__ == "__main__":
    main()
