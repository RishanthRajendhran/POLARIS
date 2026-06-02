# reward/components/story_quality.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import asyncio
import json
import os

import regex as re
import requests
import yaml

try:
    import aiohttp
    _HAS_AIOHTTP = True
except Exception:
    _HAS_AIOHTTP = False

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai.types import HttpOptions
    _HAS_GENAI = True
except Exception:
    genai = None
    genai_types = None
    HttpOptions = None
    _HAS_GENAI = False

from .base import RewardComponent
from .llm_judge_utils import (
    _extract_first_json_object,
    _strip_code_fences,
    sanitize_json_schema_for_genai,
    supports_native_schema,
)

WORDCOUNT_REGEX = re.compile(r"(?P<word_count>\d+)\s+words\s+long", flags=re.IGNORECASE)
LENGTH_SENTENCE_REGEX = re.compile(r"words\s+long\b", flags=re.IGNORECASE)


def _strip_length_requirement_sentence(prompt: str) -> str:
    if not prompt:
        return prompt
    matches = list(LENGTH_SENTENCE_REGEX.finditer(prompt))
    if not matches:
        return prompt

    last = matches[-1]
    idx = last.start()

    boundary_chars = ".!?\n"
    start = None
    for i in range(idx - 1, -1, -1):
        if prompt[i] in boundary_chars:
            start = i + 1
            break

    if start is None:
        return prompt
    return prompt[:start].rstrip()


class StoryQualityComponent(RewardComponent):
    """Sequence-level Story Quality judge for the public POLARIS release.

    This public copy intentionally supports only the main paper path:
    sequence-level judged story quality with optional overall-score override and
    disqualification gates. Paragraph/sentence locator mode and token-level
    shaping helpers are intentionally omitted from the release version.
    """

    name = "story_quality"

    def __init__(self):
        self._api_key: Optional[str] = None
        self._endpoint_url = "https://api.openai.com/v1/chat/completions"
        self._provider: str = "openai"
        self._vertex_client = None
        self._vertex_http_options = None

    def enabled(self, cfg: Any) -> bool:
        return bool(getattr(cfg.algorithm, "reward_components", {}).get("enable_story_quality", False))

    def needs_gpu(self) -> bool:
        return False

    def keys(self) -> List[str]:
        return ["story_quality"]

    def _ensure_api_key(self) -> str:
        if self._api_key is None:
            key = os.getenv("OPENAI_API_KEY", None)
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set for StoryQualityComponent.")
            self._api_key = key
        return self._api_key

    def _make_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._ensure_api_key()}",
            "Content-Type": "application/json",
        }

    def _is_gpt5(self, model: str) -> bool:
        return isinstance(model, str) and ("gpt-5" in model.lower())

    def _make_body(
        self,
        system_msg: str,
        user_msg: str,
        schema_obj: Dict[str, Any],
        cfg: Any,
        use_native_schema: bool,
    ) -> Dict[str, Any]:
        alg = cfg.algorithm
        model = str(getattr(alg, "story_quality_model_name", "gpt-5-mini"))

        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }

        if use_native_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": schema_obj}

        service_tier = getattr(alg, "story_quality_service_tier", "flex")
        if isinstance(service_tier, str) and service_tier.strip():
            body["service_tier"] = service_tier.strip()

        max_tokens = int(getattr(alg, "story_quality_max_output_tokens", 4096))

        if self._is_gpt5(model):
            reasoning = str(getattr(alg, "story_quality_reasoning", "medium")).lower()
            verbosity = str(getattr(alg, "story_quality_verbosity", "medium")).lower()
            body["reasoning_effort"] = reasoning
            body["verbosity"] = verbosity
            body["max_completion_tokens"] = max_tokens
        else:
            temperature = float(getattr(alg, "story_quality_temperature", 0.0) or 0.0)
            top_p = float(getattr(alg, "story_quality_top_p", 1.0) or 1.0)
            body["temperature"] = temperature
            body["top_p"] = top_p
            body["max_tokens"] = max_tokens

        return body

    def _call_single_sync(
        self, url: str, headers: Dict[str, str], body: Dict[str, Any], timeout_s: float
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout_s)
            try:
                payload = r.json()
            except Exception:
                payload = None
            if r.status_code != 200 or not isinstance(payload, dict):
                return None, False, f"http_status_{r.status_code}"
            text = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
            return text, False, None
        except requests.exceptions.Timeout:
            return None, True, "timeout"
        except Exception as e:
            return None, False, f"exception:{e!r}"

    async def _call_single_async(
        self,
        session: "aiohttp.ClientSession",
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        timeout_s: float,
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with session.post(url, headers=headers, json=body, timeout=timeout) as resp:
                status = resp.status
                payload = None
                try:
                    payload = await resp.json()
                except Exception:
                    pass
                if status != 200 or not isinstance(payload, dict):
                    return None, False, f"http_status_{status}"
                text = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
                return text, False, None
        except asyncio.TimeoutError:
            return None, True, "timeout"
        except Exception as e:
            return None, False, f"exception:{e!r}"

    def _get_provider(self, cfg: Any) -> str:
        p = str(getattr(cfg.algorithm, "story_quality_provider", "openai")).strip().lower()
        return p if p in ("openai", "vertex") else "openai"

    def _get_force_structured_mode(self, cfg: Any) -> str:
        m = str(getattr(cfg.algorithm, "story_quality_force_structured_mode", "")).strip().lower()
        return m if m in ("native", "prompt") else ""

    def _maybe_embed_schema_in_system(self, base_system: str, schema_wrapper: Dict[str, Any]) -> str:
        inner = (schema_wrapper or {}).get("schema") or {}
        if not inner:
            return base_system
        schema_json = json.dumps(inner, indent=2, ensure_ascii=False)
        return (
            base_system
            + "\n\n"
            + "==================================================\n"
            + "FORMAL JSON SCHEMA FOR OUTPUT (FOR MODELS WITHOUT NATIVE SCHEMA SUPPORT)\n"
            + "==================================================\n"
            + "The model MUST return a single JSON object that conforms to this schema:\n"
            + schema_json
            + "\n"
            + "Return ONLY that JSON object and nothing else.\n"
        )

    def _ensure_vertex_client(self, cfg: Any):
        if hasattr(self, "_vertex_client") and self._vertex_client is not None:
            return self._vertex_client
        if not _HAS_GENAI:
            raise RuntimeError("google-genai not installed. Install: pip install google-genai")

        alg = cfg.algorithm
        service_tier = str(getattr(alg, "story_quality_service_tier", "standard")).strip().lower()

        headers: Dict[str, str] = {}
        if service_tier in ("flex", "priority"):
            headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
            headers["X-Vertex-AI-LLM-Shared-Request-Type"] = service_tier

        http_options = HttpOptions(api_version="v1", headers=headers or None)

        project = getattr(alg, "story_quality_vertex_project", None) or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = getattr(alg, "story_quality_vertex_location", None) or os.getenv("GOOGLE_CLOUD_LOCATION")

        if project and location:
            self._vertex_client = genai.Client(
                vertexai=True,
                project=str(project),
                location=str(location),
                http_options=http_options,
            )
        else:
            self._vertex_client = genai.Client(http_options=http_options)
        return self._vertex_client

    def _vertex_extract_text(self, resp: Any) -> Optional[str]:
        try:
            t = getattr(resp, "text", None)
            if isinstance(t, str) and t.strip():
                return t
        except Exception:
            pass
        try:
            parts = getattr(resp, "parts", None) or []
            chunks = []
            for p in parts:
                tx = getattr(p, "text", None)
                if isinstance(tx, str):
                    chunks.append(tx)
            out = "".join(chunks).strip()
            return out if out else None
        except Exception:
            return None

    async def _call_vertex_single_async_threaded(
        self,
        client: Any,
        model: str,
        contents: str,
        config: Any,
        timeout_s: float,
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        try:
            async def _do():
                return await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                )

            resp = await asyncio.wait_for(_do(), timeout=timeout_s)
            return self._vertex_extract_text(resp), False, None
        except asyncio.TimeoutError:
            return None, True, "timeout"
        except Exception as e:
            return None, False, f"vertex_exception:{e!r}"

    def compute(
        self,
        texts: List[str],
        batch_non_tensor: Dict[str, Any],
        tokenizer,
        cfg: Any,
        gpu_actor=None,
    ) -> Dict[str, List[float]]:
        alg = cfg.algorithm

        def _load_text(path_key: str, inline_key: str) -> Optional[str]:
            path_val = getattr(alg, path_key, None)
            if isinstance(path_val, str) and path_val.strip():
                p = path_val.strip()
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as fh:
                        return fh.read()
                print(f"[story_quality] WARNING: {path_key}={p!r} does not exist; falling back to inline config.")
            return getattr(alg, inline_key, None)

        def _load_schema(path_key: str, inline_key: str) -> Optional[Any]:
            path_val = getattr(alg, path_key, None)
            if isinstance(path_val, str) and path_val.strip():
                p = path_val.strip()
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                print(f"[story_quality] WARNING: {path_key}={p!r} does not exist; falling back to inline config.")
            return getattr(alg, inline_key, None)

        system_message = _load_text("story_quality_system_message_path", "story_quality_system_message") or ""
        user_template = _load_text("story_quality_user_template_path", "story_quality_user_template") or "{story}"
        schema_raw = _load_schema("story_quality_schema_path", "story_quality_schema")

        system_message = system_message.strip()
        if not system_message or schema_raw is None:
            raise RuntimeError(
                "story_quality_system_message and story_quality_schema must be set in config "
                "(or via story_quality_*_path variants pointing to existing files)."
            )

        if isinstance(schema_raw, str):
            try:
                schema_obj = json.loads(schema_raw)
            except json.JSONDecodeError:
                schema_obj = yaml.safe_load(schema_raw)
        else:
            schema_obj = schema_raw

        score_divisor_cfg = getattr(alg, "story_quality_score_divisor", None)
        if score_divisor_cfg is not None:
            score_divisor = float(score_divisor_cfg)
        else:
            try:
                os_props = (
                    (schema_obj or {})
                    .get("schema", {})
                    .get("properties", {})
                    .get("overall_score", {})
                )
                inferred_max = os_props.get("maximum", None)
                inferred_min = os_props.get("minimum", None)
                if isinstance(inferred_max, (int, float)):
                    score_divisor = float(inferred_max)
                    print(
                        f"[story_quality] Inferred score_divisor={score_divisor} from schema "
                        f"(max={inferred_max}, min={inferred_min}). "
                        "Set story_quality_score_divisor explicitly to suppress this warning."
                    )
                    if isinstance(inferred_min, (int, float)) and inferred_min < 0:
                        print(
                            f"[story_quality] Note: overall_score can be negative (min={inferred_min}); "
                            "rewards will be negative for poor stories."
                        )
                else:
                    score_divisor = 100.0
                    print(
                        "[story_quality] WARNING: story_quality_score_divisor not set and schema has no "
                        "maximum annotation on overall_score; defaulting to 100.0. "
                        "Set story_quality_score_divisor explicitly to suppress this warning."
                    )
            except Exception:
                score_divisor = 100.0

        if score_divisor == 0.0:
            print("[story_quality] WARNING: score_divisor resolved to 0.0; resetting to 100.0 to avoid division by zero.")
            score_divisor = 100.0

        url = str(getattr(alg, "story_quality_endpoint_url", self._endpoint_url))
        timeout_s = float(getattr(alg, "story_quality_timeout_s", 120.0))
        async_enable = bool(getattr(alg, "story_quality_async_enable", True))
        concurrency = max(1, int(getattr(alg, "story_quality_concurrency", 4)))

        provider = self._get_provider(cfg)
        force_mode = self._get_force_structured_mode(cfg)
        model = str(getattr(cfg.algorithm, "story_quality_model_name", ""))

        if async_enable and provider == "openai" and not _HAS_AIOHTTP:
            async_enable = False
        if async_enable and provider == "vertex" and not _HAS_GENAI:
            async_enable = False

        if force_mode == "native":
            use_native = True
        elif force_mode == "prompt":
            use_native = False
        else:
            use_native = supports_native_schema(provider, model)

        system_message_final = system_message
        if not use_native:
            system_message_final = self._maybe_embed_schema_in_system(system_message_final, schema_obj)

        max_retries = int(getattr(alg, "story_quality_max_retries", 0) or 0)
        backoff_s = float(getattr(alg, "story_quality_retry_backoff_s", 0.0) or 0.0)

        prompts = batch_non_tensor.get("story_prompt") or batch_non_tensor.get("prompt") or [""] * len(texts)

        headers = self._make_headers() if provider == "openai" else None
        bodies: List[Dict[str, Any]] = []
        vertex_reqs: List[Tuple[str, Any]] = []

        vertex_response_schema_obj = None
        if provider == "vertex" and use_native:
            sanitize_enable = bool(getattr(cfg.algorithm, "story_quality_vertex_schema_sanitize", True))
            enum_mode = str(getattr(cfg.algorithm, "story_quality_vertex_numeric_enum_mode", "range")).lower()
            const_mode = str(getattr(cfg.algorithm, "story_quality_vertex_numeric_const_mode", "range")).lower()

            inner_raw = (schema_obj or {}).get("schema") or {}
            inner_use = inner_raw
            if sanitize_enable:
                inner_use = sanitize_json_schema_for_genai(
                    inner_raw,
                    numeric_enum_mode=enum_mode,
                    numeric_const_mode=const_mode,
                )

            try:
                if not _HAS_GENAI:
                    raise RuntimeError("Vertex provider selected but google-genai is unavailable.")
                vertex_response_schema_obj = genai_types.Schema.model_validate(inner_use)
            except Exception as e:
                use_native = False
                system_message_final = self._maybe_embed_schema_in_system(system_message, schema_obj)
                vertex_response_schema_obj = None
                print(f"[story_quality] Vertex response_schema validate failed; falling back to prompt schema. err={e!r}")

        for i, story in enumerate(texts):
            prompt_text = ""
            try:
                if isinstance(prompts, list) and i < len(prompts) and isinstance(prompts[i], str):
                    prompt_text = prompts[i]
            except Exception:
                prompt_text = ""
            prompt_clean = _strip_length_requirement_sentence(prompt_text)

            story_for_model = story or ""
            fmt = {
                "story_prompt": prompt_clean or "",
                "prompt": prompt_clean or "",
                "story": story_for_model,
                "story_text": story_for_model,
            }

            try:
                user_msg = user_template.format(**fmt)
            except Exception:
                user_msg = user_template.format(story=fmt.get("story", story_for_model))

            if provider == "openai":
                body = self._make_body(
                    system_msg=system_message_final,
                    user_msg=user_msg,
                    schema_obj=schema_obj,
                    cfg=cfg,
                    use_native_schema=use_native,
                )
                bodies.append(body)
            else:
                vcfg: Dict[str, Any] = {
                    "system_instruction": system_message_final,
                    "temperature": float(getattr(cfg.algorithm, "story_quality_temperature", 0.0) or 0.0),
                    "top_p": float(getattr(cfg.algorithm, "story_quality_top_p", 1.0) or 1.0),
                    "max_output_tokens": int(getattr(cfg.algorithm, "story_quality_max_output_tokens", 4096)),
                    "labels": {"user": getattr(cfg.algorithm, "vertexai_user", "polaris")},
                }
                if use_native and vertex_response_schema_obj is not None:
                    vcfg["response_mime_type"] = "application/json"
                    vcfg["response_schema"] = vertex_response_schema_obj
                thinking_cfg = getattr(cfg.algorithm, "story_quality_thinking_config", None)
                if thinking_cfg:
                    vcfg["thinking_config"] = thinking_cfg
                if not _HAS_GENAI:
                    raise RuntimeError("Vertex provider selected but google-genai is unavailable.")
                allowed = set(getattr(genai_types.GenerateContentConfig, "__annotations__", {}).keys())
                vcfg_f = {k: v for k, v in vcfg.items() if k in allowed}
                vertex_reqs.append((user_msg, genai_types.GenerateContentConfig(**vcfg_f)))

        N = len(texts)
        results_text: List[Optional[str]] = [None] * N
        results_timeout: List[bool] = [False] * N
        results_err: List[Optional[str]] = [None] * N

        if async_enable and N > 0:
            if provider == "openai":
                async def runner_openai():
                    sem = asyncio.Semaphore(concurrency)
                    async with aiohttp.ClientSession() as session:
                        tasks = []
                        for idx, body in enumerate(bodies):
                            async def do(ii=idx, b=body):
                                async with sem:
                                    attempts = 0
                                    while True:
                                        text, timed_out, err = await self._call_single_async(session, url, headers, b, timeout_s)
                                        attempts += 1
                                        if timed_out and attempts <= (1 + max_retries):
                                            if backoff_s > 0:
                                                await asyncio.sleep(backoff_s)
                                            continue
                                        results_text[ii] = text
                                        results_timeout[ii] = timed_out
                                        results_err[ii] = err
                                        break
                            tasks.append(asyncio.create_task(do()))
                        await asyncio.gather(*tasks)

                try:
                    asyncio.run(runner_openai())
                except RuntimeError:
                    import concurrent.futures

                    def _run_in_thread(coro_fn):
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(coro_fn())
                        finally:
                            loop.close()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(_run_in_thread, runner_openai).result()
            else:
                async def runner_vertex():
                    client = self._ensure_vertex_client(cfg)
                    sem = asyncio.Semaphore(concurrency)
                    model_name = str(getattr(cfg.algorithm, "story_quality_model_name", "gemini-2.5-flash"))
                    tasks = []
                    for idx, (contents, vcfg) in enumerate(vertex_reqs):
                        async def do(ii=idx, c=contents, conf=vcfg):
                            async with sem:
                                attempts = 0
                                while True:
                                    text, timed_out, err = await self._call_vertex_single_async_threaded(client, model_name, c, conf, timeout_s)
                                    attempts += 1
                                    if timed_out and attempts <= (1 + max_retries):
                                        if backoff_s > 0:
                                            await asyncio.sleep(backoff_s)
                                        continue
                                    results_text[ii] = text
                                    results_timeout[ii] = timed_out
                                    results_err[ii] = err
                                    break
                        tasks.append(asyncio.create_task(do()))
                    await asyncio.gather(*tasks)

                try:
                    asyncio.run(runner_vertex())
                except RuntimeError:
                    import concurrent.futures

                    def _run_in_thread(coro_fn):
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(coro_fn())
                        finally:
                            loop.close()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(_run_in_thread, runner_vertex).result()

        if not async_enable:
            if provider == "openai":
                for i, body in enumerate(bodies):
                    attempts = 0
                    while True:
                        text, timed_out, err = self._call_single_sync(url, headers, body, timeout_s)
                        attempts += 1
                        if timed_out and attempts <= (1 + max_retries):
                            if backoff_s > 0:
                                import time as _t
                                _t.sleep(backoff_s)
                            continue
                        results_text[i] = text
                        results_timeout[i] = timed_out
                        results_err[i] = err
                        break
            else:
                client = self._ensure_vertex_client(cfg)
                model_name = str(getattr(cfg.algorithm, "story_quality_model_name", "gemini-2.5-flash"))
                for i, (contents, vcfg) in enumerate(vertex_reqs):
                    attempts = 0
                    while True:
                        try:
                            resp = client.models.generate_content(model=model_name, contents=contents, config=vcfg)
                            text = self._vertex_extract_text(resp)
                            results_text[i] = text
                            results_timeout[i] = False
                            results_err[i] = None
                            break
                        except Exception as e:
                            attempts += 1
                            if attempts <= (1 + max_retries):
                                if backoff_s > 0:
                                    import time as _t
                                    _t.sleep(backoff_s)
                                continue
                            results_text[i] = None
                            results_timeout[i] = False
                            results_err[i] = f"vertex_exception:{e!r}"
                            break

        def _flatten_numeric_storyq_fields(src: dict) -> Dict[str, float]:
            out: Dict[str, float] = {}
            if not isinstance(src, dict):
                return out
            for k, v in src.items():
                if isinstance(v, (int, float)):
                    if k == "overall_score" or k.endswith("_score") or k.endswith("_total"):
                        out[f"storyq_{k}"] = float(v)
                elif isinstance(v, dict):
                    score_val = v.get("score", None)
                    if isinstance(score_val, (int, float)):
                        out[f"storyq_{k}_score"] = float(score_val)
            if "storyq_overall_score" in out and score_divisor != 0.0:
                out["storyq_overall_score_norm"] = out["storyq_overall_score"] / score_divisor
            return out

        scores: List[float] = []
        extras: List[Dict[str, Any]] = []
        per_row_flat: List[Dict[str, float]] = []
        all_flat_keys = set()

        for i, text_json in enumerate(results_text):
            valid = False
            reward = 0.0
            diag: Dict[str, Any] = {
                "valid": False,
                "overall_score": None,
                "positive_total": None,
                "negative_total": None,
                "bonus_total": None,
                "penalty_total": None,
                "timed_out": bool(results_timeout[i]),
                "error": results_err[i],
                "raw": None,
            }
            flat_i: Dict[str, float] = {}

            if text_json:
                try:
                    clean = _strip_code_fences(text_json)
                    candidate = clean
                    try:
                        resp = json.loads(candidate)
                    except Exception:
                        candidate2 = _extract_first_json_object(clean)
                        if not candidate2:
                            raise
                        resp = json.loads(candidate2)

                    diag["raw"] = resp
                    src = resp
                    if isinstance(resp, dict) and "global_evaluation" in resp:
                        src = resp.get("global_evaluation", {}) or {}

                    pos_tot = src.get("positive_total", None)
                    neg_tot = src.get("negative_total", None)
                    bonus_tot = src.get("bonus_total", src.get("bonus_score", 0))
                    penalty_tot = src.get("penalty_total", src.get("penalty_score", 0))
                    overall = src.get("overall_score", None)

                    diag["positive_total"] = pos_tot
                    diag["negative_total"] = neg_tot
                    diag["bonus_total"] = bonus_tot
                    diag["penalty_total"] = penalty_tot
                    diag["overall_score"] = overall

                    override_enable = bool(getattr(cfg.algorithm, "story_quality_overall_override_enable", False))
                    if override_enable:
                        if isinstance(pos_tot, (int, float)) and isinstance(neg_tot, (int, float)):
                            pos_f = float(pos_tot)
                            neg_f = float(neg_tot)
                            diff = pos_f - neg_f

                            bonus_contrib = float(bonus_tot or 0)
                            penalty_contrib = float(penalty_tot or 0)
                            bonus_floor_enable = bool(getattr(cfg.algorithm, "story_quality_bonus_floor_enable", False))
                            penalty_cap_enable = bool(getattr(cfg.algorithm, "story_quality_penalty_cap_enable", False))
                            penalty_cap_value = float(getattr(cfg.algorithm, "story_quality_penalty_cap_value", 3))
                            if bonus_floor_enable and neg_f > pos_f / 2.0:
                                diag["bonus_floor_applied"] = True
                                bonus_contrib = 0.0
                            if penalty_cap_enable and pos_f > neg_f * 1.5 and penalty_contrib > penalty_cap_value:
                                diag["penalty_cap_applied"] = True
                                penalty_contrib = penalty_cap_value

                            scale = float(getattr(cfg.algorithm, "story_quality_overall_scale", 1.0))
                            bias = float(getattr(cfg.algorithm, "story_quality_overall_bias", 0.0))
                            new_overall = scale * (diff + bonus_contrib - penalty_contrib) + bias

                            clamp_min = float(getattr(cfg.algorithm, "story_quality_overall_min", 0.0))
                            clamp_max = float(getattr(cfg.algorithm, "story_quality_overall_max", 100))

                            dq_enable = bool(getattr(cfg.algorithm, "story_quality_dq_enable", False))
                            if dq_enable:
                                dq_n2_thresh = float(getattr(cfg.algorithm, "story_quality_dq_n2_threshold", 7))
                                dq_n10_thresh = float(getattr(cfg.algorithm, "story_quality_dq_n10_threshold", 8))
                                dq_wc_min_frac = float(getattr(cfg.algorithm, "story_quality_dq_wc_min_frac", 0.40))
                                n2_score = src.get("negative_coherence_continuity_internal_consistency_pov_confusion_score", None)
                                n10_score = src.get("negative_overwrought_nonsensical_language_score", None)
                                wc_dq = False
                                if dq_wc_min_frac > 0:
                                    target_wc = None
                                    try:
                                        m = WORDCOUNT_REGEX.search(prompts[i] if i < len(prompts) else "")
                                        if m:
                                            target_wc = int(m.group("word_count"))
                                    except Exception:
                                        pass
                                    if target_wc:
                                        story_text = texts[i] if i < len(texts) else ""
                                        actual_wc = len(story_text.split())
                                        if actual_wc < dq_wc_min_frac * target_wc:
                                            wc_dq = True
                                n2_dq = isinstance(n2_score, (int, float)) and float(n2_score) >= dq_n2_thresh
                                n10_dq = isinstance(n10_score, (int, float)) and float(n10_score) >= dq_n10_thresh
                                any_dq = n2_dq or n10_dq or wc_dq
                                exempt_gt = bool(getattr(cfg.algorithm, "story_quality_dq_exempt_gt", True))
                                if exempt_gt:
                                    is_gt_list = batch_non_tensor.get("is_gt", None)
                                    if is_gt_list is not None and i < len(is_gt_list) and bool(is_gt_list[i]):
                                        any_dq = False
                                diag["_dq_n2"] = bool(n2_dq)
                                diag["_dq_n10"] = bool(n10_dq)
                                diag["_dq_wc"] = bool(wc_dq)
                                diag["_dq_any"] = bool(any_dq)
                                if any_dq:
                                    diag["overall_score_pre_dq"] = new_overall
                                    new_overall = clamp_min

                            new_overall = max(clamp_min, min(clamp_max, new_overall))
                            diag["overall_score_model_raw"] = overall
                            diag["overall_score_overridden"] = new_overall
                            diag["overall_overridden"] = True
                            diag["bonus_contrib_effective"] = bonus_contrib
                            diag["penalty_contrib_effective"] = penalty_contrib
                            overall = new_overall
                        else:
                            diag["error"] = "overall_override_enabled_but_missing_positive_or_negative_total"

                    use_pos_only = bool(getattr(cfg.algorithm, "story_quality_use_positive_total", False))
                    if use_pos_only and isinstance(pos_tot, (int, float)):
                        base = float(pos_tot) + float(bonus_tot or 0) - float(penalty_tot or 0)
                        reward = base / score_divisor
                        valid = True
                    elif isinstance(overall, (int, float)):
                        reward = float(overall) / score_divisor
                        valid = True
                    else:
                        if "error" not in diag:
                            diag["error"] = "missing_or_invalid_overall_and_positive_total"
                        reward = 0.0
                        valid = False

                    if isinstance(src, dict):
                        flat_i = _flatten_numeric_storyq_fields(src)
                except Exception as e:
                    diag["error"] = f"parse_json_failed:{e!r}"

            diag["valid"] = bool(valid)
            scores.append(float(reward if valid else 0.0))
            extras.append(diag)
            per_row_flat.append(flat_i)
            all_flat_keys |= set(flat_i.keys())

        out: Dict[str, Any] = {"story_quality": scores, "extra_info": extras}
        for k in sorted(all_flat_keys):
            out[k] = [float(d.get(k, float("nan"))) for d in per_row_flat]
        return out
