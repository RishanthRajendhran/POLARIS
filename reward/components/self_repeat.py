# reward/components/self_repeat.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re
from collections import Counter

from .base import RewardComponent

_WORD_RE = re.compile(r"\w+|[^\s]")


def _scan_tokens(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    toks = []
    for m in _WORD_RE.finditer(text):
        s, e = m.span()
        toks.append((m.group(0).lower(), (s, e)))
    return toks


def _tokens(text: str) -> List[str]:
    return [t for t, _ in _scan_tokens(text)]


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class SelfRepeatComponent(RewardComponent):
    name = "self_rep"

    def enabled(self, cfg: Any) -> bool:
        return bool(getattr(cfg.algorithm, "reward_components", {}).get("enable_self_rep", False))

    def needs_gpu(self) -> bool:
        return False

    def keys(self) -> List[str]:
        return ["self_rep"]

    def compute(
        self,
        texts: List[str],
        batch_non_tensor: Dict[str, Any],
        tokenizer,
        cfg: Any,
        gpu_actor=None
    ) -> Dict[str, List[float]]:
        n_local  = int(getattr(cfg.algorithm, "rep_n_local", 4))
        local_w  = int(getattr(cfg.algorithm, "rep_local_window", 80))
        hinge    = float(getattr(cfg.algorithm, "rep_ngram_hinge", 0.10))
        w_lines  = float(getattr(cfg.algorithm, "rep_w_lines", 0.5))
        w_ngram  = float(getattr(cfg.algorithm, "rep_w_ngram", 1.0))
        cap      = float(getattr(cfg.algorithm, "rep_cap", 1.0))

        out: List[float] = []
        extra: List[dict] = []

        for t in texts:
            toks_with_span = _scan_tokens(t)
            toks = [tok for tok, _ in toks_with_span]
            grams = _ngrams(toks, n_local)

            # ---- global n-gram repetition ----
            if grams:
                c = Counter(grams)
                repeats = sum((k - 1) for k in c.values() if k > 1)
                ngram_ratio = repeats / max(1, len(grams))
            else:
                ngram_ratio = 0.0

            # ---- duplicate lines ----
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            lc = Counter(lines)
            dup_lines = sum((k - 1) for k in lc.values() if k > 1)
            line_repeat_ratio = dup_lines / max(1, len(lines)) if lines else 0.0

            # ---- sliding-window local repetition ----
            local_hits = 0
            if grams:
                seen: dict = {}
                for j, g in enumerate(grams):
                    kill_idx = j - max(1, local_w)
                    if kill_idx in seen:
                        del seen[kill_idx]
                    if g in (val for _, val in seen.values()):
                        local_hits += 1
                    seen[j] = (j, g)
            local_ratio = local_hits / max(1, len(grams)) if grams else 0.0

            excess = max(0.0, ngram_ratio - hinge)
            penalty_scalar = min(cap, w_ngram * (0.7 * excess + 0.3 * local_ratio) + w_lines * line_repeat_ratio)
            out.append(float(penalty_scalar))

            # ---- token-level plan (char spans) ----

            # global n-gram duplicate spans (second+ occurrence of each gram)
            ngram_spans: List[Tuple[int, int]] = []
            if grams:
                index_map: dict = {}
                for j, g in enumerate(grams):
                    index_map.setdefault(g, []).append(j)
                for g, idxs in index_map.items():
                    if len(idxs) <= 1:
                        continue
                    for i_idx in idxs[1:]:
                        s_tok = i_idx
                        e_tok = i_idx + n_local - 1
                        s_char = toks_with_span[s_tok][1][0]
                        e_char = toks_with_span[e_tok][1][1]
                        ngram_spans.append((s_char, e_char))

            # local-window duplicate spans: grams that appear within the window
            # (distinct from the global set to avoid double-counting)
            local_spans: List[Tuple[int, int]] = []
            if grams:
                local_seen: dict = {}
                for j, g in enumerate(grams):
                    kill_idx = j - max(1, local_w)
                    if kill_idx in local_seen:
                        del local_seen[kill_idx]
                    if g in (val for _, val in local_seen.values()):
                        s_tok = j
                        e_tok = j + n_local - 1
                        s_char = toks_with_span[s_tok][1][0]
                        e_char = toks_with_span[e_tok][1][1]
                        local_spans.append((s_char, e_char))
                    local_seen[j] = (j, g)

            # duplicate line spans
            line_spans: List[Tuple[int, int]] = []
            if lines:
                raw = t.splitlines(keepends=True)
                pos = 0
                line_texts_spans: List[Tuple[str, Tuple[int, int]]] = []
                for chunk in raw:
                    s = pos
                    e = pos + len(chunk)
                    pos = e
                    line_texts_spans.append((chunk.strip(), (s, e)))
                seen_lines: set = set()
                for lt, (s, e) in line_texts_spans:
                    if not lt:
                        continue
                    if lt in seen_lines:
                        line_spans.append((s, e))
                    else:
                        seen_lines.add(lt)

            ngram_budget = max(0.0, w_ngram * (0.7 * excess))
            local_budget = max(0.0, w_ngram * (0.3 * local_ratio))
            line_budget  = max(0.0, w_lines * line_repeat_ratio)

            extra.append({
                "mode": "token_plan",
                "cap": float(cap),
                "ngram": {
                    "spans": [{"start_char": int(s), "end_char": int(e)} for s, e in ngram_spans],
                    "total_penalty": float(ngram_budget),
                },
                "local": {
                    "spans": [{"start_char": int(s), "end_char": int(e)} for s, e in local_spans],
                    "total_penalty": float(local_budget),
                },
                "lines": {
                    "spans": [{"start_char": int(s), "end_char": int(e)} for s, e in line_spans],
                    "total_penalty": float(line_budget),
                },
            })

        return {"self_rep": out, "extra_info": extra}
