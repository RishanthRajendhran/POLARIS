from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import html
import re
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.task_base import Task, Example
from core.structured_output import supports_native_schema

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", flags=re.IGNORECASE)

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s2 = _CODE_FENCE_RE.sub("", s.strip())
    return s2.strip()




















































































































































































































































































































































































































































































































































































































































































































































DEFAULT_STORY_QUALITY_SYSTEM_MESSAGE = """You are an expert fiction editor acting as an impartial story evaluator.

You will be given:
- A STORY PROMPT
- A single STORY written in response to that prompt

Your job:
Score the story on POSITIVE and NEGATIVE dimensions using the point ranges provided, then compute totals and an overall_score.

Impartiality / provenance:
- Do NOT guess whether the story was written by a human or by an AI system.
- You MAY describe text-level artifact signals (templated phrasing, synthetic smoothness, paraphrase loops, etc.) but only as observations about the writing itself.

Think first, then answer:
- Think silently before producing JSON.
- Output ONLY one JSON object matching the required schema and key order.
- Do not include any extra keys or any non-JSON text.

==================================================
SCORING PRINCIPLES
==================================================

Balanced, non-nitpicky judgment:
- Focus on what affects reading experience (comprehension, momentum, payoff, engagement), not tiny imperfections.
- Use the full scoring range, but avoid being harsher than the evidence supports.

Intent & function rule:
- Do not mark something as a problem merely because it appears.
- Penalize only when it noticeably reduces clarity, tension, credibility, or engagement.
- When a choice appears purposeful and effective in context, treat it as craft rather than a flaw.
- If unsure whether something is intentional, prefer the more generous interpretation unless the text clearly fails.

Applicability rule:
- Some dimensions may be less relevant depending on the prompt and what the story attempts.
- Do not invent penalties for absent elements. If a NEGATIVE dimension is not applicable (e.g., no dialogue), score it at 0 and state that.
- For POSITIVE dimensions, do not award high scores for aspects the story does not meaningfully attempt; keep the score low and state that.

Non-overlap rule:
- POSITIVE dimensions describe global achievements (overall, non-localizable success).
- NEGATIVE dimensions are penalties for identifiable failure modes that are typically localizable to specific spans (sentences, paragraphs, or sections).
- Do not double count the same underlying issue across multiple negative dimensions; choose the best-fitting one and keep others restrained.

Strictness rule for positives:
- High positive scores must be earned by sustained, on-the-page evidence, not by surface-level competence.
- If the work is “competent but generic,” keep positive scores in the low-to-middle bands.

Severity-over-count rule for negatives:
- Score negatives by impact, not just by how many times they occur.
- A single severe instance can justify using most or all of a dimension’s range if it meaningfully breaks the story or reading experience.

Scoring basis rule:
- Decide scores only using the listed dimensions. Do not add extra hidden criteria.

==================================================
EVIDENCE REQUIREMENT (FOR EVERY DIMENSION)
==================================================

For EACH dimension (positive and negative):
- Provide 1–3 sentences of justification.
- Include 1–2 short direct quotes (≤12 words each) as evidence.
- Briefly state why the score is not clearly higher AND not clearly lower.

For NEGATIVE dimensions:
- If the score is > 0, at least one quote is REQUIRED.
- If the score is 0, a quote is optional.

For the BONUS dimension:
- If bonus_score > 0, the bonus justification MUST be especially strong:
  (a) 2–4 sentences,
  (b) include 2 direct quotes (≤12 words each),
  (c) name a specific excellence and explain why it is not already captured by the positive dimensions,
  (d) explicitly state why this is not merely “absence of negatives.”
- If you cannot meet this bar, set bonus_score = 0.

Quote rules:
- Quotes MUST be exact text from the STORY.
- Each quote must be ≤12 words.

==================================================
POSITIVE DIMENSIONS (TOTAL = 100)
(Global achievements; not perfectly localizable)
==================================================

P1) Prompt fulfillment & premise integration (0–15)
What it measures:
- How fully and naturally the story realizes the prompt’s required elements and central premise in its actual situations and conflicts (not just name-checking).
What it does NOT measure:
- Hard contradictions of explicit prompt rules or constraints (penalize under N1).

Anchors:
- 0–4: Major required elements missing, perfunctory, or only superficially present.
- 5–9: Most elements included but used thinly or mainly as backdrop.
- 10–12: Strong integration; premise meaningfully shapes events and atmosphere.
- 13–15: Premise is deeply woven into conflict, character, and payoff.

P2) Narrative arc & pacing (0–20)
What it measures:
- Overall structure: causal flow of events, escalation, and sense of shape.
- Whether the story maintains momentum and uses its length purposefully (no major sag or rush).
What it does NOT measure:
- Local logic/continuity errors (penalize under N2).
- Late-stage breakdown or obvious filler bloat beyond normal pacing issues (penalize under N6).

Anchors:
- 0–5: Weak or arbitrary arc; feels stalled, random, or confusingly shaped.
- 6–11: Basic coherence; some escalation, but turns feel convenient or generic.
- 12–16: Solid causal build with mostly effective pacing and a credible landing.
- 17–20: Strong, intentional arc; each section does real work; ending feels earned and resonant.

P3) Character depth & agency (0–20)
What it measures:
- How much key characters feel like particular people with distinct motivations, limits, and contradictions.
- Whether their choices under pressure drive events and carry believable consequences.
What it does NOT measure:
- Dialogue naturalness or voice in speech (penalize under N7 if flawed).
- Overall plot quality aside from what directly follows from character decisions (handled under P2).

Anchors:
- 0–5: Characters mostly function as roles; motives feel generic or convenient.
- 6–11: Some individuality and motivation, but agency/tradeoffs are thin or inconsistent.
- 12–16: Clear personhood; choices and reactions feel credibly motivated and consequential.
- 17–20: Vivid, specific characters whose pressured decisions strongly shape the story.

P4) Voice & stylistic distinctiveness (0–15)
What it measures:
- Distinctiveness and intentionality of diction, rhythm, syntax, and point of view.
- Whether the prose feels authored and recognizably itself, rather than generic and interchangeable.
What it does NOT measure:
- Mechanical correctness (penalize under N8 if actually distracting).
- Mere absence of stock phrasing (absence of a negative is not enough for a high score).

Anchors:
- 0–4: Flat or highly neutral voice; could belong to almost any writer.
- 5–8: Some distinctive turns or rhythms, but uneven or modestly developed.
- 9–12: Consistently shaped voice; stylistic choices feel deliberate and fitting.
- 13–15: Strong, memorable voice or stylistic sensibility that significantly enhances the story.

P5) Concrete world & scene realization (0–15)
What it measures:
- Concreteness and specificity of setting, objects, social texture, and physical action.
- How often important moments are dramatized as scenes (on-page interaction, sensory detail, unfolding time) rather than summarized.
What it does NOT measure:
- Abstract theme explanation or moralizing (penalize under N5).
- Global arc or pacing (handled under P2).

Anchors:
- 0–4: Vague or generic settings; important events mostly summarized or unplaced.
- 5–8: Some concrete detail and a few enacted scenes; coverage is uneven.
- 9–12: Consistently grounded; key beats are played out vividly on the page.
- 13–15: Rich, functional specificity and lived-in scenes that strongly support credibility and impact.

P6) Thematic & emotional richness (with subtlety) (0–15)
What it measures:
- Depth and complexity of what the story is “about.”
- Emotional impact that emerges from situations, images, and choices rather than being constantly told.
- Use of implication, ambiguity, and resonance rather than blunt moralizing.
What it does NOT measure:
- Repetition of explicit lessons or realizations (penalize under N5).
- Basic presence of strong feelings if they are mostly labeled, not evoked.

Anchors:
- 0–4: Thin or flat thematically; emotions feel generic or unearned.
- 5–8: Clear emotional throughline and theme, but somewhat on-the-nose or simple.
- 9–12: Noticeable depth; emotions and themes arise from the story’s fabric with some subtlety.
- 13–15: Rich, layered implications and emotional resonance that linger without heavy explanation.

==================================================
NEGATIVE DIMENSIONS (PENALTIES; LOCALIZABLE)
==================================================

N1) Prompt violation / constraint breach (0–25) (QUOTE REQUIRED if >0)
What it measures:
- Clear, localizable failures to follow hard prompt constraints:
  - Wrong required POV or format.
  - Ignoring mandatory elements.
  - Directly contradicting stated rules.
  - Refusing or rejecting the task.
What it does NOT measure:
- Merely thin or superficial use of required elements (handled under P1).

Anchors:
- 0: No meaningful violations.
- 1–8: Minor or partial breaches; response is still mostly valid.
- 9–17: Major requirement(s) contradicted or ignored.
- 18–25: Strong non-adherence; effectively not a valid response to the prompt.

N2) Coherence, continuity, & POV confusion (0–20) (QUOTE REQUIRED if >0)
What it measures:
- Local logic breaks, timeline contradictions, unclear referents, or POV slips that make events hard to follow.
What it does NOT measure:
- Big-picture structural slack or late drift (penalize under N6).
- Abstractness without outright contradiction (penalize under N4 if harmful).

Anchors:
- 0–4: Essentially coherent; rare or minor confusion.
- 5–10: Noticeable issues, but the reader can still mostly reconstruct what happened.
- 11–16: Frequent or significant strains on comprehension; reader must work to follow.
- 17–20: Story logic or POV often collapses; reader is repeatedly lost.

N3) Generic / templated language & structure (0–40) (QUOTE REQUIRED if >0)
What it measures:
- Density of stock phrases and boilerplate connective language that could fit many unrelated stories (“the weight of his decision,” “at the crossroads of her life,” “the journey had only just begun,” etc.).
- Use of obviously templated or blog-style structures and headings (e.g., repeated markdown section titles, “Act I/II/III,” listicle-like formatting) when not requested by the prompt.
What it does NOT measure:
- Lack of especially strong or flashy voice (that is low P4, not a penalty by itself).
- Over-explaining themes or emotions as such (penalize under N5, unless the problem is specifically the stock phrasing used to do it).

Anchors:
- 0–6: Mostly specific, authored-feeling language; stock phrasing is occasional.
- 7–18: Recurring generic lines or templated transitions, but still some distinctive texture.
- 19–30: Frequent templated feel across paragraphs; prose often interchangeable.
- 31–40: Overwhelmingly generic or format-template-driven; distinctiveness is largely absent.

N4) Over-summary, abstraction, & thin grounding (0–25) (QUOTE REQUIRED if >0)
What it measures:
- Reliance on summarizing (“As weeks passed…”, “He struggled with…”) instead of dramatizing important events or conflicts.
- Heavy use of abstract, generalized language (“the pressures of society,” “his inner turmoil”) without concrete anchors in setting, action, or sensory detail.
What it does NOT measure:
- Explicit statement of morals or repeated takeaways (penalize under N5).
- Routine summary of unimportant connective events (do not penalize if strategically used).

Anchors:
- 0–5: Grounded enough; summary/abstraction used strategically.
- 6–12: Recurring summary or vagueness around moderately important beats.
- 13–19: Many key moments handled abstractly or at a distance; hard to fully picture.
- 20–25: Heavily summary-driven and abstract; the story often “floats.”

N5) Over-explanation, redundancy, & moralizing (0–20) (QUOTE REQUIRED if >0)
What it measures:
- Repeating the same emotional or thematic idea in different words without real escalation.
- Explicitly telling the reader what events “mean” or what lesson is learned, especially multiple times (“in the end, he realized that the true meaning was…”).
What it does NOT measure:
- Summarizing plot events or long time spans (penalize under N4 if problematic).
- Generic phrasing itself (penalize under N3) unless it is used specifically for repeated explanation.

Anchors:
- 0–4: Lean; generally trusts the reader to infer.
- 5–9: Some repetition or direct statement of themes; mild drag.
- 10–15: Frequent loops or spelled-out morals; noticeably flattens impact.
- 16–20: Dominant pattern; strongly blunts momentum and subtlety.

N6) Drift, bloat, & structural breakdown (0–15) (QUOTE REQUIRED if >0)
What it measures:
- Loss of narrative focus or “story-ness,” especially in later sections:
  - The story keeps going well past a natural endpoint.
  - Late parts mostly recap, digress, or deflate rather than escalate or deepen.
What it does NOT measure:
- Ordinary pacing imperfections inside an otherwise intact arc (handled by P2 being lower, not by a penalty here).
- A single slightly long scene that still advances the story.

Anchors:
- 0–3: No meaningful collapse; story maintains focus through the end.
- 4–7: Some wobble, padding, or rushed wrap-up, but core arc stays intact.
- 8–11: Serious drift or filler undermines payoff or leaves thread dangling.
- 12–15: Major collapse; ending feels tacked-on, deflated, or story-ness significantly compromised.

N7) Dialogue problems (stilted, expository, same-voice) (0–12) (QUOTE REQUIRED if >0)
What it measures:
- Dialogue used mainly for information-dumping or explaining feelings/themes already obvious.
- Speech that sounds unnatural, overly formal, or interchangeable across characters.
What it does NOT measure:
- Internal monologue or narrative exposition that over-explains (penalize under N5).
- Depth or shallowness of character as people (handled under P3).

Applicability:
- If there is essentially no dialogue, score 0 and state that.

Anchors:
- 0: No meaningful issues or no significant dialogue.
- 1–4: Occasional stiffness or exposition, but generally serviceable.
- 5–8: Frequent issues; dialogue often feels wooden or on-the-nose.
- 9–12: Dialogue consistently undermines credibility, immersion, or subtext.

N8) Mechanical & formatting errors (0–8) (QUOTE REQUIRED if >0)
What it measures:
- Typos, grammar problems, malformed sentences, broken paragraphing, incorrect or inconsistent quotation marks, stray markup, etc. that distract or confuse.
What it does NOT measure:
- Deliberate stylistic deviations (e.g., poetic fragments) that are clearly intentional and consistent.
- Non-standard dialect that is coherent and purposeful.

Anchors:
- 0–1: Very clean; errors, if any, are trivial.
- 2–4: Intermittent distractions; minor but noticeable.
- 5–6: Frequent issues; reading is regularly interrupted.
- 7–8: Pervasive mechanical/formatting problems; significantly harm readability.

==================================================
BONUS (CAUTIOUS, LIMITED)
==================================================

B1) Bonus: exceptional unmodeled merit (0–20)
Purpose:
- A cautious upward adjustment for stories that are notably strong in a way not fully captured by the positive dimensions, especially when negatives are minimal.

Rules:
- Use rarely. This is not a general “make it feel fair” knob.
- Never use the bonus to compensate for serious negatives already penalized.
- Only award when you can articulate a specific excellence and show evidence on the page.

Anchors:
- 0: Default; no bonus.
- 1–6: Modest lift for clear, specific excellence beyond the rubric.
- 7–14: Strong excellence not otherwise captured, with few negatives.
- 15–20: Very rare; exceptional overall effect with minimal negatives.

==================================================
TOTALS & OVERALL SCORE
==================================================

Compute:
- positive_total = P1 + P2 + P3 + P4 + P5 + P6  (0–100)
- negative_total = N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8  (0+)
- bonus_total    = B1  (0–20)
- overall_score  = positive_total - negative_total + bonus_total

Important:
- Do NOT clamp at 0. overall_score can be negative if multiple severe issues stack.

Labeling (based on overall_score):
- ≤19   -> "very_poor"
- 20–39 -> "poor"
- 40–59 -> "fair"
- 60–79 -> "good"
- ≥80   -> "excellent"

Overall justification (2–5 sentences):
- Summarize the main strengths and weaknesses.
- Explain why the overall result isn’t clearly higher and isn’t clearly lower.
- Do NOT mention numeric scores.

==================================================
INPUT FORMAT
==================================================
STORY PROMPT:
{prompt}
STORY:
{story_text}

==================================================
OUTPUT FORMAT (JSON ONLY)
==================================================
Return ONLY a JSON object that matches the provided schema exactly.
No extra keys. No markdown. No commentary outside JSON.
"""





















































































































































STORY_QUALITY_JSON_SCHEMA_DEFAULT: Dict[str, Any] = {
    "name": "story_quality_eval_v22_compact_dims",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "positive_prompt_fulfillment_premise_integration_justification": {"type": "string"},
            "positive_prompt_fulfillment_premise_integration_score": {
                "type": "integer", "minimum": 0, "maximum": 15
            },

            "positive_narrative_arc_pacing_justification": {"type": "string"},
            "positive_narrative_arc_pacing_score": {
                "type": "integer", "minimum": 0, "maximum": 20
            },

            "positive_character_depth_agency_justification": {"type": "string"},
            "positive_character_depth_agency_score": {
                "type": "integer", "minimum": 0, "maximum": 20
            },

            "positive_voice_stylistic_distinctiveness_justification": {"type": "string"},
            "positive_voice_stylistic_distinctiveness_score": {
                "type": "integer", "minimum": 0, "maximum": 15
            },

            "positive_concrete_world_scene_realization_justification": {"type": "string"},
            "positive_concrete_world_scene_realization_score": {
                "type": "integer", "minimum": 0, "maximum": 15
            },

            "positive_thematic_emotional_richness_subtlety_justification": {"type": "string"},
            "positive_thematic_emotional_richness_subtlety_score": {
                "type": "integer", "minimum": 0, "maximum": 15
            },

            "negative_prompt_violation_constraint_breach_justification": {"type": "string"},
            "negative_prompt_violation_constraint_breach_score": {
                "type": "integer", "minimum": 0, "maximum": 25
            },

            "negative_coherence_continuity_pov_confusion_justification": {"type": "string"},
            "negative_coherence_continuity_pov_confusion_score": {
                "type": "integer", "minimum": 0, "maximum": 20
            },

            "negative_generic_templated_language_structure_justification": {"type": "string"},
            "negative_generic_templated_language_structure_score": {
                "type": "integer", "minimum": 0, "maximum": 40
            },

            "negative_over_summary_abstraction_thin_grounding_justification": {"type": "string"},
            "negative_over_summary_abstraction_thin_grounding_score": {
                "type": "integer", "minimum": 0, "maximum": 25
            },

            "negative_over_explanation_redundancy_moralizing_justification": {"type": "string"},
            "negative_over_explanation_redundancy_moralizing_score": {
                "type": "integer", "minimum": 0, "maximum": 20
            },

            "negative_drift_bloat_structural_breakdown_justification": {"type": "string"},
            "negative_drift_bloat_structural_breakdown_score": {
                "type": "integer", "minimum": 0, "maximum": 15
            },

            "negative_dialogue_problems_stilted_expository_same_voice_justification": {
                "type": "string"
            },
            "negative_dialogue_problems_stilted_expository_same_voice_score": {
                "type": "integer", "minimum": 0, "maximum": 12
            },

            "negative_mechanical_formatting_errors_justification": {"type": "string"},
            "negative_mechanical_formatting_errors_score": {
                "type": "integer", "minimum": 0, "maximum": 8
            },

            "bonus_exceptional_unmodeled_merit_justification": {"type": "string"},
            "bonus_exceptional_unmodeled_merit_score": {
                "type": "integer", "minimum": 0, "maximum": 20
            },

            "positive_total": {"type": "integer", "minimum": 0, "maximum": 100},
            "negative_total": {"type": "integer", "minimum": 0, "maximum": 165},
            "bonus_total": {"type": "integer", "minimum": 0, "maximum": 20},

            "overall_justification": {"type": "string"},
            "overall_score": {
                "type": "integer",
                "minimum": -165,
                "maximum": 120,
                "description": "Computed as positive_total - negative_total + bonus_total. Not clamped; can be negative.",
            },
            "overall_label": {
                "type": "string",
                "enum": ["very_poor", "poor", "fair", "good", "excellent"],
            },
        },
        "required": [
            "positive_prompt_fulfillment_premise_integration_justification",
            "positive_prompt_fulfillment_premise_integration_score",
            "positive_narrative_arc_pacing_justification",
            "positive_narrative_arc_pacing_score",
            "positive_character_depth_agency_justification",
            "positive_character_depth_agency_score",
            "positive_voice_stylistic_distinctiveness_justification",
            "positive_voice_stylistic_distinctiveness_score",
            "positive_concrete_world_scene_realization_justification",
            "positive_concrete_world_scene_realization_score",
            "positive_thematic_emotional_richness_subtlety_justification",
            "positive_thematic_emotional_richness_subtlety_score",

            "negative_prompt_violation_constraint_breach_justification",
            "negative_prompt_violation_constraint_breach_score",
            "negative_coherence_continuity_pov_confusion_justification",
            "negative_coherence_continuity_pov_confusion_score",
            "negative_generic_templated_language_structure_justification",
            "negative_generic_templated_language_structure_score",
            "negative_over_summary_abstraction_thin_grounding_justification",
            "negative_over_summary_abstraction_thin_grounding_score",
            "negative_over_explanation_redundancy_moralizing_justification",
            "negative_over_explanation_redundancy_moralizing_score",
            "negative_drift_bloat_structural_breakdown_justification",
            "negative_drift_bloat_structural_breakdown_score",
            "negative_dialogue_problems_stilted_expository_same_voice_justification",
            "negative_dialogue_problems_stilted_expository_same_voice_score",
            "negative_mechanical_formatting_errors_justification",
            "negative_mechanical_formatting_errors_score",

            "bonus_exceptional_unmodeled_merit_justification",
            "bonus_exceptional_unmodeled_merit_score",

            "positive_total",
            "negative_total",
            "bonus_total",
            "overall_justification",
            "overall_score",
            "overall_label",
        ],
        "additionalProperties": False,
    },
}

















































































































@dataclass(frozen=True)
class InferredRubric:
    ordered_keys: List[str]
    score_fields: List[str]          # integer fields
    justification_fields: List[str]  # string justification-ish fields
    total_fields: List[str]          # totals for 10-pt histogram buckets
    max_by_field: Dict[str, int]     # integer maximums
    overall_label_field: Optional[str]


def _infer_rubric_from_json_schema(json_schema: Dict[str, Any]) -> InferredRubric:
    sch = (json_schema or {}).get("schema", {}) or {}
    props: Dict[str, Any] = sch.get("properties", {}) or {}

    ordered_keys: List[str] = list(sch.get("required") or list(props.keys()))

    score_fields: List[str] = []
    justification_fields: List[str] = []
    max_by_field: Dict[str, int] = {}
    overall_label_field: Optional[str] = None

    for k in ordered_keys:
        spec = props.get(k, {}) or {}
        t = spec.get("type")

        if t == "integer":
            score_fields.append(k)
            mx = spec.get("maximum")
            if isinstance(mx, int):
                max_by_field[k] = mx

        elif t == "string":
            if k.endswith("_justification") or k == "overall_justification":
                justification_fields.append(k)

            if k == "overall_label" or (k.endswith("label") and isinstance(spec.get("enum"), list)):
                overall_label_field = k

    canonical_totals = ["positive_total", "negative_total", "bonus_total", "overall_score"]
    total_fields = [k for k in canonical_totals if k in score_fields]
    if not total_fields:
        total_fields = [k for k in score_fields if max_by_field.get(k) == 100]

    return InferredRubric(
        ordered_keys=ordered_keys,
        score_fields=score_fields,
        justification_fields=justification_fields,
        total_fields=total_fields,
        max_by_field=max_by_field,
        overall_label_field=overall_label_field,
    )


def _load_json_schema_from_run_config(run_config: Dict[str, Any]) -> Dict[str, Any]:
    tp = run_config.get("task_params") or {}

    if isinstance(tp.get("json_schema"), dict):
        return tp["json_schema"]

    schema_path = tp.get("json_schema_path")
    if schema_path:
        with open(schema_path, "r") as f:
            return json.load(f)

    return STORY_QUALITY_JSON_SCHEMA_DEFAULT

def _get_system_message(run_config: Dict[str, Any]) -> str:
    tp = run_config.get("task_params") or {}
    if tp.get("system_message_path"):
        with open(tp["system_message_path"], "r", encoding="utf-8") as _f:
            base = _f.read()
    else:
        base = tp.get("system_message") or DEFAULT_STORY_QUALITY_SYSTEM_MESSAGE

    force_mode = (tp.get("force_structured_mode") or "").strip().lower()

    engine_cfg = run_config.get("engine", {}) or {}
    provider = engine_cfg.get("provider", "")
    if force_mode == "native":
        use_native = True
    elif force_mode == "prompt":
        use_native = False
    else:
        use_native = supports_native_schema(provider, run_config)

    if use_native:
        return base

    schema = _load_json_schema_from_run_config(run_config)
    inner = (schema or {}).get("schema") or {}
    if not inner:
        return base

    schema_json = json.dumps(inner, indent=2, ensure_ascii=False)
    return (
        base
        + "\n\n"
        + "==================================================\n"
        + "FORMAL JSON SCHEMA FOR OUTPUT\n"
        + "==================================================\n"
        + "The model MUST return a single JSON object that conforms to this schema:\n"
        + schema_json
        + "\n"
        + "Return ONLY that JSON object and nothing else.\n"
    )



WORDCOUNT_REGEX = re.compile(r"(?P<word_count>\d+)\s+words\s+long", flags=re.IGNORECASE)
LENGTH_SENTENCE_REGEX = re.compile(r"words\s+long\b", flags=re.IGNORECASE)

QWEN_MARKERS_REGEX = re.compile(
    r"(?:<\|im_start\|>\w+)|(?:<\|im_end\|>)|(?:<im_end\|>)",
    flags=re.IGNORECASE,
)

def _strip_qwen_chat_markers(text: str) -> str:
    """
    Remove Qwen3 chat formatting markers (e.g., '<|im_start|>user', '<|im_end|>', '<im_end|>')
    from a prompt-like string.
    """
    if not text:
        return text
    cleaned = QWEN_MARKERS_REGEX.sub("", text)
    return cleaned.strip()


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


def _word_count(s: str) -> int:
    t = html.unescape(s or "")
    return len(re.findall(r"\w+", t))


def _compute_length_score(prompt: str, story: str) -> Optional[float]:
    if not prompt:
        return None

    m = WORDCOUNT_REGEX.search(prompt)
    if not m:
        return None

    try:
        target_w = int(m.group("word_count"))
    except Exception:
        return None

    if target_w <= 0:
        return None

    wc = _word_count(story)
    len_tolerance = max(50, int(0.1 * target_w))

    if abs(wc - target_w) <= len_tolerance:
        length_pen = 0.0
    elif wc < (target_w - len_tolerance):
        length_pen = min(1.0, 1.0 - (wc / max(1.0, float(target_w - len_tolerance))))
    else:
        length_pen = min(1.0, (wc / max(1.0, float(target_w + len_tolerance))) - 1.0)

    length_score = 1.0 - float(length_pen)
    return max(0.0, min(1.0, length_score))


def _slugify(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def _auto_title(field: str) -> str:
    """
    Heuristic title from field name. You can override via task_params.field_titles.
    """
    base = field
    if base.endswith("_score"):
        base = base[:-6]
    if base.endswith("_justification"):
        base = base[:-14]

    prefix = ""
    if base.startswith("positive_"):
        prefix = "Positive: "
        base = base[len("positive_") :]
    elif base.startswith("negative_"):
        prefix = "Negative: "
        base = base[len("negative_") :]

    base = base.replace("_", " ").strip()
    base = re.sub(r"\s+", " ", base)
    return prefix + base



class StoryQualityTask(Task):
    """
    Task for evaluating individual stories using an LLM judge.
    Designed to be "schema-driven": score fields, maxima, and extraction
    are inferred from the provided json_schema.

    Configure:
      task_params.system_message       (optional; defaults to DEFAULT_STORY_QUALITY_SYSTEM_MESSAGE)
      task_params.json_schema          (optional inline dict; defaults to STORY_QUALITY_JSON_SCHEMA_DEFAULT)
      task_params.json_schema_path     (optional path; used if json_schema not provided)
      task_params.field_titles         (optional dict[str,str] overrides for plotting/summary)
    """

    name = "story_quality"


    def load_examples(self, task_config: Dict[str, Any]) -> Sequence[Example]:
        path = task_config.get("input_path")
        if not path:
            raise ValueError("task_config.input_path must be set for StoryQualityTask.")

        story_field = task_config.get("story_field", "output")
        prompt_field = task_config.get("prompt_field", "input")
        model_field = task_config.get("model_field", "model")
        id_field = task_config.get("id_field", None)
        default_model_name = task_config.get("default_model_name", None)

        examples: List[Example] = []

        pattern = r"user\n(?P<story_prompt>.*)\nassistant\n<think>\n\n</think>"

        with open(path, "r") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)

                if story_field not in row:
                    continue

                story = row[story_field]
                prompt_raw = row.get(prompt_field, "")

                if model_field in row:
                    model = row[model_field]
                elif default_model_name is not None:
                    model = default_model_name
                else:
                    model = path

                ex_id = str(row[id_field]) if id_field and id_field in row else str(idx)

                try:
                    prompt_match = re.search(pattern, prompt_raw)
                    prompt = prompt_match.group("story_prompt") if prompt_match else prompt_raw
                except Exception:
                    prompt = prompt_raw

                prompt = _strip_qwen_chat_markers(prompt)

                examples.append(
                    Example(
                        id=ex_id,
                        data={
                            "story": story,
                            "prompt": prompt,
                            "model": model,
                            "raw": row,
                        },
                    )
                )

        return examples


    def _get_schema_and_inferred(self, run_config: Dict[str, Any]) -> Tuple[Dict[str, Any], InferredRubric]:
        schema = _load_json_schema_from_run_config(run_config)
        inferred = _infer_rubric_from_json_schema(schema)
        return schema, inferred

    def _field_titles(self, run_config: Dict[str, Any], inferred: InferredRubric) -> Dict[str, str]:
        tp = run_config.get("task_params") or {}
        overrides = tp.get("field_titles") or {}
        out: Dict[str, str] = {}
        for k in inferred.score_fields:
            out[k] = overrides.get(k) or _auto_title(k)
        return out


    def build_messages(self, ex: Example, run_config: Dict[str, Any]) -> List[Dict[str, str]]:
        d = ex.data
        story = d.get("story", "")
        prompt_raw = d.get("prompt", "") or ""
        prompt = _strip_length_requirement_sentence(prompt_raw)

        user_content = f"""STORY PROMPT:
{prompt}

STORY:
{story}
"""

        sys_msg = _get_system_message(run_config)
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ]


    def get_response_format(self, provider: str, run_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        schema, _ = self._get_schema_and_inferred(run_config)

        if provider == "openai" and supports_native_schema("openai", run_config):
            return {"type": "json_schema", "json_schema": schema}

        if provider == "vertex" and supports_native_schema("vertex", run_config):
            inner = (schema or {}).get("schema") or {}
            if not inner:
                return None
            return {
                "type": "vertex_json_schema",
                "response_mime_type": "application/json",
                "response_schema": inner,
            }

        return None

    def parse_response(
        self,
        ex: Example,
        provider_output: Any,
        run_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        d = ex.data
        schema, inferred = self._get_schema_and_inferred(run_config)

        raw_output = provider_output
        parsed: Optional[Dict[str, Any]] = None

        if isinstance(provider_output, dict):
            parsed = provider_output
        elif isinstance(provider_output, str):
            txt_raw = provider_output.strip()
            txt = _strip_code_fences(txt_raw)
            if txt.startswith("{") and txt.endswith("}"):
                try:
                    cand = json.loads(txt)
                    if isinstance(cand, dict):
                        parsed = cand
                        raw_output = cand
                except Exception:
                    pass

        rec: Dict[str, Any] = {
            "id": ex.id,
            "prompt": d.get("prompt", ""),
            "story": d.get("story", ""),
            "model": d.get("model", ""),
            "raw_output": raw_output,
            "schema_name": schema.get("name"),
        }

        if parsed is not None:
            for k in inferred.score_fields:
                v = parsed.get(k, None)
                try:
                    rec[k] = int(v) if v is not None else None
                except Exception:
                    rec[k] = None

            if inferred.overall_label_field:
                rec["overall_label"] = parsed.get(inferred.overall_label_field)
            else:
                rec["overall_label"] = parsed.get("overall_label")

            for k in inferred.justification_fields:
                v = parsed.get(k, None)
                rec[k] = str(v) if v is not None else ""

            tp = run_config.get("task_params") or {}
            if tp.get("override_overall_from_totals"):
                props = ((schema or {}).get("schema") or {}).get("properties") or {}
                if "positive_total" not in props or "negative_total" not in props:
                    raise ValueError(
                        "task_params.override_overall_from_totals is True, "
                        "but the JSON schema for storyquality does not define "
                        "'positive_total' and 'negative_total'."
                    )

                pos = rec.get("positive_total")
                neg = rec.get("negative_total")
                bonus = rec.get("bonus_total", 0)

                if isinstance(pos, int) and isinstance(neg, int):
                    if "overall_score" in rec and "overall_score_model" not in rec:
                        rec["overall_score_model"] = rec["overall_score"]

                    derived_unclipped = pos - neg + bonus
                    derived = min(100, derived_unclipped)

                    rec["derived_overall_unclipped"] = derived_unclipped
                    rec["derived_overall_clipped"] = derived

                    rec["overall_score"] = derived
                    rec["overall_score_source"] = "derived_from_positive_minus_negative_clamped_None_100"
                else:
                    rec.setdefault("overall_score_source", "model_or_missing")

        else:
            for k in inferred.score_fields:
                rec[k] = None
            rec["overall_label"] = None
            for k in inferred.justification_fields:
                rec[k] = ""

        try:
            pos = rec.get("positive_total")
            neg = rec.get("negative_total")
            bonus = rec.get("bonus_total", 0)
            ov = rec.get("overall_score")
            if isinstance(pos, int) and isinstance(neg, int):
                rec["derived_overall_unclipped"] = pos - neg + bonus
                rec["derived_overall_clipped"] = max(0, min(100, pos - neg + bonus))
            if isinstance(ov, int):
                rec["overall_score_in_range_0_100"] = (0 <= ov <= 100)
        except Exception:
            pass

        return rec


    def aggregate(
        self,
        records: List[Dict[str, Any]],
        run_config: Dict[str, Any],
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        if not records:
            return {"num_records": 0}

        _, inferred = self._get_schema_and_inferred(run_config)

        dim_fields = list(inferred.score_fields)
        total_fields = list(inferred.total_fields)
        max_by_field: Dict[str, int] = dict(inferred.max_by_field)

        for f in dim_fields:
            if f not in max_by_field:
                if f in ("positive_total", "overall_score"):
                    max_by_field[f] = 100
                elif f == "negative_total":
                    max_by_field[f] = 100
                else:
                    max_by_field[f] = 100

        field_titles = self._field_titles(run_config, inferred)

        def _build_histogram(values: List[int], max_value: int, bucket_size: int) -> Dict[str, Any]:
            if not values:
                return {"bins": [], "counts": [], "total": 0}

            clamped = [max(0, min(int(v), max_value)) for v in values]
            n_bins = max_value // bucket_size + 1
            counts = [0] * n_bins

            for v in clamped:
                idx = v // bucket_size
                if idx >= n_bins:
                    idx = n_bins - 1
                counts[idx] += 1

            bins: List[str] = []
            for i in range(n_bins):
                lo = i * bucket_size
                hi = min((i + 1) * bucket_size - 1, max_value)
                bins.append(f"{lo}" if bucket_size == 1 else f"{lo}-{hi}")

            return {"bins": bins, "counts": counts, "total": sum(counts)}


        dim_stats: Dict[str, Dict[str, Any]] = {}
        histograms_global: Dict[str, Dict[str, Any]] = {}

        for field in dim_fields:
            vals = [r.get(field) for r in records if isinstance(r.get(field), (int, float))]
            if not vals:
                dim_stats[field] = {"mean": None, "median": None, "min": None, "max": None, "count": 0}
                histograms_global[field] = {"bins": [], "counts": [], "total": 0}
                continue

            arr = np.array(vals, dtype=float)
            dim_stats[field] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": int(arr.size),
            }

            max_val = max_by_field.get(field, 100)
            bucket_size = 10 if field in total_fields else 1
            histograms_global[field] = _build_histogram([int(v) for v in vals], max_value=max_val, bucket_size=bucket_size)


        length_scores: List[float] = []
        per_record_len_score: Dict[str, float] = {}

        for r in records:
            prompt = r.get("prompt", "")
            story = r.get("story", "")
            ls = _compute_length_score(prompt, story)
            if ls is not None:
                length_scores.append(ls)
                per_record_len_score[str(r.get("id"))] = ls

        if length_scores:
            arr = np.array(length_scores, dtype=float)
            length_stats: Dict[str, Any] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": int(arr.size),
                "note": "length_score in [0,1], where 1.0 is near target word count; computed from prompt regex '(?P<word_count>\\d+) words long'.",
            }
        else:
            length_stats = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "count": 0,
                "note": "No valid length requirement found in prompts using regex '(?P<word_count>\\d+) words long'.",
            }


        def _extract_target_words(prompt: str) -> Optional[int]:
            if not prompt:
                return None
            m = WORDCOUNT_REGEX.search(prompt)
            if not m:
                return None
            try:
                wc = int(m.group("word_count"))
            except Exception:
                return None
            return wc if wc > 0 else None

        def _bucket_label(target_w: int) -> str:
            if target_w <= 1000:
                return "1k"
            if target_w <= 2000:
                return "2k"
            if target_w <= 3000:
                return "3k"
            if target_w <= 4000:
                return "4k"
            if target_w <= 5000:
                return "5k"
            if target_w <= 6000:
                return "6k"
            if target_w <= 7000:
                return "7k"
            if target_w <= 8000:
                return "8k"
            return ">8k"

        buckets: Dict[str, Dict[str, List[float]]] = {}
        buckets_len: Dict[str, List[float]] = {}

        for r in records:
            prompt = r.get("prompt", "")
            target_w = _extract_target_words(prompt)
            if target_w is None:
                continue
            bucket = _bucket_label(target_w)

            if bucket not in buckets:
                buckets[bucket] = {field: [] for field in dim_fields}
                buckets_len[bucket] = []

            for field in dim_fields:
                v = r.get(field)
                if isinstance(v, (int, float)):
                    buckets[bucket][field].append(float(v))

            rid = str(r.get("id"))
            if rid in per_record_len_score:
                buckets_len[bucket].append(per_record_len_score[rid])

        by_length_bucket: Dict[str, Any] = {}
        histograms_by_length_bucket: Dict[str, Any] = {}

        for bucket, field_vals in buckets.items():
            bucket_dim_stats: Dict[str, Dict[str, Any]] = {}
            for field, vals in field_vals.items():
                if not vals:
                    bucket_dim_stats[field] = {"mean": None, "median": None, "min": None, "max": None, "count": 0}
                else:
                    arr = np.array(vals, dtype=float)
                    bucket_dim_stats[field] = {
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "count": int(arr.size),
                    }

            ls_vals = buckets_len.get(bucket, [])
            if ls_vals:
                arr_ls = np.array(ls_vals, dtype=float)
                bucket_len_stats = {
                    "mean": float(np.mean(arr_ls)),
                    "median": float(np.median(arr_ls)),
                    "min": float(np.min(arr_ls)),
                    "max": float(np.max(arr_ls)),
                    "count": int(arr_ls.size),
                }
            else:
                bucket_len_stats = {"mean": None, "median": None, "min": None, "max": None, "count": 0}

            by_length_bucket[bucket] = {
                "dimensions": bucket_dim_stats,
                "length_score": bucket_len_stats,
                "target_words_range_note": "Bucket based on target word count extracted from prompt via '(?P<word_count>\\d+) words long'.",
            }

            bucket_hist_fields: Dict[str, Dict[str, Any]] = {}
            for field in total_fields:
                vals = field_vals.get(field, [])
                if not vals:
                    bucket_hist_fields[field] = {"bins": [], "counts": [], "total": 0}
                else:
                    hist = _build_histogram(
                        [int(v) for v in vals],
                        max_value=max_by_field.get(field, 100),
                        bucket_size=10,
                    )
                    bucket_hist_fields[field] = hist

            histograms_by_length_bucket[bucket] = bucket_hist_fields

        summary = {
            "num_records": len(records),
            "dimensions": dim_stats,
            "length_score": length_stats,
            "by_length_bucket": by_length_bucket,
            "histograms_global": histograms_global,
            "histograms_by_length_bucket": histograms_by_length_bucket,
            "field_titles": field_titles,
            "schema_inferred": {
                "score_fields": inferred.score_fields,
                "justification_fields": inferred.justification_fields,
                "total_fields": inferred.total_fields,
                "overall_label_field": inferred.overall_label_field,
            },
        }

        try:
            os.makedirs(output_dir, exist_ok=True)
            self._save_histogram_plots(summary, output_dir)
        except Exception:
            pass

        return summary

    def _save_histogram_plots(self, summary: Dict[str, Any], output_dir: str) -> None:
        histograms_global = summary.get("histograms_global", {}) or {}
        histograms_by_length_bucket = summary.get("histograms_by_length_bucket", {}) or {}
        field_titles = summary.get("field_titles", {}) or {}

        def _plot_hist(bins: List[str], counts: List[int], title: str, save_path: str) -> None:
            if not bins or not counts or sum(counts) == 0:
                return
            x = np.arange(len(bins))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x, counts, label="count", color="#4477aa")
            ax.set_xticks(x)
            ax.set_xticklabels(bins, rotation=45, ha="right")
            ax.set_ylabel("Count")
            ax.set_xlabel("Score bin")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)

        for field, hist in histograms_global.items():
            bins = hist.get("bins") or []
            counts = hist.get("counts") or []
            if not bins or not counts or sum(counts) == 0:
                continue
            base_title = field_titles.get(field, field)
            title = f"Global distribution: {base_title}"
            fname = f"hist_global_{_slugify(field)}.png"
            _plot_hist(bins, counts, title, os.path.join(output_dir, fname))

        for bucket, fields_hist in histograms_by_length_bucket.items():
            for field, hist in fields_hist.items():
                bins = hist.get("bins") or []
                counts = hist.get("counts") or []
                if not bins or not counts or sum(counts) == 0:
                    continue
                base_title = field_titles.get(field, field)
                title = f"{base_title} by target-length bucket {bucket}"
                fname = f"hist_lenbucket_{bucket}_{_slugify(field)}.png"
                _plot_hist(bins, counts, title, os.path.join(output_dir, fname))


    def format_input_preview(
        self,
        ex: Example,
        messages: List[Dict[str, str]],
        run_config: Dict[str, Any],
        max_len: int = 10000,
    ) -> str:
        d = ex.data
        prompt = d.get("prompt", "") or ""
        story = d.get("story", "") or ""
        model = d.get("model", "")

        lines: List[str] = []
        header = f"Example id: {ex.id}"
        if model:
            header += f" (model={model})"
        lines.append(header)

        lines.append("Prompt:")
        if prompt:
            for ln in str(prompt).splitlines() or [""]:
                lines.append(f"  {ln}")
        else:
            lines.append("  (empty prompt)")

        lines.append("Story:")
        if story:
            for ln in str(story).splitlines() or [""]:
                lines.append(f"  {ln}")
        else:
            lines.append("  (empty story)")

        out = "\n".join(lines)
        return out if len(out) <= max_len else out[: max_len - 1] + "…"

    def format_record_preview(
        self,
        record: Dict[str, Any],
        run_config: Dict[str, Any],
        max_len: int = 100000,
    ) -> str:
        _, inferred = self._get_schema_and_inferred(run_config)
        field_titles = self._field_titles(run_config, inferred)

        rid = record.get("id")
        model = record.get("model", "")
        schema_name = record.get("schema_name")

        lines: List[str] = []
        header = f"Record preview for id={rid}"
        if model:
            header += f" (model={model})"
        if schema_name:
            header += f" (schema={schema_name})"
        lines.append(header)

        if "overall_score" in record or "overall_label" in record:
            lines.append(f"Overall: {record.get('overall_score')} ({record.get('overall_label')})")
            lines.append("")

        printed_any = False
        for k in inferred.ordered_keys:
            if not k.endswith("_score"):
                continue
            title = field_titles.get(k, k)
            score = record.get(k)
            just_key = k[:-6] + "_justification"
            just = record.get(just_key, "")

            lines.append(f"- {title}: {score}")
            if just:
                for ln in str(just).splitlines() or [""]:
                    lines.append(f"    {ln}")
            else:
                lines.append("    (no justification)")
            lines.append("")
            printed_any = True

        for k in inferred.ordered_keys:
            if k in ("positive_total", "negative_total", "bonus_total", "overall_score") and not k.endswith("_score"):
                title = field_titles.get(k, _auto_title(k))
                lines.append(f"- {title}: {record.get(k)}")
                lines.append("")
                printed_any = True

        if "overall_justification" in record:
            lines.append("OVERALL JUSTIFICATION:")
            ovj = record.get("overall_justification") or ""
            if ovj:
                for ln in str(ovj).splitlines() or [""]:
                    lines.append(f"  {ln}")
            else:
                lines.append("  (no overall_justification)")

        if not printed_any:
            lines.append("(No recognized fields to preview; check schema inference.)")

        out = "\n".join(lines)
        return out if len(out) <= max_len else out[: max_len - 1] + "…"

    def format_summary(
        self,
        summary: Dict[str, Any],
        run_config: Dict[str, Any],
        max_len: int = 100000,
    ) -> str:
        lines: List[str] = []

        num_records = summary.get("num_records", 0) or 0
        task_cfg = run_config.get("task_config") or {}
        engine_cfg = run_config.get("engine") or {}

        input_path = task_cfg.get("input_path", "<unknown input_path>")
        judge_model = engine_cfg.get("model", "<unknown judge model>")
        story_model = task_cfg.get("default_model_name", "<varies-per-row>")

        lines.append(f"Total stories evaluated: {num_records}")
        lines.append(f"Source file: {input_path}")
        lines.append(f"Story model (source): {story_model}")
        lines.append(f"Judge model: {judge_model}")

        dim_stats = summary.get("dimensions", {}) or {}
        length_stats = summary.get("length_score", {}) or {}
        histograms_global = summary.get("histograms_global", {}) or {}
        by_bucket = summary.get("by_length_bucket", {}) or {}
        histograms_by_length_bucket = summary.get("histograms_by_length_bucket", {}) or {}
        field_titles = summary.get("field_titles", {}) or {}

        if dim_stats:
            lines.append("")
            lines.append("Global stats (mean / median / min / max / N):")
            header = f"{'metric':65s} {'mean':>7s} {'med':>7s} {'min':>7s} {'max':>7s} {'N':>6s}"
            lines.append(header)
            lines.append("-" * len(header))

            def _fmt(v):
                if v is None:
                    return "  n/a "
                return f"{v:7.3f}" if isinstance(v, float) else f"{v:7}"

            _, inferred = self._get_schema_and_inferred(run_config)
            ordered = [f for f in inferred.score_fields if f in dim_stats] + [f for f in dim_stats.keys() if f not in inferred.score_fields]

            for field in ordered:
                stats = dim_stats.get(field, {})
                mean = stats.get("mean")
                med = stats.get("median")
                mn = stats.get("min")
                mx = stats.get("max")
                cnt = stats.get("count", 0)
                label = f"{field_titles.get(field, field)} [{field}]"
                lines.append(
                    f"{label:65.65s} "
                    f"{_fmt(mean)} "
                    f"{_fmt(med)} "
                    f"{_fmt(mn)} "
                    f"{_fmt(mx)} "
                    f"{cnt:6d}"
                )

        if length_stats:
            lines.append("")
            lines.append("Global length_score details (1.0 = near requested word count):")
            mean = length_stats.get("mean")
            med = length_stats.get("median")
            mn = length_stats.get("min")
            mx = length_stats.get("max")
            cnt = length_stats.get("count", 0)
            note = length_stats.get("note", "")

            if all(isinstance(x, (int, float)) for x in [mean, med, mn, mx]):
                lines.append(f"  mean={mean:.3f}  median={med:.3f}  min={mn:.3f}  max={mx:.3f}  N={cnt}")
            else:
                lines.append(f"  N={cnt}, insufficient data for numeric summary.")
            if note:
                lines.append(f"  note: {note}")

        if histograms_global:
            lines.append("")
            lines.append("Global score distributions (histograms):")

            def _fmt_hist(hist: Dict[str, Any]) -> str:
                bins = hist.get("bins", []) or []
                counts = hist.get("counts", []) or []
                parts = [f"{b}:{c}" for b, c in zip(bins, counts) if c]
                return "  ".join(parts) if parts else "(no data)"

            _, inferred = self._get_schema_and_inferred(run_config)
            for field in inferred.score_fields:
                hist = histograms_global.get(field, {})
                if not hist:
                    continue
                title = field_titles.get(field, field)
                lines.append(f"  {title:45.45s} [{field:40s}] {_fmt_hist(hist)}")

        if by_bucket and num_records > 0:
            lines.append("")
            lines.append("Per target-length bucket (by requested word count in prompt):")
            lines.append("  Bucket  t_range    N_all  %all")
            lines.append("  ------  ---------  -----  ----")

            bucket_order = ["1k", "2k", "3k", "4k", "5k", "6k", "7k", "8k", ">8k"]

            def _pct(n: int, total: int) -> str:
                if total <= 0 or n <= 0:
                    return " 0.0"
                return f"{(100.0 * n / total):4.1f}"

            for b in bucket_order:
                info = by_bucket.get(b)
                if not info:
                    continue
                dims_b = info.get("dimensions", {}) or {}

                n = int((dims_b.get("overall_score") or dims_b.get("positive_total") or {}).get("count", 0))

                if b == "1k":
                    rng = "<=1000"
                elif b == ">8k":
                    rng = ">8000"
                else:
                    try:
                        upper = int(b.replace("k", "")) * 1000
                        lower = upper - 999
                        rng = f"{lower}-{upper}"
                    except Exception:
                        rng = "n/a"

                lines.append(f"  {b:6s}  {rng:9s}  {n:5d}  {_pct(n, num_records)}")

        if histograms_by_length_bucket:
            lines.append("")
            lines.append("Per target-length bucket histograms (10-point buckets for totals):")

            def _fmt_hist2(hist: Dict[str, Any]) -> str:
                bins = hist.get("bins", []) or []
                counts = hist.get("counts", []) or []
                parts = [f"{b}:{c}" for b, c in zip(bins, counts) if c]
                return "  ".join(parts) if parts else "(no data)"

            _, inferred = self._get_schema_and_inferred(run_config)
            totals = inferred.total_fields or ["positive_total", "negative_total", "bonus_total", "overall_score"]

            bucket_order = ["1k", "2k", "3k", "4k", "5k", "6k", "7k", "8k", ">8k"]
            for b in bucket_order:
                hinfo = histograms_by_length_bucket.get(b)
                if not hinfo:
                    continue
                lines.append(f"  Bucket {b}:")
                for field in totals:
                    hist = hinfo.get(field, {})
                    if not hist:
                        continue
                    title = field_titles.get(field, field)
                    lines.append(f"    {title:22.22s} [{field:15s}] {_fmt_hist2(hist)}")

        out = "\n".join(lines)
        return out if len(out) <= max_len else out[: max_len - 1] + "…"










































































































































































































































































































































































































































































































































































    




















































































































































    





















