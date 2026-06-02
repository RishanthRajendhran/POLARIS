# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _apply_chat_template_text(tokenizer, messages, apply_chat_kwargs, **extra_kwargs):
    """Render messages via apply_chat_template and return raw text (not tokenized)."""
    merged = dict(apply_chat_kwargs)
    merged.update(extra_kwargs)
    return tokenizer.apply_chat_template(messages, tokenize=False, **merged)


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Extract system_message from apply_chat_template_kwargs (not passed to tokenizer).
        # OmegaConf DictConfigs in struct mode don't support .pop(), so we read
        # the value first, then delete if present using OmegaConf flag override.
        self._system_message = None
        if hasattr(self, 'apply_chat_template_kwargs') and self.apply_chat_template_kwargs:
            from omegaconf import OmegaConf, flag_override
            sm = OmegaConf.select(self.apply_chat_template_kwargs, "system_message", default=None)
            if sm is not None:
                self._system_message = sm
                with flag_override(self.apply_chat_template_kwargs, "struct", False):
                    del self.apply_chat_template_kwargs["system_message"]

    def _build_think_prefix_ids(self, messages: list[dict], gt_reasoning: str) -> list[int]:
        """Build token IDs for <think>...reasoning...</think>\\n that follow the
        prompt.  Used for GT-think-prefix rollouts: teacher-force reasoning,
        then let the model sample the story freely.

        Returns only the think-prefix token IDs (NOT the prompt tokens).
        """
        apply_kw = dict(self.apply_chat_template_kwargs)
        think_text = f"<think>\n{gt_reasoning}\n</think>\n"

        # Render prompt + partial assistant (think block only)
        partial_msgs = list(messages) + [{"role": "assistant", "content": think_text}]
        partial_text = _apply_chat_template_text(
            self.tokenizer, partial_msgs, apply_kw, add_generation_prompt=False,
        )
        partial_ids = self.tokenizer(partial_text, add_special_tokens=False)["input_ids"]

        # Render prompt with generation prompt (same kwargs as normal tokenization)
        prompt_text = _apply_chat_template_text(
            self.tokenizer, messages, apply_kw, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        if partial_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Chat template mismatch: prompt_ids not a prefix of partial_ids (think-prefix). "
                "Check apply_chat_template kwargs (enable_thinking, roles, etc.)."
            )

        think_ids = partial_ids[len(prompt_ids):]
        if len(think_ids) == 0:
            raise ValueError("Think-prefix tokenized to zero tokens; check reasoning text and template.")

        # Strip trailing <|im_end|>\n — the assistant turn is intentionally left
        # open so the model continues generating (story) after </think>.
        im_end_str = "<|im_end|>"
        im_end_id = self.tokenizer.convert_tokens_to_ids(im_end_str)
        if im_end_id != self.tokenizer.unk_token_id:  # tokenizer knows this token
            suffix = self.tokenizer(im_end_str + "\n", add_special_tokens=False)["input_ids"]
            if list(think_ids[-len(suffix):]) == suffix:
                think_ids = think_ids[: -len(suffix)]
            elif think_ids and think_ids[-1] == im_end_id:
                think_ids = think_ids[:-1]

        return [int(x) for x in think_ids]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        gt_think_prefix_reasoning = kwargs.get("gt_think_prefix_reasoning", None)

        # System message injection
        if self._system_message and (not messages or messages[0].get("role") != "system"):
            messages.insert(0, {"role": "system", "content": self._system_message})

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. build think-prefix if GT reasoning is provided
        think_prefix_ids = []
        think_prefix_len = 0
        if gt_think_prefix_reasoning is not None and gt_think_prefix_reasoning:
            think_prefix_ids = self._build_think_prefix_ids(messages, gt_think_prefix_reasoning)
            think_prefix_len = len(think_prefix_ids)

        # 4. generate sequences (from extended prompt if think-prefix)
        gen_prompt_ids = prompt_ids + think_prefix_ids if think_prefix_ids else prompt_ids
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            gen_output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=gen_prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = gen_output.num_preempted if gen_output.num_preempted is not None else -1

        # 5. assemble response: think_prefix + generated story
        #    response_mask: 0 for think-prefix (teacher-forced), 1 for generated
        remaining_len = self.response_length - think_prefix_len
        gen_token_ids = gen_output.token_ids[:remaining_len]
        gen_log_probs = gen_output.log_probs[:remaining_len] if gen_output.log_probs else None

        response_ids = think_prefix_ids + gen_token_ids
        response_mask = [0] * think_prefix_len + [1] * len(gen_token_ids)
        response_logprobs = None
        if gen_log_probs is not None:
            response_logprobs = [0.0] * think_prefix_len + gen_log_probs

        # Handle routed_experts: think-prefix positions need zero padding
        routed_experts = None
        if gen_output.routed_experts is not None:
            import numpy as np
            gen_experts = gen_output.routed_experts[:remaining_len]
            if think_prefix_len > 0:
                # Pad think-prefix positions with zeros
                layer_num = gen_experts.shape[1] if gen_experts.ndim >= 2 else 1
                topk_num = gen_experts.shape[2] if gen_experts.ndim >= 3 else 1
                prefix_experts = np.zeros((think_prefix_len, layer_num, topk_num), dtype=gen_experts.dtype)
                routed_experts = np.concatenate([prefix_experts, gen_experts], axis=0)
            else:
                routed_experts = gen_experts
            routed_experts = routed_experts[: len(prompt_ids) + self.response_length]
        elif gen_output.routed_experts is not None:
            routed_experts = gen_output.routed_experts[: len(prompt_ids) + self.response_length]

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,  # original prompt (not extended)
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=gen_output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({
            "turn_scores": [],
            "tool_rewards": [],
            "think_prefix_len": think_prefix_len,
        })

        return output
