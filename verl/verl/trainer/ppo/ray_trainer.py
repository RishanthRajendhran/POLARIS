# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _compute_hri_group_metrics(
    reward_tensor: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    group_ids: np.ndarray,
    is_gt_arr: np.ndarray,
) -> dict[str, float]:
    """Compute HRI-vs-policy diagnostics from the current training batch.

    Metrics are intentionally group-based so we can explain whether the GT anchor
    is acting as a stable upper bound and whether it changes the advantage signal.
    """

    metrics: dict[str, float] = {}
    if reward_tensor.ndim != 2 or response_mask.ndim != 2 or advantages.ndim != 2:
        return metrics
    if len(group_ids) != reward_tensor.shape[0] or len(is_gt_arr) != reward_tensor.shape[0]:
        return metrics

    seq_mask = response_mask.to(torch.float32)
    seq_len = seq_mask.sum(-1).clamp_min(1.0)
    seq_reward = ((reward_tensor * seq_mask).sum(-1) / seq_len).detach().cpu().numpy()
    seq_adv = ((advantages.to(torch.float32) * seq_mask).sum(-1) / seq_len).detach().cpu().numpy()

    gt_rewards = seq_reward[is_gt_arr]
    pol_rewards = seq_reward[~is_gt_arr]
    gt_advs = seq_adv[is_gt_arr]
    pol_advs = seq_adv[~is_gt_arr]

    if gt_rewards.size > 0:
        metrics["hri/gt_reward_mean"] = float(np.nanmean(gt_rewards))
        metrics["hri/gt_adv_mean"] = float(np.nanmean(gt_advs))
    if pol_rewards.size > 0:
        metrics["hri/policy_reward_mean"] = float(np.nanmean(pol_rewards))
        metrics["hri/policy_reward_std"] = float(np.nanstd(pol_rewards))
        metrics["hri/policy_adv_mean"] = float(np.nanmean(pol_advs))
        metrics["hri/policy_adv_std"] = float(np.nanstd(pol_advs))
        metrics["hri/policy_positive_adv_frac"] = float(np.nanmean((pol_advs > 0).astype(np.float32)))

    gt_minus_policy_mean = []
    gt_minus_policy_best = []
    gt_minus_policy_worst = []
    policy_reward_std_by_group = []
    policy_adv_std_by_group = []
    gt_beats_all = []
    group_sizes = []
    valid_groups = 0

    for gid in np.unique(group_ids):
        gmask = group_ids == gid
        g_gt = is_gt_arr[gmask]
        if g_gt.sum() != 1 or (~g_gt).sum() == 0:
            continue

        rewards_g = seq_reward[gmask]
        advs_g = seq_adv[gmask]
        gt_reward = float(rewards_g[g_gt][0])
        policy_rewards = rewards_g[~g_gt]
        gt_adv = float(advs_g[g_gt][0])
        policy_advs = advs_g[~g_gt]

        gt_minus_policy_mean.append(gt_reward - float(np.mean(policy_rewards)))
        gt_minus_policy_best.append(gt_reward - float(np.max(policy_rewards)))
        gt_minus_policy_worst.append(gt_reward - float(np.min(policy_rewards)))
        policy_reward_std_by_group.append(float(np.std(policy_rewards)))
        policy_adv_std_by_group.append(float(np.std(policy_advs)))
        gt_beats_all.append(float(gt_reward > float(np.max(policy_rewards))))
        group_sizes.append(int(policy_rewards.shape[0] + 1))
        valid_groups += 1

    if valid_groups > 0:
        metrics["hri/group_count"] = float(valid_groups)
        metrics["hri/group_size_mean"] = float(np.mean(group_sizes))
        metrics["hri/gt_minus_policy_mean"] = float(np.mean(gt_minus_policy_mean))
        metrics["hri/gt_minus_policy_best"] = float(np.mean(gt_minus_policy_best))
        metrics["hri/gt_minus_policy_worst"] = float(np.mean(gt_minus_policy_worst))
        metrics["hri/policy_reward_std_group_mean"] = float(np.mean(policy_reward_std_by_group))
        metrics["hri/policy_adv_std_group_mean"] = float(np.mean(policy_adv_std_by_group))
        metrics["hri/frac_groups_gt_beats_all_policy"] = float(np.mean(gt_beats_all))

    return metrics


def _masked_stats_1d(arr: np.ndarray, prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if arr.size == 0:
        return metrics
    metrics[f"{prefix}/mean"] = float(np.nanmean(arr))
    metrics[f"{prefix}/std"] = float(np.nanstd(arr))
    metrics[f"{prefix}/max"] = float(np.nanmax(arr))
    metrics[f"{prefix}/min"] = float(np.nanmin(arr))
    return metrics


def _compute_hri_split_batch_metrics(batch: DataProto, is_gt_arr: np.ndarray, use_critic: bool) -> dict[str, float]:
    """Compute GT-vs-policy splits for reward/advantage/return/value statistics."""

    metrics: dict[str, float] = {}
    if "response_mask" not in batch.batch:
        return metrics

    response_mask = batch.batch["response_mask"].bool()
    response_len = response_mask.sum(-1).detach().cpu().numpy().astype(np.float64)
    aborted_mask = response_len == 0

    seq_score = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy().astype(np.float64)
    seq_reward = batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy().astype(np.float64)

    advantages = batch.batch["advantages"].detach()
    returns = batch.batch["returns"].detach()
    values = batch.batch["values"].detach() if use_critic and "values" in batch.batch else None

    for split_name, split_mask in (("gt", is_gt_arr), ("policy", ~is_gt_arr)):
        if split_mask.shape[0] != response_mask.shape[0] or not split_mask.any():
            continue

        non_aborted_split = split_mask & (~aborted_mask)
        if non_aborted_split.any():
            metrics.update(_masked_stats_1d(seq_score[non_aborted_split], f"hri_split/{split_name}_score"))
            metrics.update(_masked_stats_1d(seq_reward[non_aborted_split], f"hri_split/{split_name}_reward"))

        split_mask_t = torch.as_tensor(split_mask, dtype=torch.bool, device=response_mask.device)
        token_mask = response_mask & split_mask_t.unsqueeze(-1)

        valid_adv = torch.masked_select(advantages, token_mask)
        valid_returns = torch.masked_select(returns, token_mask)
        if valid_adv.numel() > 0:
            adv_np = valid_adv.detach().cpu().numpy().astype(np.float64)
            metrics.update(_masked_stats_1d(adv_np, f"hri_split/{split_name}_advantage"))
            metrics[f"hri_split/{split_name}_positive_adv_frac"] = float((adv_np > 0).mean())
        if valid_returns.numel() > 0:
            metrics.update(_masked_stats_1d(valid_returns.detach().cpu().numpy().astype(np.float64), f"hri_split/{split_name}_return"))
        if values is not None:
            valid_values = torch.masked_select(values, token_mask)
            if valid_values.numel() > 0:
                val_np = valid_values.detach().cpu().numpy().astype(np.float64)
                metrics.update(_masked_stats_1d(val_np, f"hri_split/{split_name}_value"))
                td_err = (valid_returns - valid_values).abs().detach().cpu().numpy().astype(np.float64)
                metrics.update(_masked_stats_1d(td_err, f"hri_split/{split_name}_value_error_abs"))

        metrics.update(_masked_stats_1d(response_len[split_mask], f"hri_split/{split_name}_response_length"))
        metrics[f"hri_split/{split_name}_aborted_frac"] = float(aborted_mask[split_mask].mean())
        metrics[f"hri_split/{split_name}_count"] = float(int(split_mask.sum()))

    return metrics


def _compute_hri_reward_component_metrics(
    reward_extra_infos_dict: dict[str, Any], is_gt_arr: np.ndarray
) -> dict[str, float]:
    """Split reward component diagnostics into GT and policy aggregates."""

    metrics: dict[str, float] = {}
    for comp_name, comp_vals in reward_extra_infos_dict.items():
        if isinstance(comp_vals, (list, np.ndarray)):
            arr = np.array(comp_vals, dtype=np.float64)
            if arr.size > 0 and arr.shape[0] == is_gt_arr.shape[0]:
                gt_arr = arr[is_gt_arr]
                pol_arr = arr[~is_gt_arr]
                if gt_arr.size > 0:
                    metrics[f"reward_comp_gt/{comp_name}_mean"] = float(np.nanmean(gt_arr))
                if pol_arr.size > 0:
                    metrics[f"reward_comp_policy/{comp_name}_mean"] = float(np.nanmean(pol_arr))

    for diag_key, diag_list in reward_extra_infos_dict.items():
        if not (isinstance(diag_list, list) and diag_key.endswith("_diag")):
            continue
        comp_name = diag_key[:-5]
        per_field_vals_gt: dict[str, list[float]] = {}
        per_field_vals_policy: dict[str, list[float]] = {}
        for idx_d, diag in enumerate(diag_list):
            if not isinstance(diag, dict):
                continue
            raw = diag.get("raw", diag)
            if not isinstance(raw, dict):
                continue
            target = per_field_vals_gt if bool(is_gt_arr[idx_d]) else per_field_vals_policy
            for field_name, field_val in raw.items():
                if isinstance(field_val, (int, float, np.integer, np.floating)):
                    target.setdefault(field_name, []).append(float(field_val))
        for field_name, vals in per_field_vals_gt.items():
            if vals:
                metrics[f"{comp_name}_gt/{field_name}_mean"] = float(np.nanmean(np.array(vals, dtype=np.float64)))
        for field_name, vals in per_field_vals_policy.items():
            if vals:
                metrics[f"{comp_name}_policy/{field_name}_mean"] = float(np.nanmean(np.array(vals, dtype=np.float64)))

    return metrics


def _append_hri_sequence_log(
    log_path: str,
    global_step: int,
    batch: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_infos_dict: dict[str, Any],
    group_ids: np.ndarray,
    is_gt_arr: np.ndarray,
):
    """Append per-sequence HRI stats so we can inspect raw trajectory-level signals later."""

    response_mask = batch.batch["response_mask"].to(torch.float32)
    seq_len = response_mask.sum(-1).clamp_min(1.0)
    seq_score = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy().astype(np.float64)
    seq_reward = ((reward_tensor * response_mask).sum(-1) / seq_len).detach().cpu().numpy().astype(np.float64)
    seq_adv = ((batch.batch["advantages"].to(torch.float32) * response_mask).sum(-1) / seq_len).detach().cpu().numpy().astype(np.float64)
    seq_return = ((batch.batch["returns"].to(torch.float32) * response_mask).sum(-1) / seq_len).detach().cpu().numpy().astype(np.float64)
    seq_value = None
    if "values" in batch.batch:
        seq_value = ((batch.batch["values"].to(torch.float32) * response_mask).sum(-1) / seq_len).detach().cpu().numpy().astype(np.float64)

    uids = batch.non_tensor_batch.get("uid", np.arange(len(seq_reward)))
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for idx in range(len(seq_reward)):
            row = {
                "global_step": int(global_step),
                "uid": str(uids[idx]),
                "group_id": int(group_ids[idx]),
                "is_gt": bool(is_gt_arr[idx]),
                "response_length": float(seq_len[idx].detach().cpu().item()),
                "seq_score": float(seq_score[idx]),
                "seq_reward": float(seq_reward[idx]),
                "seq_advantage": float(seq_adv[idx]),
                "seq_return": float(seq_return[idx]),
            }
            if seq_value is not None:
                row["seq_value"] = float(seq_value[idx])
                row["seq_value_error_abs"] = float(abs(seq_return[idx] - seq_value[idx]))

            for comp_name, comp_vals in reward_extra_infos_dict.items():
                if isinstance(comp_vals, (list, np.ndarray)) and len(comp_vals) == len(seq_reward):
                    val = comp_vals[idx]
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        row[f"reward_comp/{comp_name}"] = float(val)

            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        is_gt_arr = data.non_tensor_batch.get("is_gt", None)
        if is_gt_arr is not None:
            is_gt_arr = np.asarray(is_gt_arr, dtype=bool)
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
            is_gt=is_gt_arr,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # GT think-prefix: teacher-force reasoning, let model sample story freely
        self.gt_think_prefix = self.config.algorithm.get("gt_think_prefix", False)
        self.gt_think_prefix_count = self.config.algorithm.get("gt_think_prefix_count", 1)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.aw_scheduler = None

        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    # =====================================================================
    # SUPPORT FOR GROUND TRUTH ADDITION TO GROUP
    # =====================================================================
    def _extract_gt_text(self, ntb: dict, i: int) -> str:
        """Extract ground-truth text for row i from non_tensor_batch."""
        for k in ("ground_truth", "solution", "answer", "target"):
            if k in ntb:
                return ntb[k][i]
        if "reward_model" in ntb:
            rm = ntb["reward_model"][i]
            if isinstance(rm, dict):
                for k in ("ground_truth", "solution", "answer", "target"):
                    if k in rm:
                        return rm[k]
        raise KeyError(
            "No ground-truth text found. Provide one of non_tensor_batch keys: "
            "ground_truth / solution / answer / target, or reward_model[...]['ground_truth']."
        )

    def _extract_gt_reasoning(self, ntb: dict, i: int) -> str:
        """
        Strict extractor for ground-truth reasoning text when thinking is enabled.
        Looks for 'ground_truth_reasoning' at top-level, else nested under 'reward_model'[i].
        Raises ValueError if not found or if it contains <think> tags (trainer wraps).
        """
        if "ground_truth_reasoning" in ntb:
            val = ntb["ground_truth_reasoning"][i]
            if not isinstance(val, str):
                raise ValueError("ground_truth_reasoning must be a string per row.")
            if ("<think>" in val) or ("</think>" in val):
                raise ValueError(
                    "Do not include <think> or </think> in ground_truth_reasoning; the trainer wraps GT reasoning."
                )
            return val

        if "reward_model" in ntb:
            rm = ntb["reward_model"][i]
            if isinstance(rm, dict) and ("ground_truth_reasoning" in rm):
                val = rm["ground_truth_reasoning"]
                if not isinstance(val, str):
                    raise ValueError("reward_model['ground_truth_reasoning'] must be a string.")
                if ("<think>" in val) or ("</think>" in val):
                    raise ValueError(
                        "Do not include <think> or </think> in reward_model['ground_truth_reasoning']; the trainer wraps GT reasoning."
                    )
                return val

        raise ValueError(
            "Thinking is enabled but 'ground_truth_reasoning' is missing for GT row. "
            "Expected non_tensor_batch['ground_truth_reasoning'][i] or "
            "non_tensor_batch['reward_model'][i]['ground_truth_reasoning']."
        )

    def _build_gt_rollouts(self, batch: DataProto) -> DataProto:
        """
        Build one GT rollout per unique uid group (first occurrence), with:
        - non_tensor_batch["is_gt"] as a clean np.bool_ 1D array (never object/arrays)
        - all other non-tensor fields copied row-wise, without creating object-shaped nests
        """
        from tensordict import TensorDict

        # ---- Required tensors ----
        if "prompts" in batch.batch:
            prompts = batch.batch["prompts"]  # [B, P]
        else:
            assert "input_ids" in batch.batch and "responses" in batch.batch, "Need prompts or (input_ids,responses)"
            R = batch.batch["responses"].shape[1]
            P = batch.batch["input_ids"].shape[1] - R
            prompts = batch.batch["input_ids"][:, :P]

        responses = batch.batch["responses"]              # [B, R]
        attention_mask = batch.batch["attention_mask"]    # [B, P+R]

        B, P = prompts.shape
        R = responses.shape[1]
        pad_id = int(self.tokenizer.pad_token_id) if (self.tokenizer.pad_token_id is not None) else 0

        # ---- Unique grouping ----
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is None:
            raise KeyError("batch.non_tensor_batch['uid'] is required to build GT rollouts.")
        if isinstance(uids, list):
            uids = np.asarray(uids, dtype=object)
        if not isinstance(uids, np.ndarray) or len(uids) != B:
            raise ValueError("uid must be a numpy array of length B.")

        _, first_idx = np.unique(uids, return_index=True)
        first_idx = sorted(first_idx.tolist())

        thinking_enabled = bool(
            getattr(self.config.data, "apply_chat_template_kwargs", {}).get("enable_thinking", False)
            if hasattr(self.config, "data")
            else False
        )

        def _row_value(v, i: int):
            if isinstance(v, np.ndarray):
                return v[i]
            if isinstance(v, list):
                return v[i]
            return v

        # ---- Accumulators ----
        gt_prompt_list: list[torch.Tensor] = []
        gt_resp_list: list[torch.Tensor] = []
        gt_attn_list: list[torch.Tensor] = []
        gt_ntb_lists: dict[str, list] = {k: [] for k in batch.non_tensor_batch.keys() if k != "is_gt"}
        gt_is_gt_list: list[bool] = []

        # ---- Build GT rows ----
        for i in first_idx:
            p_ids = prompts[i]

            gt_text = self._extract_gt_text(batch.non_tensor_batch, i)
            if not isinstance(gt_text, str):
                raise ValueError("Ground-truth text must be a string per row.")
            if thinking_enabled and (("<think>" in gt_text) or ("</think>" in gt_text)):
                raise ValueError(
                    "Do not include <think> or </think> in ground_truth/solution/answer/target; trainer wraps."
                )

            gt_reason = None
            if thinking_enabled:
                gt_reason = self._extract_gt_reasoning(batch.non_tensor_batch, i)
                response_text = (
                    "<think>\n" + gt_reason + "\n</think>\n" + gt_text + (self.tokenizer.eos_token or "")
                )
            else:
                response_text = gt_text + (self.tokenizer.eos_token or "")

            gt_ids_full = self.tokenizer.encode(response_text, add_special_tokens=False)

            r_len = min(len(gt_ids_full), R)
            r_ids = torch.full((R,), pad_id, dtype=torch.long, device=p_ids.device)
            if r_len > 0:
                r_ids[:r_len] = torch.tensor(gt_ids_full[:r_len], dtype=torch.long, device=p_ids.device)

            if thinking_enabled and gt_reason is not None:
                pre_story_text = "<think>\n" + gt_reason + "\n</think>\n"
                thought_ids = self.tokenizer.encode(pre_story_text, add_special_tokens=False)
                if r_len <= len(thought_ids):
                    print(f"[GT] Warning: only thought tokens fit into response (R={R}, uid={uids[i]}).")

            attn_prompt = attention_mask[i, :P]
            attn_resp = torch.zeros((R,), dtype=attention_mask.dtype, device=p_ids.device)
            if r_len > 0:
                attn_resp[:r_len] = 1
            attn_full = torch.cat([attn_prompt, attn_resp], dim=0)

            gt_prompt_list.append(p_ids.unsqueeze(0))
            gt_resp_list.append(r_ids.unsqueeze(0))
            gt_attn_list.append(attn_full.unsqueeze(0))

            for k, v in batch.non_tensor_batch.items():
                if k == "is_gt":
                    continue
                gt_ntb_lists[k].append(_row_value(v, i))

            gt_is_gt_list.append(True)

        # ---- Stack tensors ----
        gt_prompts = torch.cat(gt_prompt_list, dim=0)
        gt_responses = torch.cat(gt_resp_list, dim=0)
        gt_attention = torch.cat(gt_attn_list, dim=0)

        input_ids = torch.cat([gt_prompts, gt_responses], dim=1)
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.size(0), -1)

        gt_td = TensorDict(
            {
                "prompts": gt_prompts,
                "responses": gt_responses,
                "attention_mask": gt_attention,
                "input_ids": input_ids,
                "position_ids": position_ids,
            },
            batch_size=[gt_prompts.size(0)],
            device=gt_prompts.device,
        )

        # ---- Finalize non-tensor batch ----
        out_ntb: dict[str, np.ndarray] = {}
        out_ntb["is_gt"] = np.asarray(gt_is_gt_list, dtype=np.bool_)

        for k, lst in gt_ntb_lists.items():
            orig = batch.non_tensor_batch.get(k, None)
            if isinstance(orig, np.ndarray):
                try:
                    out_ntb[k] = np.asarray(lst, dtype=orig.dtype)
                except Exception:
                    out_ntb[k] = np.asarray(lst, dtype=object)
            else:
                out_ntb[k] = np.asarray(lst, dtype=object)

        return DataProto(batch=gt_td, non_tensor_batch=out_ntb, meta_info=batch.meta_info)

    def _align_tensordict_keys_for_concat(self, a: DataProto, b: DataProto, keys: set[str] | None = None) -> None:
        """
        Make a.batch and b.batch have identical keys by padding missing tensor keys with zeros.
        Only aligns keys in `keys` if provided; otherwise aligns the union of keys.
        """
        a_keys = set(a.batch.keys())
        b_keys = set(b.batch.keys())
        use_keys = keys if keys is not None else (a_keys | b_keys)

        Ba = a.batch["responses"].shape[0]
        Bb = b.batch["responses"].shape[0]

        for k in use_keys:
            in_a = k in a.batch
            in_b = k in b.batch
            if in_a and in_b:
                a_tail = tuple(a.batch[k].shape[1:])
                b_tail = tuple(b.batch[k].shape[1:])
                if a_tail != b_tail:
                    # Try to reshape the lower-dim tensor to match the higher-dim one
                    # e.g. position_ids: policy (24, 4, 4352) vs GT (8, 4352)
                    # -> expand GT to (8, 4, 4352) by unsqueezing + expanding
                    if len(a_tail) > len(b_tail) and a_tail[-len(b_tail):] == b_tail:
                        # b is missing leading dims that a has
                        extra_dims = a_tail[:len(a_tail) - len(b_tail)]
                        reshaped = b.batch[k]
                        for d in reversed(extra_dims):
                            reshaped = reshaped.unsqueeze(1).expand(-1, d, *reshaped.shape[1:]).contiguous()
                        b.batch[k] = reshaped
                    elif len(b_tail) > len(a_tail) and b_tail[-len(a_tail):] == a_tail:
                        extra_dims = b_tail[:len(b_tail) - len(a_tail)]
                        reshaped = a.batch[k]
                        for d in reversed(extra_dims):
                            reshaped = reshaped.unsqueeze(1).expand(-1, d, *reshaped.shape[1:]).contiguous()
                        a.batch[k] = reshaped
                    else:
                        # Incompatible shapes — drop key from both to avoid crash
                        print(f"[_align_tensordict] WARNING: dropping key '{k}' due to shape mismatch: "
                              f"a{tuple(a.batch[k].shape)} vs b{tuple(b.batch[k].shape)}", flush=True)
                        del a.batch[k]
                        del b.batch[k]
                continue

            if in_a and not in_b:
                tmpl = a.batch[k]
                b.batch[k] = tmpl.new_zeros((Bb,) + tmpl.shape[1:])
            elif in_b and not in_a:
                tmpl = b.batch[k]
                a.batch[k] = tmpl.new_zeros((Ba,) + tmpl.shape[1:])

    def _rollout_log_probs_expected(self) -> bool:
        """Whether the user asked the rollout engine to attach rollout_log_probs."""
        try:
            return bool(getattr(self.config.actor_rollout_ref.rollout, "calculate_log_probs", False))
        except Exception:
            return False

    def _rollout_correction_active(self) -> bool:
        """Whether rollout-correction is intended to be used (not bypassed).
        Returns False if rollout_is is null (no IS weights) even if bypass_mode is False."""
        try:
            rc = getattr(self.config.algorithm, "rollout_correction", None)
            if rc is None:
                return False
            # If rollout_is is null/None, correction is effectively disabled
            if rc.get("rollout_is", None) is None:
                return False
            return not bool(rc.get("bypass_mode", True))
        except Exception:
            return False
    # =====================================================================
    # END SUPPORT FOR GROUND TRUTH ADDITION TO GROUP
    # =====================================================================

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # Batch reward path: runs when streaming is disabled or use_rm is True
                if self.use_rm:
                    # for colocate reward models, sleep rollout model to spare GPU memory
                    self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                if self.use_rm:
                    # wake up rollout model
                    self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Validation reward distributions (hist + quantiles)
        if bool(getattr(self.config.trainer, "log_val_reward_distributions", False)):
            bins = int(getattr(self.config.trainer, "val_reward_hist_bins", 50))
            max_pts = int(getattr(self.config.trainer, "val_reward_hist_max_points", 20000))
            allow = getattr(self.config.trainer, "val_reward_dist_keys", None)

            try:
                import wandb
                _HAS_WANDB = True
            except Exception:
                wandb = None
                _HAS_WANDB = False

            exclude_prefixes = ("storyq_",)
            exclude_suffixes = ("_diag", "_debug")
            exclude_exact = {"weights_used", "pos_gate", "story_quality_diag", "human_like_diag"}

            def _to_float_array(xlist):
                if not isinstance(xlist, list) or len(xlist) == 0:
                    return None
                out = []
                for x in xlist:
                    try:
                        xf = float(x)
                    except Exception:
                        continue
                    if np.isfinite(xf):
                        out.append(xf)
                if not out:
                    return None
                return np.asarray(out, dtype=np.float32)

            if allow:
                keys = [k for k in allow if k in reward_extra_infos_dict]
            else:
                keys = []
                for k, v in reward_extra_infos_dict.items():
                    if k in exclude_exact:
                        continue
                    if any(k.startswith(p) for p in exclude_prefixes):
                        continue
                    if any(k.endswith(sfx) for sfx in exclude_suffixes):
                        continue
                    if not isinstance(v, list):
                        continue
                    if len(v) != len(reward_extra_infos_dict.get("reward", [])):
                        continue
                    arr = _to_float_array(v)
                    if arr is None:
                        continue
                    keys.append(k)

            rng = np.random.RandomState(0)
            for k in keys:
                arr = _to_float_array(reward_extra_infos_dict.get(k, []))
                if arr is None or arr.size == 0:
                    continue

                metric_dict[f"val-dist/{k}/n"] = int(arr.size)
                metric_dict[f"val-dist/{k}/mean"] = float(arr.mean())
                metric_dict[f"val-dist/{k}/std"] = float(arr.std())
                metric_dict[f"val-dist/{k}/min"] = float(arr.min())
                metric_dict[f"val-dist/{k}/max"] = float(arr.max())
                for q, name in [(0.05, "p05"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95")]:
                    metric_dict[f"val-dist/{k}/{name}"] = float(np.quantile(arr, q))

                if _HAS_WANDB:
                    if arr.size > max_pts:
                        idx = rng.choice(arr.size, size=max_pts, replace=False)
                        arr_h = arr[idx]
                    else:
                        arr_h = arr
                    metric_dict[f"val-dist/{k}/hist"] = wandb.Histogram(arr_h, num_bins=bins)

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """

        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # Allow disabling streaming reward via config (reward.streaming: false).
        # When disabled, all rollouts are generated first, then reward is computed
        # in batch via compute_rm_score — better for batch-optimized reward components.
        if not self.config.reward.get("streaming", True):
            enable_agent_reward_loop = False

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            routed_experts = tu.get(output, "routed_experts")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            if routed_experts is not None:
                old_log_prob = tu.get_tensordict(
                    {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
                )
            else:
                old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training:
        # - Fresh start (step 0): always run if val_before_train=True
        # - Resume: run if val rollout for the current step is missing
        #   (handles preemption between checkpoint save and val completion)
        run_val_now = False
        if self.config.trainer.get("val_before_train", True) and self.global_steps == 0:
            run_val_now = True
        elif self.global_steps > 0:
            test_freq = self.config.trainer.get("test_freq", 0)
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            # On resume, checkpoint saves at step N before validation runs.
            # If preempted between save and val, we resume at step N with
            # val rollout missing. Only check if step N is a val boundary.
            if test_freq > 0 and val_data_dir and (self.global_steps % test_freq == 0):
                val_rollout_path = os.path.join(val_data_dir, f"{self.global_steps}.jsonl")
                if not os.path.exists(val_rollout_path):
                    print(f"[resume] Val rollout missing for step {self.global_steps} "
                          f"(expected {val_rollout_path}). Running validation.", flush=True)
                    run_val_now = True

        if run_val_now:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                # ---- GT/BC scheduling ----
                gt_enabled_eff = False
                gt_weight_eff = 0.0
                bc_enabled_eff = False
                bc_coef_eff = 0.0

                gt_cfg = self.config.algorithm
                if gt_cfg.get("gt_rollout_enable", False):
                    step = self.global_steps - 1
                    gt_peak = float(gt_cfg.get("gt_peak_weight", gt_cfg.get("gt_rollout_weight", 1.0)))
                    gt_final = float(gt_cfg.get("gt_final_weight", 0.0))
                    gt_warmup = int(gt_cfg.get("gt_warmup_steps", 0))
                    gt_decay = int(gt_cfg.get("gt_decay_steps", 0))
                    gt_schedule = str(gt_cfg.get("gt_schedule", "linear"))

                    if step < gt_warmup:
                        gt_weight_eff = gt_peak * (step / max(1, gt_warmup))
                    elif gt_decay > 0 and step < gt_warmup + gt_decay:
                        t = (step - gt_warmup) / max(1, gt_decay)
                        if gt_schedule == "cosine":
                            import math
                            gt_weight_eff = gt_final + (gt_peak - gt_final) * 0.5 * (1 + math.cos(math.pi * t))
                        else:
                            gt_weight_eff = gt_peak + (gt_final - gt_peak) * t
                    else:
                        gt_weight_eff = gt_final if gt_decay > 0 else gt_peak

                    gt_enabled_eff = gt_weight_eff > 0

                if gt_cfg.get("bc_enable", False):
                    step = self.global_steps - 1
                    bc_peak = float(gt_cfg.get("bc_peak_coef", 0.0))
                    bc_final = float(gt_cfg.get("bc_final_coef", 0.0))
                    bc_warmup = int(gt_cfg.get("bc_warmup_steps", 0))
                    bc_decay = int(gt_cfg.get("bc_decay_steps", 0))
                    bc_schedule = str(gt_cfg.get("bc_schedule", "linear"))

                    if step < bc_warmup:
                        bc_coef_eff = bc_peak * (step / max(1, bc_warmup))
                    elif bc_decay > 0 and step < bc_warmup + bc_decay:
                        t = (step - bc_warmup) / max(1, bc_decay)
                        if bc_schedule == "cosine":
                            import math
                            bc_coef_eff = bc_final + (bc_peak - bc_final) * 0.5 * (1 + math.cos(math.pi * t))
                        else:
                            bc_coef_eff = bc_peak + (bc_final - bc_peak) * t
                    else:
                        bc_coef_eff = bc_final if bc_decay > 0 else bc_peak

                    bc_enabled_eff = bc_coef_eff > 0

                need_gt_any = gt_enabled_eff or bc_enabled_eff

                # Propagate GT/BC flags to meta_info
                gen_batch.meta_info["gt_weight_eff"] = float(gt_weight_eff)
                gen_batch.meta_info["gt_enabled_eff"] = gt_enabled_eff
                gen_batch.meta_info["gt_in_group_eff"] = gt_enabled_eff
                gen_batch.meta_info["bc_enabled_eff"] = bc_enabled_eff
                gen_batch.meta_info["bc_coef_eff"] = float(bc_coef_eff)
                gen_batch.meta_info["bc_append_gt_eff"] = bool(bc_enabled_eff and not gt_enabled_eff)
                gen_batch.meta_info["bc_only_gt_eff"] = gen_batch.meta_info["bc_append_gt_eff"]

                # Adjust repeat: n-1 when GT is on (policy rollouts + 1 GT later), else n
                n_rollout = self.config.actor_rollout_ref.rollout.n
                repeat_n = n_rollout - 1 if need_gt_any else n_rollout
                gen_batch_output = gen_batch.repeat(
                    repeat_times=repeat_n, interleave=True
                )

                # GT think-prefix: assign gt_think_prefix_reasoning to the last
                # gt_think_prefix_count copies per group; set None for the rest.
                # After repeat(n, interleave=True), samples are ordered:
                #   [p0_r0, p0_r1, ..., p0_r(n-1), p1_r0, p1_r1, ...]
                # We assign think-prefix to the last gt_think_prefix_count per group.
                if self.gt_think_prefix and self.gt_think_prefix_count > 0:
                    tp_count = self.gt_think_prefix_count
                    assert tp_count < repeat_n, (
                        f"gt_think_prefix_count ({tp_count}) must be < repeat_n ({repeat_n}); "
                        "need at least 1 pure policy slot."
                    )
                    total = len(gen_batch_output.batch)
                    gt_reasoning_arr = np.empty(total, dtype=object)
                    gt_reasoning_arr[:] = None  # default: no think-prefix

                    # Check if ground_truth_reasoning is available
                    if "ground_truth_reasoning" in gen_batch_output.non_tensor_batch:
                        src_reasoning = gen_batch_output.non_tensor_batch["ground_truth_reasoning"]
                        enable_thinking = self.config.data.get(
                            "apply_chat_template_kwargs", {}
                        ).get("enable_thinking", False)

                        for group_start in range(0, total, repeat_n):
                            # Assign to last tp_count slots in each group
                            for offset in range(repeat_n - tp_count, repeat_n):
                                idx = group_start + offset
                                reasoning = src_reasoning[idx]
                                if enable_thinking and reasoning is not None and reasoning:
                                    gt_reasoning_arr[idx] = reasoning

                    gen_batch_output.non_tensor_batch["gt_think_prefix_reasoning"] = gt_reasoning_arr

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=repeat_n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Propagate GT/BC flags to post-generation batch
                    batch.meta_info["gt_weight_eff"] = float(gen_batch.meta_info["gt_weight_eff"])
                    batch.meta_info["gt_enabled_eff"] = bool(gen_batch.meta_info.get("gt_enabled_eff", False))
                    batch.meta_info["gt_in_group_eff"] = bool(gen_batch.meta_info.get("gt_in_group_eff", False))
                    batch.meta_info["bc_enabled_eff"] = bool(gen_batch.meta_info.get("bc_enabled_eff", False))
                    batch.meta_info["bc_coef_eff"] = float(gen_batch.meta_info.get("bc_coef_eff", 0.0))
                    batch.meta_info["bc_append_gt_eff"] = bool(gen_batch.meta_info.get("bc_append_gt_eff", False))
                    batch.meta_info["bc_only_gt_eff"] = bool(gen_batch.meta_info.get("bc_only_gt_eff", False))

                    # GT injection: add GT rows when effectively enabled this step
                    if gt_enabled_eff:
                        if "is_gt" not in batch.non_tensor_batch:
                            batch.non_tensor_batch["is_gt"] = np.zeros(len(batch.batch), dtype=np.bool_)
                        gt_dp = self._build_gt_rollouts(batch)
                        if "is_gt" not in gt_dp.non_tensor_batch:
                            gt_dp.non_tensor_batch["is_gt"] = np.ones(len(gt_dp), dtype=np.bool_)

                        if self._rollout_correction_active():
                            raise ValueError(
                                "GT-in-group is enabled but rollout_correction is active (bypass_mode=False). "
                                "This is not supported: GT rows do not have rollout-time behavior logprobs. "
                                "Set algorithm.rollout_correction.bypass_mode=true when using GT-in-group, "
                                "or disable GT-in-group."
                            )

                        keys_to_align = set(batch.batch.keys()) | set(gt_dp.batch.keys())
                        self._align_tensordict_keys_for_concat(batch, gt_dp, keys=keys_to_align)

                        # Align non_tensor_batch shapes before concat.
                        # Ensure every key exists in both, with compatible shapes on all dims except axis=0.
                        all_ntb_keys = set(batch.non_tensor_batch.keys()) | set(gt_dp.non_tensor_batch.keys())
                        drop_keys = []
                        for k in all_ntb_keys:
                            in_a = k in batch.non_tensor_batch
                            in_b = k in gt_dp.non_tensor_batch
                            Ba = len(batch)
                            Bb = len(gt_dp)
                            if in_a and in_b:
                                a_arr = batch.non_tensor_batch[k]
                                b_arr = gt_dp.non_tensor_batch[k]
                                if not (isinstance(a_arr, np.ndarray) and isinstance(b_arr, np.ndarray)):
                                    continue
                                # Make ndims match first
                                while a_arr.ndim < b_arr.ndim:
                                    a_arr = np.expand_dims(a_arr, axis=-1)
                                while b_arr.ndim < a_arr.ndim:
                                    b_arr = np.expand_dims(b_arr, axis=-1)
                                # Check trailing shapes match
                                if a_arr.shape[1:] != b_arr.shape[1:]:
                                    # Incompatible trailing shapes — flatten both to 1D object arrays
                                    try:
                                        a_flat = np.empty(Ba, dtype=object)
                                        b_flat = np.empty(Bb, dtype=object)
                                        for i in range(Ba):
                                            a_flat[i] = a_arr[i]
                                        for i in range(Bb):
                                            b_flat[i] = b_arr[i]
                                        a_arr = a_flat
                                        b_arr = b_flat
                                    except Exception:
                                        drop_keys.append(k)
                                        continue
                                batch.non_tensor_batch[k] = a_arr
                                gt_dp.non_tensor_batch[k] = b_arr
                            elif in_a and not in_b:
                                a_arr = batch.non_tensor_batch[k]
                                if isinstance(a_arr, np.ndarray):
                                    if a_arr.dtype.kind == 'O':
                                        gt_dp.non_tensor_batch[k] = np.array([None] * Bb, dtype=object)
                                    else:
                                        gt_dp.non_tensor_batch[k] = np.zeros((Bb,) + a_arr.shape[1:], dtype=a_arr.dtype)
                            elif in_b and not in_a:
                                b_arr = gt_dp.non_tensor_batch[k]
                                if isinstance(b_arr, np.ndarray):
                                    if b_arr.dtype.kind == 'O':
                                        batch.non_tensor_batch[k] = np.array([None] * Ba, dtype=object)
                                    else:
                                        batch.non_tensor_batch[k] = np.zeros((Ba,) + b_arr.shape[1:], dtype=b_arr.dtype)
                        for k in drop_keys:
                            batch.non_tensor_batch.pop(k, None)
                            gt_dp.non_tensor_batch.pop(k, None)

                        batch = DataProto.concat([batch, gt_dp])
                        try:
                            is_gt_np = np.asarray(batch.non_tensor_batch.get("is_gt", None), dtype=np.bool_).reshape(-1)
                            if is_gt_np is not None and len(is_gt_np) == len(batch.batch):
                                batch.batch["is_gt"] = torch.as_tensor(
                                    is_gt_np, dtype=torch.bool, device=batch.batch["responses"].device
                                )
                        except Exception as e:
                            print(f"[GT] Warning: could not materialize batch['is_gt']: {e!r}")

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score (batch path)
                        # Runs when: (a) use_rm is True, or (b) streaming is disabled
                        if "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Log reward component metrics (all + policy-only when GT is in group)
                    _gt_in_group_active = bool(
                        gt_enabled_eff
                        and self.config.algorithm.get("gt_in_group", False)
                        and "is_gt" in batch.non_tensor_batch
                    )
                    _is_gt_comp = None
                    if _gt_in_group_active:
                        _is_gt_comp = np.array(
                            batch.non_tensor_batch.get("is_gt", np.zeros(reward_tensor.shape[0], dtype=bool)),
                            dtype=bool,
                        )

                    if reward_extra_infos_dict:
                        for comp_name, comp_vals in reward_extra_infos_dict.items():
                            if isinstance(comp_vals, (list, np.ndarray)):
                                arr = np.array(comp_vals, dtype=np.float64)
                                if arr.size > 0:
                                    metrics[f"reward_comp/{comp_name}_mean"] = float(np.nanmean(arr))
                                    # Policy-only split when GT is in group
                                    if _is_gt_comp is not None and arr.shape[0] == _is_gt_comp.shape[0]:
                                        pol_arr = arr[~_is_gt_comp]
                                        if pol_arr.size > 0:
                                            metrics[f"reward_comp_policy/{comp_name}_mean"] = float(np.nanmean(pol_arr))

                        # Also log diagnostic dicts (story_quality_diag, etc.)
                        for diag_key, diag_list in reward_extra_infos_dict.items():
                            if not (isinstance(diag_list, list) and diag_key.endswith("_diag")):
                                continue
                            comp_name = diag_key[:-5]
                            per_field_vals: dict[str, list[float]] = {}
                            per_field_vals_policy: dict[str, list[float]] = {}
                            for idx_d, diag in enumerate(diag_list):
                                if not isinstance(diag, dict):
                                    continue
                                raw = diag.get("raw", diag)
                                if not isinstance(raw, dict):
                                    continue
                                is_gt_row = bool(_is_gt_comp is not None and idx_d < len(_is_gt_comp) and _is_gt_comp[idx_d])
                                for field_name, field_val in raw.items():
                                    if isinstance(field_val, (int, float)):
                                        per_field_vals.setdefault(field_name, []).append(float(field_val))
                                        if not is_gt_row:
                                            per_field_vals_policy.setdefault(field_name, []).append(float(field_val))
                            for field_name, vals in per_field_vals.items():
                                if vals:
                                    metrics[f"{comp_name}/{field_name}_mean"] = float(np.nanmean(np.array(vals)))
                            # Policy-only diagnostic metrics when GT is in group
                            if _gt_in_group_active:
                                for field_name, vals in per_field_vals_policy.items():
                                    if vals:
                                        metrics[f"{comp_name}_policy/{field_name}_mean"] = float(np.nanmean(np.array(vals)))

                        if _is_gt_comp is not None:
                            metrics.update(_compute_hri_reward_component_metrics(reward_extra_infos_dict, _is_gt_comp))

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # NaN safety for advantages
                        batch.batch["advantages"] = torch.nan_to_num(
                            batch.batch["advantages"], nan=0.0, posinf=0.0, neginf=0.0
                        )

                        # GT advantage weighting: per-sequence weights for the policy gradient
                        if gt_enabled_eff and "is_gt" in batch.non_tensor_batch:
                            adv = batch.batch["advantages"]
                            B_adv = adv.size(0)
                            is_gt_t = torch.as_tensor(
                                batch.non_tensor_batch["is_gt"], dtype=torch.bool, device=adv.device
                            )
                            weights = torch.ones(B_adv, dtype=torch.float32, device=adv.device)
                            weights[is_gt_t] = gt_weight_eff
                            batch.batch["sample_weights_gt"] = weights
                        else:
                            if "sample_weights_gt" in batch.batch:
                                del batch.batch["sample_weights_gt"]

                        # Stable group id for analytics
                        uids_np = batch.non_tensor_batch["uid"]
                        _, inv = np.unique(uids_np, return_inverse=True)
                        batch.non_tensor_batch["group_ids"] = inv.astype(np.int64)

                        if gt_enabled_eff and "is_gt" in batch.non_tensor_batch:
                            is_gt_arr_hri = np.asarray(batch.non_tensor_batch["is_gt"], dtype=bool)
                            hri_metrics = _compute_hri_group_metrics(
                                reward_tensor=reward_tensor,
                                response_mask=batch.batch["response_mask"],
                                advantages=batch.batch["advantages"],
                                group_ids=batch.non_tensor_batch["group_ids"],
                                is_gt_arr=is_gt_arr_hri,
                            )
                            metrics.update(hri_metrics)
                            metrics.update(_compute_hri_split_batch_metrics(batch=batch, is_gt_arr=is_gt_arr_hri, use_critic=self.use_critic))
                            _append_hri_sequence_log(
                                log_path=os.path.join(self.config.trainer.default_local_dir, "hri_sequence_stats.jsonl"),
                                global_step=self.global_steps,
                                batch=batch,
                                reward_tensor=reward_tensor,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                group_ids=batch.non_tensor_batch["group_ids"],
                                is_gt_arr=is_gt_arr_hri,
                            )

                    # Policy-only reward metrics
                    B_batch = reward_tensor.shape[0]
                    is_gt_arr = np.array(batch.non_tensor_batch.get("is_gt", np.zeros(B_batch, dtype=bool)))
                    resp_mask_rw = batch.batch["response_mask"].to(torch.float32)
                    seq_len_rw = resp_mask_rw.sum(-1).clamp_min(1.0)
                    seq_reward = (reward_tensor.sum(-1) / seq_len_rw).detach().cpu().numpy()
                    if (~is_gt_arr).any():
                        metrics["reward/seq_mean_policy"] = float(np.nanmean(seq_reward[~is_gt_arr]))
                        metrics["reward/seq_std_policy"] = float(np.nanstd(seq_reward[~is_gt_arr]))

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            # Propagate GT/BC meta_info to the batch for the actor
                            batch.meta_info["global_steps"] = self.global_steps
                            batch.meta_info["gt_weight_eff"] = float(gt_weight_eff)
                            batch.meta_info["gt_enabled_eff"] = gt_enabled_eff
                            batch.meta_info["bc_enabled_eff"] = bc_enabled_eff
                            batch.meta_info["bc_coef_eff"] = float(bc_coef_eff)

                            # GT-specific PPO clip params
                            try:
                                gt_lo = getattr(self.config.algorithm, "gt_clip_ratio_low", None)
                                gt_hi = getattr(self.config.algorithm, "gt_clip_ratio_high", None)
                                if gt_lo is None:
                                    gt_lo = getattr(self.config.actor_rollout_ref.actor, "clip_ratio_low", None)
                                if gt_hi is None:
                                    gt_hi = getattr(self.config.actor_rollout_ref.actor, "clip_ratio_high", None)
                                if gt_lo is None:
                                    gt_lo = float(getattr(self.config.actor_rollout_ref.actor, "clip_ratio", 0.2))
                                if gt_hi is None:
                                    gt_hi = float(getattr(self.config.actor_rollout_ref.actor, "clip_ratio", 0.2))
                                batch.meta_info["gt_clip_ratio_low"] = float(gt_lo)
                                batch.meta_info["gt_clip_ratio_high"] = float(gt_hi)
                            except Exception as e:
                                print(f"[GT] Warning: could not set gt_clip_ratio in meta_info: {e!r}")

                            # BC-only GT append: when BC is on but GT-in-group is off, append GT rows for BC loss only
                            batch_for_actor = batch
                            bc_append_gt = bool(batch.meta_info.get("bc_append_gt_eff", False))
                            bc_active = bool(bc_enabled_eff and bc_coef_eff > 0.0)

                            if bc_append_gt and bc_active:
                                if "is_gt" not in batch_for_actor.non_tensor_batch:
                                    batch_for_actor.non_tensor_batch["is_gt"] = np.zeros(len(batch_for_actor), dtype=np.bool_)

                                try:
                                    gt_dp = self._build_gt_rollouts(batch_for_actor)
                                except Exception as e:
                                    raise RuntimeError(f"[BC-only] Failed to build GT rollouts: {e!r}") from e

                                if gt_dp is not None:
                                    Bgt, Tresp = gt_dp.batch["responses"].shape
                                    dev = gt_dp.batch["responses"].device

                                    if "response_mask" not in gt_dp.batch:
                                        gt_dp.batch["response_mask"] = gt_dp.batch["attention_mask"][:, -Tresp:].to(torch.float32)

                                    gt_dp.batch["old_log_probs"] = torch.zeros((Bgt, Tresp), dtype=torch.float32, device=dev)
                                    gt_dp.batch["advantages"] = torch.zeros((Bgt, Tresp), dtype=torch.float32, device=dev)

                                    if "ref_log_prob" in batch_for_actor.batch and "ref_log_prob" not in gt_dp.batch:
                                        gt_dp.batch["ref_log_prob"] = torch.zeros((Bgt, Tresp), dtype=torch.float32, device=dev)

                                    if "is_gt" not in gt_dp.non_tensor_batch:
                                        gt_dp.non_tensor_batch["is_gt"] = np.ones(len(gt_dp), dtype=np.bool_)

                                    # is_gt as tensor
                                    batch_for_actor.batch["is_gt"] = torch.zeros(
                                        (len(batch_for_actor),), dtype=torch.bool, device=dev
                                    )
                                    gt_dp.batch["is_gt"] = torch.ones((len(gt_dp),), dtype=torch.bool, device=dev)

                                    keys_to_align = set(batch_for_actor.batch.keys()) | set(gt_dp.batch.keys())
                                    self._align_tensordict_keys_for_concat(batch_for_actor, gt_dp, keys=keys_to_align)

                                    batch_for_actor = DataProto.concat([batch_for_actor, gt_dp])
                                    batch_for_actor.meta_info = dict(batch.meta_info)

                                    try:
                                        idx = torch.randperm(len(batch_for_actor), device="cpu")
                                        batch_for_actor.reorder(idx)
                                    except Exception as e:
                                        print(f"[BC-only] Warning: could not reorder actor batch: {e!r}")

                                    try:
                                        n_gt = int(batch_for_actor.batch["is_gt"].sum().item())
                                        print(f"[BC-only] actor batch size={len(batch_for_actor)} gt_rows={n_gt}")
                                    except Exception:
                                        pass

                            actor_output = self._update_actor(batch_for_actor)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GT think-prefix metrics
                if self.gt_think_prefix and "think_prefix_len" in batch.non_tensor_batch:
                    tp_lens = np.array(
                        [int(x) if x is not None else 0 for x in batch.non_tensor_batch["think_prefix_len"]],
                        dtype=np.float32,
                    )
                    n_tp = int(np.sum(tp_lens > 0))
                    metrics["gt_think_prefix/count"] = n_tp
                    if n_tp > 0:
                        metrics["gt_think_prefix/mean_len"] = float(np.mean(tp_lens[tp_lens > 0]))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
