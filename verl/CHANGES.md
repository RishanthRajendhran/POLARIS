# Custom VERL Modifications

This document describes all modifications made to this VERL fork relative to upstream.
It covers both the conceptual motivation and the code-level implementation, and provides
detailed guidance on configuring the new parameters.

---

## Table of Contents

1. [GRPO Enhancements](#1-grpo-enhancements)
   - 1a. Outcome Score Modes
   - 1b. Token-Mean Loss Aggregation
2. [Ground-Truth (GT) Rollout Injection](#2-ground-truth-gt-rollout-injection)
3. [Behavioral Cloning (BC) Loss](#3-behavioral-cloning-bc-loss)
4. [Composite Reward Manager](#4-composite-reward-manager)
5. [Auto-Weight Scheduler](#5-auto-weight-scheduler)
6. [Validation Reward Distributions](#6-validation-reward-distributions)
7. [Separate Val Reward Kwargs](#7-separate-val-reward-kwargs)
8. [System Message Injection](#8-system-message-injection)
9. [DataProto Chunking Robustness](#9-dataproto-chunking-robustness)
10. [Config Quick-Reference](#10-config-quick-reference)

---

## 1. GRPO Enhancements

### Conceptual Overview

Standard GRPO computes a scalar "score" per response by summing token-level rewards, then
baselines within a uid group. This fork adds two independent extensions:

1. **Outcome score reduction modes** — control how token rewards collapse to a per-response scalar.
2. **Token-mean loss aggregation** — a DAPO-style loss normalisation option.

### 1a. Outcome Score Modes

**Code:** `verl/trainer/ppo/core_algos.py` → `compute_grpo_outcome_advantage`, outcome branch.

The token rewards are first reduced to a scalar score per response:

| `grpo_score_mode` | Formula | When to use |
|---|---|---|
| `"sum"` | `score = Σ r_t` | Original VERL behaviour. Score scales with length. |
| `"sum_norm"` | `score = Σ r_t / T` | Divides by the fixed tensor width T. Use when rewards are already distributed across the sequence and you want length-invariant scoring without masking. |
| `"mean"` | `score = Σ (r_t · m_t) / Σ m_t` | Masked mean over valid response tokens. **Best default when rewards are dense** (every token gets a reward). |

**Config:**
```yaml
algorithm:
  grpo_score_mode: "mean"       # "sum" | "sum_norm" | "mean"
  grpo_score_clip: 0.0          # clip scalar scores before baselining; 0 = off
```

**`grpo_score_clip`** hard-clips the per-response score to `[-clip, +clip]` before computing
the group mean/std. Useful when a few outlier responses dominate the baseline.

### 1b. Token-Mean Loss Aggregation

**Code:** `verl/trainer/ppo/core_algos.py` → `agg_loss`.

```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean"   # new option; existing: "token", "seq-mean-token-sum", etc.
```

`"token-mean"` divides the total loss by the total number of valid tokens across the batch
(as in DAPO). This means a step with many long responses has the same effective learning rate
per token as a step with few short ones. Use this when response lengths vary significantly and
you observe training instability with sequence-level normalisation.

---

## 2. Ground-Truth (GT) Rollout Injection

### Conceptual Overview

In standard GRPO, the group for a prompt consists entirely of model rollouts. GT injection adds
one additional "response" to each group: the ground-truth answer from the dataset. This provides
a stable high-reward anchor in every group, especially useful early in training when the model
rarely produces correct responses on its own (cold-start).

The GT response is constructed by the trainer (not the rollout engine) by:
1. Tokenizing the ground-truth text from `non_tensor_batch["ground_truth"]` (or `solution`,
   `answer`, `target`).
2. If `enable_thinking=True`, wrapping it as `<think>\n{gt_reasoning}\n</think>\n{gt_answer}`.
3. Padding/truncating to the response length `R`.

Because GT rows bypass the rollout engine, they have no behaviour-policy log-probs; rollout
correction (`algorithm.rollout_correction`) must be disabled when GT is active.

### Dataset Requirements

Your dataset's `non_tensor_batch` must contain at least one of:
- `ground_truth`, `solution`, `answer`, or `target` — the answer string.
- If `enable_thinking=True`, also `ground_truth_reasoning` — the chain-of-thought string
  (without `<think>` tags; the trainer wraps it).

### Config

```yaml
algorithm:
  gt_rollout_enable: false        # master switch
  gt_rollout_weight: 1.0          # base importance weight applied to GT advantages

  # --- Scheduling (optional) ---
  gt_schedule: "linear"           # "linear" | "cosine"  (decay shape)
  gt_warmup_steps: 0              # steps to ramp weight from 0 → gt_peak_weight
  gt_decay_steps: 0               # steps to decay from gt_peak_weight → gt_final_weight
  gt_peak_weight: 1.0             # peak effective weight (alias: gt_rollout_weight)
  gt_final_weight: 0.0            # final weight; if 0 GT stops being added entirely

actor_rollout_ref:
  rollout:
    n: 8                          # MUST be ≥ 2 when gt_rollout_enable=true
                                  # The trainer uses n-1 policy rollouts + 1 GT per group
```

**Weight schedule mechanics:**

```
Step < gt_warmup_steps:
    w_eff = gt_peak_weight * (step / gt_warmup_steps)

gt_warmup_steps ≤ step < gt_warmup_steps + gt_decay_steps:
    t = (step - gt_warmup_steps) / gt_decay_steps
    linear:  w_eff = gt_peak_weight + (gt_final_weight - gt_peak_weight) * t
    cosine:  w_eff = gt_final_weight + (gt_peak_weight - gt_final_weight) * 0.5 * (1 + cos(π·t))

step ≥ gt_warmup_steps + gt_decay_steps:
    w_eff = gt_final_weight
```

When `w_eff = 0`, no GT rows are appended and the batch is identical to standard GRPO.

**n requirement:** With GT enabled, the trainer sets `repeat_times = n - 1`, producing `n-1`
policy rollouts per prompt. The GT row brings the group back to size `n`. Therefore `n ≥ 2`
is enforced — the code raises a `ValueError` at the start of training if this is violated.

**Thinking mode:** Set `data.apply_chat_template_kwargs.enable_thinking: true` to enable. The
dataset must then provide `ground_truth_reasoning` per row. Do not include `<think>`/`</think>`
in these fields; the trainer injects them.

**GT clip ratios (optional):** You can apply different PPO clip ratios for GT rows vs policy rows:
```yaml
algorithm:
  gt_clip_ratio_low: 0.2
  gt_clip_ratio_high: 0.2
```
If not set, the actor's standard `clip_ratio` is used for GT rows too.

---

## 3. Behavioral Cloning (BC) Loss

### Conceptual Overview

BC loss provides an additional supervised signal: the actor minimises the negative log-likelihood
of the ground-truth response, independently of the GRPO advantage. This is complementary to GT
injection:

- **GT-in-group** (`gt_rollout_enable=true`): the GT response participates in reward scoring and
  advantage computation. Encourages the model to "beat" the GT.
- **BC-only** (`bc_enable=true`, `gt_rollout_enable=false`): GT is appended *only* for the BC
  loss; advantages for GT rows are zeroed so they don't affect the GRPO gradient.
- **Both active**: GT participates in GRPO *and* gets a BC loss. Advantages are non-zero and BC
  reinforces the ground truth directly.

### Config

```yaml
algorithm:
  bc_enable: false          # master switch
  bc_peak_coef: 0.1         # peak BC coefficient (scales the BC NLL loss)
  bc_final_coef: 0.0        # final BC coefficient after decay
  bc_warmup_steps: 0        # steps to ramp from 0 → bc_peak_coef
  bc_decay_steps: 0         # steps to decay from bc_peak_coef → bc_final_coef
  bc_schedule: "linear"     # "linear" | "cosine"
```

The schedule follows the same piecewise warmup → decay logic as GT weights. When
`bc_coef_eff = 0`, no GT rows are appended for BC and no BC loss is computed.

**When to use BC:**
- Early training as a curriculum: ramp up BC, then decay it as GRPO takes over.
- Sparse reward settings: the model struggles to get any reward signal; BC keeps it on track.
- With `bc_peak_coef` around `0.05–0.2`. Too high and BC dominates GRPO; too low and it has no effect.

**Interaction with GT:**

| `gt_rollout_enable` | `bc_enable` | Behaviour |
|---|---|---|
| `false` | `false` | Standard GRPO |
| `true` | `false` | GT in group; GT advantages non-zero |
| `false` | `true` | GT appended only for BC; GRPO unaffected |
| `true` | `true` | GT in GRPO group AND gets BC loss |

---

## 4. Composite Reward Manager

### Conceptual Overview

`CompositeBatchRewardManager` is a new reward manager (registered as `"composite_batch"`) that
calls your reward function once per *batch* (not per-sample). This is the intended interface
for composite reward functions that compute multiple reward components in a single pass and
return them as `reward_extra_info`.

**Code:** `verl/workers/reward_manager/composite_batch.py`

### How it works

```python
out = reward_fn(data, return_dict=True)
# out == {
#   "reward_tensor": Tensor[B, T],       # token-level rewards
#   "reward_extra_info": {               # per-component scores, per-sample lists
#       "component_a": [float, ...],     # length B
#       "component_b": [float, ...],
#   }
# }
```

The manager:
1. Injects `tokenizer` and `reward_kwargs` into `data.meta_info` so your reward function can
   access them without needing them as constructor arguments.
2. Normalises `reward_extra_info`: broadcasts singleton lists `[x] → [x]*B`, drops wrong-length
   lists, drops non-list values.
3. Falls back gracefully to `(tensor, dict)` tuple return if `return_dict=True` raises `TypeError`.

### Config

```yaml
reward_model:
  reward_manager: "composite_batch"
  reward_kwargs:
    # anything here is forwarded to your reward function via data.meta_info["__reward_cfg_alg__"]
    reward_components:
      correctness: 1.0
      format: 0.2
```

### Writing a compatible reward function

```python
def my_reward(data: DataProto, return_dict: bool = True):
    cfg = data.meta_info.get("__reward_cfg_alg__", {})
    tokenizer = data.meta_info.get("__tokenizer__")
    weights = cfg.get("reward_components", {})
    ...
    return {
        "reward_tensor": reward_tensor,   # [B, T] float
        "reward_extra_info": {
            "correctness": [float, ...],  # length B
            "format":      [float, ...],
        }
    }
```

### Metric logging

All keys in `reward_extra_info` are automatically logged as:
- `reward_comp/{key}_mean` — batch mean
- `reward_comp/{key}_mean_policy` — batch mean over non-GT rows only
- `{comp_name}/{field}_mean` — for diagnostic dicts (keys ending in `_diag`)

---

## 5. Auto-Weight Scheduler

### Conceptual Overview

Inspired by AW-GRPO, the auto-weight scheduler adaptively adjusts the weights of reward
components during training. Components whose scores are *improving* (positive slope over
a recent window) get their weight decreased; stagnating or regressing components get more
weight. The intuition: once a component is being solved, it doesn't need as much emphasis.

**Code:** `verl/utils/auto_weight.py` (`AutoWeightScheduler`, `AWSchedConfig`)

### Algorithm

Every `update_every` steps (after collecting at least `min_points` data points):

1. Fit a degree-`deg` polynomial to the recent `window` per-step means of each key.
2. Extract the slope `ŝ_k` (leading coefficient).
3. Exponentiated gradient update:
   ```
   α_k ← clip(α_k · exp(-η · ŝ_k),  w_min_k, w_max_k)
   ```
4. Re-normalise `α` to sum to 1 within the tracked subset.
5. Return outer weights: `w_k = α_k · sum_target`.

If all weights hit their min or max bounds simultaneously, updates are frozen until scores change.

### Config

```yaml
algorithm:
  auto_weight:
    enable: false
    keys: ["correctness", "format"]   # reward_extra_info keys to track
    update_every: 10                  # steps between weight updates
    window: 100                       # rolling window size (steps)
    min_points: 30                    # minimum history before first update
    eta: 0.5                          # learning rate; higher = faster adaptation
    w_min: 0.05                       # global lower bound on any weight
    w_max: 0.95                       # global upper bound on any weight
    deg: 1                            # polynomial degree for slope estimation (1=linear)
    ema_beta: 0.0                     # EMA smoothing of per-step means; 0=off
    start_step: 0                     # don't update before this step
    normalize_domain: "subset"        # "subset": re-scale so keys sum to sum_target
                                      # "global": return normalised alphas directly
    sum_target: "auto"                # "auto": use initial sum of weights from reward_kwargs
                                      #  float: fixed target sum
    log_to_wandb: true                # log updated weights as aw/alpha/{key}
    freeze_if_clamped: true           # skip update if all keys are at their bounds
    key_overrides: null               # per-key overrides, e.g.:
                                      # {"correctness": {"w_min": 0.3, "w_max": 0.9, "eta_scale": 0.5}}
```

**Tuning guidance:**

| Situation | Recommendation |
|---|---|
| Fast-moving rewards | Lower `eta` (0.1–0.3), longer `window` (200+) |
| Noisy rewards | Enable `ema_beta` (0.7–0.9) to smooth history |
| One component dominates | Set per-key `w_max` via `key_overrides` |
| Component must stay relevant | Set per-key `w_min` via `key_overrides` |
| Slow training | Increase `update_every` to reduce overhead |

**`normalize_domain`:**
- `"subset"`: the scheduler only adjusts weights for `keys`; other reward components are
  untouched. The `keys` weights sum to `sum_target` (preserving their total mass).
- `"global"`: returns raw normalised alphas; the caller is responsible for overall weight normalisation.

**Interaction with `reward_kwargs`:** Updated weights are written back into
`self.reward_fn.rw["reward_weights"]` each cycle, so your reward function sees the new weights
on the next batch without any restart.

---

## 6. Validation Reward Distributions

### Conceptual Overview

Logs per-component score distributions (quantiles, std, optional W&B histograms) at each
validation step. Useful for understanding which reward components the model is or isn't learning.

**Code:** `verl/trainer/ppo/ray_trainer.py` → `_validate` method.

### Config

```yaml
trainer:
  log_val_reward_distributions: false   # master switch
  val_reward_hist_bins: 50              # histogram bins (W&B)
  val_reward_hist_max_points: 20000     # downsample to this many pts before histogram
  val_reward_dist_keys: null            # null = auto-detect all numeric per-sample keys
                                        # list: ["correctness", "format"]  to restrict
```

**Auto-detection** excludes keys with:
- Prefix `storyq_`
- Suffix `_diag` or `_debug`
- Exact names in `{"weights_used", "pos_gate", "story_quality_diag", "human_like_diag"}`

Logged metrics per key: `val-dist/{key}/n`, `mean`, `std`, `min`, `max`, `p05`, `p25`, `p50`,
`p75`, `p95`, and (if W&B) `hist`.

**Validation score normalisation:** Scores are normalised by response length
(`reward_tensor.sum(-1) / seq_len`), giving a per-token average. This is the right default for
dense reward functions and makes scores comparable across different response lengths.

---

## 7. Separate Val Reward Kwargs

### Conceptual Overview

Allows using different reward function configuration at validation time vs training time.
A common use case: disable expensive components during training (to save compute) but re-enable
them for validation metrics.

**Code:** `verl/trainer/main_ppo.py` and `verl/trainer/ppo/ray_trainer.py`.

### Config

```yaml
reward_model:
  reward_kwargs:               # used for training
    reward_components:
      correctness: 1.0
      fast_fluency: 0.2
  val_reward_kwargs:           # deep-merged OVER reward_kwargs for validation
    reward_components:
      correctness: 1.0
      fast_fluency: 0.0
      slow_quality: 1.0       # expensive component, only at val
```

`val_reward_kwargs` is a partial override — keys present in `val_reward_kwargs` overwrite the
corresponding keys from `reward_kwargs`; absent keys inherit from `reward_kwargs`. You don't
need to repeat the full config.

---

## 8. System Message Injection

### Conceptual Overview

Allows injecting a system message into all prompts at dataset-load time without modifying the
dataset files. Works for both RL (`RLHFDataset`) and SFT (`SFTDataset`).

**Code:** `verl/utils/dataset/rl_dataset.py`, `verl/utils/dataset/sft_dataset.py`.

### Config

```yaml
data:
  apply_chat_template_kwargs:
    system_message: "You are a helpful assistant."
    # other valid apply_chat_template kwargs go here too
    # e.g., enable_thinking: true   (for Qwen3 thinking mode)
```

The system message is extracted and **removed** from `apply_chat_template_kwargs` before the
kwargs are forwarded to `tokenizer.apply_chat_template`. It is then inserted as the first
message in every conversation that doesn't already start with a system turn.

**Note:** If your conversations already have a system message, the injection is skipped
(checked via `messages[0]["role"] != "system"`).

---

## 9. DataProto Chunking Robustness

### Conceptual Overview

The original `DataProto.chunk()` required all `non_tensor_batch` values to be numpy arrays of
length `B`. This caused crashes when reward functions or custom code stored scalar constants,
Python dicts, or 0-D arrays in `non_tensor_batch`. The patched version handles these gracefully.

**Code:** `verl/protocol.py` → `DataProto.chunk`.

### What changed

- Values that aren't length-`B` arrays (scalars, dicts, strings, 0-D arrays) are broadcast to
  `[v] * B` before slicing.
- The chunking then creates proper `numpy.ndarray(dtype=object)` slices for each chunk.
- A helper `normalize_non_tensor_batch` (`verl/utils/dataproto_utils.py`) is called in the
  training loop before `update_actor` to pre-clean the batch: known per-batch constants
  (`system_message`, `apply_chat_template_kwargs`, etc.) are moved to `meta_info`, and any
  remaining non-B-length values are broadcast.

No config changes needed; this is transparent.

---

## 10. Config Quick-Reference

Complete annotated config block for all new parameters. Include only what you need.

```yaml
algorithm:
  # ---- GRPO outcome score ----
  grpo_score_mode: "mean"          # "sum" | "sum_norm" | "mean"
  grpo_score_clip: 0.0             # clip per-response score; 0=off

  # ---- GT rollout injection ----
  gt_rollout_enable: false
  gt_rollout_weight: 1.0           # alias for gt_peak_weight
  gt_schedule: "linear"            # "linear" | "cosine"
  gt_warmup_steps: 0
  gt_decay_steps: 0
  gt_peak_weight: 1.0
  gt_final_weight: 0.0
  gt_clip_ratio_low: null          # null = use actor clip_ratio
  gt_clip_ratio_high: null

  # ---- Behavioral cloning ----
  bc_enable: false
  bc_peak_coef: 0.0
  bc_final_coef: 0.0
  bc_warmup_steps: 0
  bc_decay_steps: 0
  bc_schedule: "linear"

  # ---- Auto-weight scheduler ----
  auto_weight:
    enable: false
    keys: []
    update_every: 10
    window: 100
    min_points: 30
    eta: 0.5
    w_min: 0.05
    w_max: 0.95
    deg: 1
    ema_beta: 0.0
    start_step: 0
    normalize_domain: "subset"
    sum_target: "auto"
    log_to_wandb: true
    freeze_if_clamped: true
    key_overrides: null

actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean"    # new option alongside existing modes

  rollout:
    n: 8                           # must be ≥ 2 if gt_rollout_enable=true

reward_model:
  reward_manager: "composite_batch"
  reward_kwargs: {}
  val_reward_kwargs: {}            # partial override of reward_kwargs for validation

trainer:
  log_val_reward_distributions: false
  val_reward_hist_bins: 50
  val_reward_hist_max_points: 20000
  val_reward_dist_keys: null

data:
  apply_chat_template_kwargs:
    system_message: null           # string, or null to disable injection
```

---

## Common Configurations

### Dense reward, GRPO, no extras

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
  grpo_score_mode: "mean"
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean"
reward_model:
  reward_manager: "composite_batch"
```

### Dense reward + GT injection for cold-start

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
  grpo_score_mode: "mean"
  gt_rollout_enable: true
  gt_peak_weight: 1.0
  gt_warmup_steps: 200
  gt_decay_steps: 800
  gt_final_weight: 0.0
actor_rollout_ref:
  rollout:
    n: 8                     # 7 policy rollouts + 1 GT
```

### Multi-component reward with auto-weighting

```yaml
algorithm:
  auto_weight:
    enable: true
    keys: ["correctness", "format"]
    eta: 0.3
    window: 200
    min_points: 50
    sum_target: "auto"
reward_model:
  reward_manager: "composite_batch"
  reward_kwargs:
    reward_components:
      correctness: 0.8
      format: 0.2
```

### GT injection + BC warmup curriculum

```yaml
algorithm:
  gt_rollout_enable: true
  gt_peak_weight: 1.0
  gt_decay_steps: 1000
  gt_final_weight: 0.0
  bc_enable: true
  bc_peak_coef: 0.1
  bc_warmup_steps: 100
  bc_decay_steps: 500
  bc_final_coef: 0.0
actor_rollout_ref:
  rollout:
    n: 4
```
