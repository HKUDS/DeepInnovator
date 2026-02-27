# Copyright 2026 DeepInnovator Ltd. and/or its affiliates
# Copyright 2025 CollabLLM team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

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

import asyncio
import importlib.util
import logging
import os
import sys
from typing import Any, Callable, Optional

import litellm
import torch
import yaml
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


async def conversation_level_reward_func(
    data_source, messages, ground_truth, extra_info, metrics, **kwargs
) -> dict[str, torch.Tensor]:
    num_retries = kwargs.get("num_retries", 6)

    rewards = {}
    
    for metric in metrics:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metric_file_path = os.path.join(current_dir, f"metrics/{metric}.py")

        if not os.path.exists(metric_file_path):
            print(f"Error: Metric file '{metric_file_path}' not found. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue

        spec = importlib.util.spec_from_file_location(f"metric_{metric}", metric_file_path)
        if spec is None:
            print(f"Error: Could not create spec for metric '{metric}'. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue

        module = importlib.util.module_from_spec(spec)

        try:
            sys.modules[f"metric_{metric}"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error loading metric module from '{metric_file_path}': {e}. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue

        if not hasattr(module, "compute_score"):
            print(
                f"Error: Function 'compute_score' not found in '{metric_file_path}'. Assigning 0 to metric '{metric}'."
            )
            rewards[metric] = 0.0
            continue

        compute_score_fn = module.compute_score

        for attempt in range(num_retries):
            try:
                if asyncio.iscoroutinefunction(compute_score_fn):
                    rewards[metric] = await compute_score_fn(data_source, messages, ground_truth, extra_info, **kwargs)
                else:
                    rewards[metric] = compute_score_fn(data_source, messages, ground_truth, extra_info, **kwargs)
                break
            except Exception as e:
                if attempt == num_retries - 1:
                    print(
                        f"Error: Failed to compute metric '{metric}' after {num_retries} attempts. "
                        f"Last error: {e}. Assigning 0 to metric '{metric}'."
                    )
                    rewards[metric] = 0.0
                else:
                    print(f"Attempt {attempt + 1} failed for metric '{metric}': {e}. Retrying...")
                    if isinstance(e, litellm.RateLimitError):
                        await asyncio.sleep(max(2**attempt, 60))

    return {metric: torch.tensor(reward, dtype=torch.float32) for metric, reward in rewards.items()}




@register("DeepInnovator")
class DeepInnovatorRewardManager(AbstractRewardManager):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int = 0,
        reward_fn_key: str = "data_source",
        compute_score: Optional[Callable] = None,
        normalize_by_data_source: bool = False,
        reward_kwargs_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not reward_kwargs_path:
            raise ValueError(
                "reward_kwargs_path is required. Please provide the path to the reward config YAML file."
            )
        
        if not os.path.exists(reward_kwargs_path):
            raise ValueError(
                f"Reward config file not found: {reward_kwargs_path}. "
                f"Please check the file path."
            )
        
        with open(reward_kwargs_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
            config = file_config.get('config', {})
        
        if 'metric_weights' not in config:
            raise ValueError(
                f"Required config 'metric_weights' not found in {reward_kwargs_path}"
            )
        
        if 'default_reward_kwargs' not in config:
            raise ValueError(
                f"Required config 'default_reward_kwargs' not found in {reward_kwargs_path}"
            )
        
        metric_weights = config['metric_weights']

        self.llm_judge_kwargs = {}
        
        self.llm_judge_kwargs['default_reward_kwargs'] = config['default_reward_kwargs']
        
        if 'delta_reward_kwargs' in config:
            self.llm_judge_kwargs['delta_reward_kwargs'] = config['delta_reward_kwargs']
        
        self.tokenizer = tokenizer
        if 'num_examine' in config:
            self.num_examine = config['num_examine']
        else:
            self.num_examine = num_examine
        if 'reward_fn_key' in config:
            self.reward_fn_key = config['reward_fn_key']
        else:
            self.reward_fn_key = reward_fn_key
        if 'normalize_by_data_source' in config:
            self.normalize_by_data_source = config['normalize_by_data_source']
        else:
            self.normalize_by_data_source = normalize_by_data_source
        self.compute_score = compute_score or default_compute_score

        self.metric_weights = metric_weights

        self.metrics = list(self.metric_weights.keys())

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._compute_rewards_async(data, return_dict))
        finally:
            loop.close()

    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        data_source = data.non_tensor_batch["data_source"]
        ground_truth = data.non_tensor_batch["ground_truth"]
        extra_info = data.non_tensor_batch["extra_info"]

        turn_scores = data.non_tensor_batch.get("turn_scores", [])
        
        batch_size = len(data_source)
        for i in range(batch_size):
            if i < len(turn_scores):
                if not isinstance(extra_info[i], dict):
                    extra_info[i] = {}
                extra_info[i]["turn_scores"] = turn_scores[i]
            else:
                if not isinstance(extra_info[i], dict):
                    extra_info[i] = {}
                extra_info[i]["turn_scores"] = []


        message_lst = data.non_tensor_batch["messages"]

        batch_size = len(data_source)
        
        if len(message_lst) != batch_size:
            logger.error(
                f"Length mismatch: message_lst length ({len(message_lst)}) != batch_size ({batch_size}). "
                f"This may cause index errors. Using min length for safety."
            )
            actual_size = min(len(message_lst), batch_size)
        else:
            actual_size = batch_size
        
        num_repeat_rollouts_list = []
        for i in range(actual_size):
            try:
                if i < len(message_lst) and isinstance(message_lst[i], dict) and "messages" in message_lst[i]:
                    num_rollouts = len(message_lst[i]["messages"])
                else:
                    num_rollouts = 0
                    if i < len(message_lst):
                        logger.warning(
                            f"Sample {i}: message_lst[{i}] is not a dict or missing 'messages' key. "
                            f"Setting num_rollouts to 0."
                        )
            except (IndexError, KeyError, TypeError) as e:
                logger.warning(f"Error accessing message_lst[{i}]: {e}. Setting num_rollouts to 0.")
                num_rollouts = 0
            num_repeat_rollouts_list.append(num_rollouts)
        
        if len(num_repeat_rollouts_list) < batch_size:
            num_repeat_rollouts_list.extend([0] * (batch_size - len(num_repeat_rollouts_list)))
            logger.warning(
                f"message_lst length ({len(message_lst)}) < batch_size ({batch_size}). "
                f"Filling remaining samples with 0 rollouts."
            )
        
        if len(set(num_repeat_rollouts_list)) > 1:
            logger.warning(
                f"Warning: Inconsistent rollout counts detected: {num_repeat_rollouts_list}. "
                f"Will process each sample separately based on its own rollout count."
            )
        
        valid_sample_indices = [i for i in range(batch_size) if i < len(num_repeat_rollouts_list) and num_repeat_rollouts_list[i] > 0]
        invalid_sample_indices = [i for i in range(batch_size) if i >= len(num_repeat_rollouts_list) or num_repeat_rollouts_list[i] == 0]
        
        if invalid_sample_indices:
            logger.warning(
                f"Found {len(invalid_sample_indices)} samples with 0 rollouts (indices: {invalid_sample_indices}). "
                f"These samples will receive 0 reward, while other samples will be processed normally."
            )
        
        scores = torch.zeros(batch_size, dtype=torch.float32, device=prompt_ids.device)
        scores_by_metrics = {
            metric: torch.zeros(batch_size, dtype=torch.float32, device=prompt_ids.device)
            for metric in self.metrics
        }
        weighted_scores_by_metrics = {
            metric: torch.zeros(batch_size, dtype=torch.float32, device=prompt_ids.device)
            for metric in self.metrics
        }
        mean_weighted_scores_by_metrics = {metric: 0.0 for metric in self.metrics}
        invalid_samples_mask = torch.zeros(batch_size, dtype=torch.bool, device=prompt_ids.device)
        
        if valid_sample_indices:
            flattened_data_sources = []
            flattened_ground_truths = []
            flattened_extra_infos = []
            flattened_messages = []
            flattened_sample_indices = []
            
            for sample_idx in valid_sample_indices:
                if sample_idx >= batch_size:
                    logger.error(f"Sample index {sample_idx} >= batch_size {batch_size}. Skipping.")
                    continue
                if sample_idx >= len(num_repeat_rollouts_list):
                    logger.error(f"Sample index {sample_idx} >= num_repeat_rollouts_list length {len(num_repeat_rollouts_list)}. Skipping.")
                    continue
                
                num_rollouts = num_repeat_rollouts_list[sample_idx]
                if num_rollouts <= 0:
                    continue
                
                if sample_idx >= len(message_lst):
                    logger.error(f"Sample index {sample_idx} >= message_lst length {len(message_lst)}. Skipping.")
                    continue
                if not isinstance(message_lst[sample_idx], dict) or "messages" not in message_lst[sample_idx]:
                    logger.error(f"message_lst[{sample_idx}] is not a dict or missing 'messages' key. Skipping.")
                    continue
                
                if sample_idx >= len(data_source) or sample_idx >= len(ground_truth) or sample_idx >= len(extra_info):
                    logger.error(
                        f"Sample index {sample_idx} out of range: "
                        f"data_source={len(data_source)}, ground_truth={len(ground_truth)}, extra_info={len(extra_info)}. Skipping."
                    )
                    continue
                
                messages = message_lst[sample_idx]["messages"]
                if len(messages) < num_rollouts:
                    logger.warning(
                        f"Sample {sample_idx}: messages length ({len(messages)}) < num_rollouts ({num_rollouts}). "
                        f"Using actual length {len(messages)}."
                    )
                    num_rollouts = len(messages)
                
                for rollout_idx in range(num_rollouts):
                    if rollout_idx >= len(messages):
                        logger.warning(f"Sample {sample_idx}: rollout_idx {rollout_idx} >= messages length {len(messages)}. Skipping.")
                        continue
                    
                    flattened_data_sources.append(data_source[sample_idx])
                    flattened_ground_truths.append(ground_truth[sample_idx])
                    flattened_extra_infos.append(extra_info[sample_idx])
                    flattened_messages.append(messages[rollout_idx])
                    flattened_sample_indices.append(sample_idx)
            
            tasks = [
                self.compute_score(
                    flattened_data_sources[i],
                    flattened_messages[i],
                    flattened_ground_truths[i],
                    flattened_extra_infos[i],
                    self.metrics,
                    **self.llm_judge_kwargs,
                )
                for i in range(len(flattened_data_sources))
            ]
            score_dicts = await asyncio.gather(*tasks)

            sample_scores_dict = {sample_idx: [] for sample_idx in valid_sample_indices}
            for i, sample_idx in enumerate(flattened_sample_indices):
                sample_scores_dict[sample_idx].append(score_dicts[i])
            
            for sample_idx in valid_sample_indices:
                num_rollouts = num_repeat_rollouts_list[sample_idx]
                sample_score_dicts = sample_scores_dict[sample_idx]
                
                sample_scores_by_metrics = {
                    metric: torch.stack([score_dict[metric] for score_dict in sample_score_dicts])
                    .sum(dim=0)
                    for metric in self.metrics
                }
                
                sample_weighted_scores_by_metrics = {
                    metric: torch.clamp(
                        sample_scores_by_metrics[metric] * self.metric_weights[metric] / num_rollouts,
                        min=-1.0,
                        max=1.0,
                    )
                    for metric in self.metrics
                }
                
                sample_score = torch.stack([sample_weighted_scores_by_metrics[metric] for metric in self.metrics]).sum(dim=0)
                
                scores[sample_idx] = sample_score
                for metric in self.metrics:
                    scores_by_metrics[metric][sample_idx] = sample_scores_by_metrics[metric]
                    weighted_scores_by_metrics[metric][sample_idx] = sample_weighted_scores_by_metrics[metric]
            
            if len(valid_sample_indices) > 0:
                mean_weighted_scores_by_metrics = {
                    metric: weighted_scores_by_metrics[metric][valid_sample_indices].mean().item()
                    for metric in self.metrics
                }
        else:
            logger.error(
                f"All samples have 0 rollouts. This usually indicates generation failure or invalid messages. "
                f"batch_size={batch_size}, message_lst length={len(message_lst)}, "
                f"num_repeat_rollouts_list={num_repeat_rollouts_list}"
            )
            score_dicts = []

        print("Scores:", scores, mean_weighted_scores_by_metrics)

        if WANDB_AVAILABLE and wandb.run is not None:
            global_steps = data.meta_info.get("global_steps", None)
            
            wandb_log_dict = {}
            for metric in self.metrics:
                raw_scores = scores_by_metrics[metric]
                if isinstance(raw_scores, torch.Tensor):
                    if len(valid_sample_indices) > 0:
                        valid_raw_scores_normalized = []
                        for sample_idx in valid_sample_indices:
                            num_rollouts = num_repeat_rollouts_list[sample_idx]
                            if num_rollouts > 0:
                                normalized_score = raw_scores[sample_idx].item() / num_rollouts
                                valid_raw_scores_normalized.append(normalized_score)
                        
                        if len(valid_raw_scores_normalized) > 0:
                            valid_raw_scores_tensor = torch.tensor(valid_raw_scores_normalized)
                            wandb_log_dict[f"reward/{metric}_raw_mean"] = valid_raw_scores_tensor.mean().item()
                            wandb_log_dict[f"reward/{metric}_raw_min"] = valid_raw_scores_tensor.min().item()
                            wandb_log_dict[f"reward/{metric}_raw_max"] = valid_raw_scores_tensor.max().item()
                        else:
                            wandb_log_dict[f"reward/{metric}_raw_mean"] = 0.0
                            wandb_log_dict[f"reward/{metric}_raw_min"] = 0.0
                            wandb_log_dict[f"reward/{metric}_raw_max"] = 0.0
                    else:
                        wandb_log_dict[f"reward/{metric}_raw_mean"] = 0.0
                        wandb_log_dict[f"reward/{metric}_raw_min"] = 0.0
                        wandb_log_dict[f"reward/{metric}_raw_max"] = 0.0
                else:
                    raw_mean_score = float(raw_scores) if isinstance(raw_scores, (int, float)) else 0.0
                    wandb_log_dict[f"reward/{metric}_raw_mean"] = raw_mean_score
                    wandb_log_dict[f"reward/{metric}_raw_min"] = raw_mean_score
                    wandb_log_dict[f"reward/{metric}_raw_max"] = raw_mean_score
                
                weighted_scores = weighted_scores_by_metrics[metric]
                if isinstance(weighted_scores, torch.Tensor):
                    valid_weighted_scores = weighted_scores
                    if len(valid_weighted_scores) > 0:
                        wandb_log_dict[f"reward/{metric}_weighted_mean"] = valid_weighted_scores.mean().item()
                        wandb_log_dict[f"reward/{metric}_weighted_min"] = valid_weighted_scores.min().item()
                        wandb_log_dict[f"reward/{metric}_weighted_max"] = valid_weighted_scores.max().item()
                    else:
                        wandb_log_dict[f"reward/{metric}_weighted_mean"] = 0.0
                        wandb_log_dict[f"reward/{metric}_weighted_min"] = 0.0
                        wandb_log_dict[f"reward/{metric}_weighted_max"] = 0.0
                else:
                    weighted_mean_score = float(weighted_scores) if isinstance(weighted_scores, (int, float)) else 0.0
                    wandb_log_dict[f"reward/{metric}_weighted_mean"] = weighted_mean_score
                    wandb_log_dict[f"reward/{metric}_weighted_min"] = weighted_mean_score
                    wandb_log_dict[f"reward/{metric}_weighted_max"] = weighted_mean_score
            
            if isinstance(scores, torch.Tensor):
                valid_scores = scores
                if len(valid_scores) > 0:
                    wandb_log_dict["reward/total_mean"] = valid_scores.mean().item()
                    wandb_log_dict["reward/total_min"] = valid_scores.min().item()
                    wandb_log_dict["reward/total_max"] = valid_scores.max().item()
                else:
                    wandb_log_dict["reward/total_mean"] = 0.0
                    wandb_log_dict["reward/total_min"] = 0.0
                    wandb_log_dict["reward/total_max"] = 0.0
            else:
                wandb_log_dict["reward/total_mean"] = float(scores) if isinstance(scores, (int, float)) else 0.0
                wandb_log_dict["reward/total_min"] = wandb_log_dict["reward/total_mean"]
                wandb_log_dict["reward/total_max"] = wandb_log_dict["reward/total_mean"]
            
            if global_steps is not None:
                wandb.log(wandb_log_dict, step=global_steps)
            else:
                logger.warning(
                    "global_steps not found in data.meta_info. "
                    "wandb.log will use auto-incrementing step, which may cause step mismatch with training loop."
                )
                wandb.log(wandb_log_dict)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            if scores[i].item() < -50.0:
                continue
            
            valid_len = valid_response_length[i].item()
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = scores[i]
    
        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor






# async def reward_func(
#     data_source, messages, ground_truth, extra_info, metrics, **kwargs
# ) -> dict[str, torch.Tensor]:
#     pass


#     turn_scores = extra_info.get("turn_scores", [])
    
#     if turn_scores and len(turn_scores) > 0:
#         if -999.0 in [float(score) for score in turn_scores if score is not None]:
#             aggregated_score = -999.0
#         else:
#             scores_list = [float(score) for score in turn_scores if score is not None]
#             if scores_list:
#                 aggregated_score = sum(scores_list)
#             else:
#                 aggregated_score = 0.0
#     else:
#         aggregated_score = 0.0
    
#     rewards = {metric: torch.tensor(aggregated_score, dtype=torch.float32) for metric in metrics}
    
#     return rewards

