
import asyncio
import importlib.util
import os
import sys
from typing import Any, Callable, Optional

import litellm
import torch
from transformers import PreTrainedTokenizer

import json


async def compute_score(
    data_source, messages, ground_truth, extra_info, **kwargs
) -> dict[str, torch.Tensor]:
    """
    Reward computation function based on turn_scores (async version)
    
    This function extracts turn_scores (scores for each conversation turn) from extra_info,
    and returns them as rewards. turn_scores are accumulated rewards returned by generate_response during interaction.
    
    Args:
        data_source (str): Data source identifier, used to distinguish different datasets or task types
        messages (list): Complete conversation message list, containing all interaction messages from user and assistant
        ground_truth (str): True answer or reference standard (not used in this function)
        extra_info (dict): Additional context information, must contain 'turn_scores' key
        metrics (list[str]): List of metrics to compute, e.g., ['accuracy', 'interactivity', 'token_amount']
        **kwargs: Other parameters (not used in this function)
    
    Returns:
        dict[str, torch.Tensor]: Dictionary with metric names as keys and reward scores (torch.Tensor) as values
        Each metric returns the same aggregated turn_scores value (sum)
        Example: {'accuracy': tensor(1.0), 'interactivity': tensor(1.0), 'token_amount': tensor(1.0)}
    
    Workflow:
    1. Extract turn_scores from extra_info
    2. If turn_scores exists and is non-empty, calculate their sum
    3. If turn_scores doesn't exist or is empty, return 0.0
    4. Return the same aggregated value for each metric
    """
    
    turn_scores = extra_info.get("turn_scores", [])
    
    # Check if turn_scores contains -999 (indicates invalid sample, not participating in training)
    # -999 is a special value returned by generate_response in ResearchGAN_interation.py
    if turn_scores and len(turn_scores) > 0:
        # Check if -999 is included
        if -999.0 in [float(score) for score in turn_scores if score is not None]:
            # If -999 is included, return -999 as marker, indicating this sample doesn't participate in loss calculation
            aggregated_score = -999.0
        else:
            # Convert turn_scores to float list (handle possible type inconsistencies)
            scores_list = [float(score) for score in turn_scores if score is not None]
            if scores_list:
                aggregated_score = sum(scores_list)
            else:
                aggregated_score = 0.0
    else:
        aggregated_score = 0.0
    
    # Return the same aggregated value for each metric
    rewards = aggregated_score#{metric: torch.tensor(aggregated_score, dtype=torch.float32) for metric in metrics}
    
    return rewards